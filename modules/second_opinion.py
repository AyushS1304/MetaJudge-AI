"""
modules/second_opinion.py — Gemini Second Opinion + Metric Web Search
----------------------------------------------------------------------
Full fallback chain when Groq returns INSUFFICIENT_EVIDENCE:
  1. Retry with broader web + arXiv searches
  2. Search Papers With Code / web for exact metric values  
  3. Use Gemini Flash to read the full PDF
  4. Ask Gemini as independent second judge

Key fix: API key is read fresh inside each function call,
not at import time — so sidebar key entry always works.
"""

import os, re, sys, requests, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

_PDF_CACHE: dict[str, bytes] = {}


def _get_gemini_key() -> str:
    """Always read fresh — never cached at import time."""
    return os.environ.get("GEMINI_API_KEY", "")


GEMINI_JUDGE_PROMPT = """You are a precise fact-checker for AI/ML research papers.

CLAIM TO VERIFY: {claim}

EVIDENCE (includes abstracts, web results, and possibly full paper text):
{evidence}

Your job:
1. Find the ACTUAL value for whatever metric/fact is being claimed
2. Compare it to the claimed value
3. Numbers must match exactly: 8 ≠ 6, 91.3% ≠ 90.7%, 2022 ≠ 2021

VERDICTS:
- CONTRADICTED: Evidence gives a DIFFERENT specific value for the same thing
- SUPPORTED: Evidence explicitly confirms the EXACT claimed value  
- INSUFFICIENT_EVIDENCE: Evidence has nothing relevant at all

Return ONLY valid JSON:
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
  "reasoning": "Claim says [X], evidence says [Y] — therefore [verdict]",
  "evidence_quote": "Exact verbatim quote from evidence",
  "evidence_source": "URL or source name"
}}"""


def _download_pdf(arxiv_id: str) -> bytes | None:
    """Download PDF bytes from arXiv with caching."""
    if arxiv_id in _PDF_CACHE:
        return _PDF_CACHE[arxiv_id]
    try:
        r = requests.get(
            f"https://arxiv.org/pdf/{arxiv_id}",
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (research bot)"}
        )
        if r.status_code == 200:
            _PDF_CACHE[arxiv_id] = r.content
            return r.content
    except Exception as e:
        print(f"  [PDF download] {e}")
    return None


def _search_papers_with_code(fact: str, context: str) -> str:
    """
    Search Papers With Code for exact benchmark results.
    PWC has structured tables of model performance on benchmarks.
    """
    results = []

    # Extract model name and benchmark from fact
    try:
        from modules.retriever import _search_web

        # Build targeted PWC query
        nums  = re.findall(r'\d+\.?\d*\s*%?', fact)
        words = re.findall(r'\b[A-Z][a-zA-Z0-9\-]{2,}\b', fact + " " + context)
        model = words[0] if words else ""
        bench_keywords = ["squad","mnli","mmlu","glue","bleu","wmt","imagenet",
                          "coco","hellaswag","arc","winogrande","truthfulqa"]
        bench = next((k for k in bench_keywords if k in fact.lower()), "")

        if model and bench:
            q1 = f"{model} {bench} results paperswithcode leaderboard"
            results.extend(_search_web(q1))

        if model and nums:
            q2 = f"{model} actual performance benchmark results {' '.join(nums[:2])}"
            results.extend(_search_web(q2))

        # Also try semantic scholar
        if model:
            q3 = f"site:semanticscholar.org {model} results"
            results.extend(_search_web(q3))

    except Exception as e:
        print(f"  [PWC search] {e}")

    if not results:
        return ""

    from modules.retriever import format_evidence_block
    return format_evidence_block(results[:5])


def _retry_searches(fact: str, context: str) -> str:
    """Try multiple search strategies for better evidence coverage."""
    from modules.retriever import _search_web, _search_arxiv, format_evidence_block

    all_ev = []
    seen   = set()

    def add(evs):
        for e in evs:
            u = e.get("url", "")
            if u and u not in seen:
                seen.add(u)
                all_ev.append(e)

    # Strategy 1: fact as direct query
    add(_search_web(fact[:100]))
    add(_search_arxiv(fact[:80]))

    # Strategy 2: extract key terms and numbers
    nums  = re.findall(r'\d+\.?\d*\s*%?', fact)
    words = [w for w in fact.split() if len(w) > 4 and w[0].isupper()]
    key_q = " ".join(words[:3] + nums[:2])
    if key_q.strip():
        add(_search_web(key_q))
        add(_search_arxiv(" ".join(words[:3])))

    # Strategy 3: paper-specific query
    paper_m = re.search(r'[\'"]([^\'"]{5,60})[\'"]', context)
    if paper_m:
        add(_search_web(f"{paper_m.group(1)} results benchmark accuracy"))
        add(_search_arxiv(paper_m.group(1)))

    # Strategy 4: Papers With Code specific
    pwc = _search_papers_with_code(fact, context)
    if pwc:
        all_ev.append({
            "source": "web", "url": "paperswithcode.com",
            "title":  "Papers With Code benchmark results",
            "snippet": pwc,
        })

    return format_evidence_block(all_ev) if all_ev else ""


def _gemini_judge(fact: str, evidence: str,
                  arxiv_id: str = None) -> dict | None:
    """
    Ask Gemini Flash to judge a claim.
    Optionally uploads the full PDF for deeper analysis.
    Key is read fresh every call.
    """
    key = _get_gemini_key()
    if not key:
        print("  [Gemini] No GEMINI_API_KEY in environment")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        parts  = []

        # Try to add PDF if we have an arxiv_id
        pdf_used = False
        if arxiv_id:
            pdf_bytes = _download_pdf(arxiv_id)
            if pdf_bytes:
                parts.append(types.Part.from_bytes(
                    data=pdf_bytes, mime_type="application/pdf"))
                parts.append(
                    "Above is the full research paper including all results tables.\n\n"
                    f"Also consider this additional evidence:\n{evidence[:2000]}\n\n"
                )
                pdf_used = True
                print(f"  [Gemini] PDF uploaded ({len(pdf_bytes)//1024}KB)")

        if not pdf_used:
            parts.append(f"Evidence:\n{evidence}\n\n")

        parts.append(GEMINI_JUDGE_PROMPT.format(
            claim=fact,
            evidence="[See above]" if pdf_used else evidence[:3000]
        ))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=parts
        )

        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)
        result["gemini_used"] = True
        result["pdf_used"]    = pdf_used
        return result

    except json.JSONDecodeError as e:
        print(f"  [Gemini] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  [Gemini] Error: {e}")
        return None


def get_second_opinion(
    fact: str,
    context: str,
    original_evidence: str,
    known_papers: dict,
    verbose: bool = True,
) -> dict | None:
    """
    Full fallback chain when Groq returns INSUFFICIENT_EVIDENCE.

    Steps:
    1. Retry searches (broader strategies + Papers With Code)
    2. Gemini with PDF if paper is known
    3. Gemini text-only if no PDF

    Returns verdict dict or None.
    Key is always read fresh — sidebar entry always works.
    """
    key = _get_gemini_key()

    # Find arxiv ID for this paper
    search_text = (fact + " " + context).lower()
    arxiv_id    = None
    for keyword, aid in known_papers.items():
        if keyword in search_text:
            arxiv_id = aid
            break

    # Step 1: retry with better searches
    if verbose:
        print("  [Second opinion] Running broader searches + Papers With Code...")
    retry_ev = _retry_searches(fact, context)

    combined = retry_ev
    if original_evidence:
        combined = combined + "\n\n---\n\n" + original_evidence if combined else original_evidence

    # Step 2 & 3: Gemini judge (with or without PDF)
    if key:
        if verbose:
            mode = f"PDF (arXiv:{arxiv_id})" if arxiv_id else "text-only"
            print(f"  [Second opinion] Gemini Flash judging ({mode})...")
        result = _gemini_judge(fact, combined, arxiv_id=arxiv_id)
        if result:
            return result
    else:
        if verbose:
            print("  [Second opinion] No Gemini key — skipping Gemini judge")
        # Even without Gemini, return the retry search results as a text-based verdict
        # so the pipeline can try to judge with richer evidence
        return None

    return None


if __name__ == "__main__":
    from modules.retriever import KNOWN_PAPERS, retrieve_evidence, format_evidence_block
    from modules.query_generator import generate_skeptical_queries

    fact    = "The paper demonstrated results with a rank of 8, achieving 91.3% accuracy on MNLI."
    context = "LoRA was proposed by Hu et al. from Microsoft in 2022."

    queries  = generate_skeptical_queries(fact)
    evidence = retrieve_evidence(queries, fact=fact, context=context)
    ev_block = format_evidence_block(evidence)

    result = get_second_opinion(fact, context, ev_block, KNOWN_PAPERS)
    if result:
        print(f"\nVerdict:  {result.get('verdict')}")
        print(f"Reasoning: {result.get('reasoning')}")
        print(f"PDF used:  {result.get('pdf_used')}")
    else:
        print("No second opinion available")