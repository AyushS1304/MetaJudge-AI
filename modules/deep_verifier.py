"""
modules/deep_verifier.py — Deep Fact Verification
--------------------------------------------------
Called when Groq returns INSUFFICIENT_EVIDENCE.
Uses three escalating strategies:

1. Broader web searches (Papers With Code, Semantic Scholar, direct queries)
2. LangChain ArxivAPIWrapper — fetches full paper text including results
3. Gemini 2.0 Flash — reads the PDF + text and verifies the specific claim

Key design: gemini_key passed explicitly every call — never cached at module level.
"""

import os, re, sys, json, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

_PDF_CACHE:  dict[str, bytes] = {}
_TEXT_CACHE: dict[str, str]   = {}


# ── 1. Broader searches ────────────────────────────────────────────────────

def _broader_searches(fact: str, context: str) -> str:
    """Run 4 targeted search strategies to find metric evidence."""
    try:
        from modules.retriever import _search_web, _search_arxiv, format_evidence_block
    except Exception:
        return ""

    all_ev, seen = [], set()

    def add(evs):
        for e in evs:
            u = e.get("url", "")
            if u and u not in seen:
                seen.add(u); all_ev.append(e)

    nums  = re.findall(r'\d+\.?\d*\s*%?', fact)
    words = re.findall(r'\b[A-Z][a-zA-Z0-9\-]{2,}\b', fact + " " + context)

    # Strategy 1 — direct fact
    add(_search_web(fact[:100]))

    # Strategy 2 — model + benchmark
    if words and nums:
        add(_search_web(f"{words[0]} benchmark results {' '.join(nums[:2])}"))

    # Strategy 3 — Papers With Code leaderboard
    bench_kw = ["squad","mnli","mmlu","glue","bleu","wmt","imagenet","hellaswag","arc"]
    bench = next((b for b in bench_kw if b in fact.lower()), "")
    if words and bench:
        add(_search_web(f"{words[0]} {bench} paperswithcode leaderboard state of the art"))

    # Strategy 4 — arXiv
    add(_search_arxiv(" ".join(words[:3])))

    return format_evidence_block(all_ev[:8]) if all_ev else ""


# ── 2. LangChain ArxivAPIWrapper ───────────────────────────────────────────

def _fetch_via_langchain(arxiv_id: str) -> str:
    """Fetch full paper text via LangChain (includes results sections)."""
    if arxiv_id in _TEXT_CACHE:
        return _TEXT_CACHE[arxiv_id]
    try:
        from langchain_community.utilities import ArxivAPIWrapper
        wrapper = ArxivAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=10000,
            load_all_available_meta=True,
        )
        result = wrapper.run(f"id:{arxiv_id}")
        if result and len(result) > 200:
            _TEXT_CACHE[arxiv_id] = result
            print(f"  [LangChain] Fetched {len(result)} chars for arXiv:{arxiv_id}")
            return result
    except Exception as e:
        print(f"  [LangChain] {e}")
    return ""


def _fetch_pdf(arxiv_id: str) -> bytes | None:
    """Download PDF bytes from arXiv."""
    if arxiv_id in _PDF_CACHE:
        return _PDF_CACHE[arxiv_id]
    try:
        r = requests.get(f"https://arxiv.org/pdf/{arxiv_id}",
                         timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            _PDF_CACHE[arxiv_id] = r.content
            print(f"  [PDF] Downloaded {len(r.content)//1024}KB for arXiv:{arxiv_id}")
            return r.content
    except Exception as e:
        print(f"  [PDF] {e}")
    return None


# ── 3. Gemini judge ────────────────────────────────────────────────────────

GEMINI_PROMPT = """You are an expert AI/ML research fact-checker given the FULL paper to read.

arXiv paper ID: {arxiv_id}
CLAIM TO VERIFY: "{claim}"

STEP 1 — IDENTIFY WHAT IS BEING CLAIMED:
Extract the specific factual element:
- A numeric metric (score, accuracy, BLEU, F1, perplexity)?
- A hyperparameter value (rank, layers, heads, learning rate, batch size)?
- A date or year?
- An author or institution name?

STEP 2 — SEARCH THE ENTIRE PAPER:
Look through EVERY section — do not stop at the abstract:
- Section 4/5/6 Results or Experiments
- ALL tables (Table 1, Table 2, Table 3... check every one)
- The main results table which compares methods
- Ablation study tables
- Appendix tables
- The experimental setup / implementation details section (hyperparameters are HERE)

For hyperparameters like "rank": search for "r =", "rank =", "r=4", "r=8", "rank of"
For benchmark scores: search for the benchmark name + number
For dates: check "Published:" or "Submitted:" metadata

STEP 3 — COMPARE VALUES:
Claimed value vs actual value found in paper:
- Different values for the same thing → CONTRADICTED
- Same value confirmed → SUPPORTED
- Genuinely not found anywhere after thorough search → INSUFFICIENT_EVIDENCE

Return ONLY valid JSON (no markdown):
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
  "reasoning": "Step 2 found [actual value] in [Table N / Section X] vs claimed [Y] — CONTRADICTED/SUPPORTED",
  "evidence_quote": "Exact sentence or table entry from paper showing the real value",
  "evidence_source": "https://arxiv.org/abs/{arxiv_id}"
}}"""


def _call_gemini(fact: str, arxiv_id: str,
                 paper_text: str, pdf_bytes: bytes | None,
                 extra_ev: str, gemini_key: str) -> dict | None:
    """Call Gemini Flash with all available evidence."""
    if not gemini_key:
        return None
    try:
        from google import genai
        from google.genai import types

        client   = genai.Client(api_key=gemini_key)
        parts    = []
        pdf_used = False

        if pdf_bytes:
            parts.append(types.Part.from_bytes(
                data=pdf_bytes, mime_type="application/pdf"))
            parts.append(
                f"[Full paper PDF uploaded above]\n\n"
                f"Additional text evidence:\n{paper_text[:3000]}\n\n"
                f"Web evidence:\n{extra_ev[:1500]}\n\n"
            )
            pdf_used = True
        else:
            ctx = ""
            if paper_text: ctx += f"Paper text (LangChain):\n{paper_text}\n\n"
            if extra_ev:   ctx += f"Web evidence:\n{extra_ev[:2000]}\n\n"
            if ctx: parts.append(ctx)

        parts.append(GEMINI_PROMPT.format(
            arxiv_id=arxiv_id or "unknown",
            claim=fact,
        ))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=parts,
        )

        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw, re.DOTALL)
            if m:
                result = json.loads(m.group())
            else:
                v = ("CONTRADICTED" if "CONTRADICTED" in raw.upper()
                     else "SUPPORTED" if "SUPPORTED" in raw.upper()
                     else "INSUFFICIENT_EVIDENCE")
                result = {
                    "verdict": v,
                    "reasoning": raw[:200],
                    "evidence_quote": "",
                    "evidence_source": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                }

        result["gemini_used"] = True
        result["pdf_used"]    = pdf_used
        print(f"  [Gemini] {result.get('verdict')} — {result.get('reasoning','')[:80]}")
        return result

    except Exception as e:
        print(f"  [Gemini] {type(e).__name__}: {e}")
        return None


# ── Main entry point ───────────────────────────────────────────────────────

def deep_verify(
    fact: str,
    context: str,
    original_evidence: str,
    known_papers: dict,
    gemini_key: str = "",
    verbose: bool = True,
) -> dict | None:
    """
    Full deep verification pipeline.
    gemini_key passed explicitly — never read from module-level variable.

    Returns verdict dict or None.
    """
    # Resolve API key — argument takes priority, then env
    key = gemini_key or os.environ.get("GEMINI_API_KEY", "")

    # Find arXiv ID
    search_text = (fact + " " + context).lower()
    arxiv_id    = None
    for keyword, aid in known_papers.items():
        if keyword in search_text:
            arxiv_id = aid
            break

    if verbose:
        print(f"  [Deep verify] Paper: arXiv:{arxiv_id or 'unknown'} | Gemini: {'yes' if key else 'no'}")

    # Step 1: broader searches
    if verbose: print("  [Deep verify] Running broader searches...")
    extra_ev = _broader_searches(fact, context)
    if original_evidence:
        extra_ev = (extra_ev + "\n\n---\n\n" + original_evidence) if extra_ev else original_evidence

    # Step 2: LangChain full paper text
    paper_text = ""
    if arxiv_id:
        if verbose: print(f"  [Deep verify] Fetching full text via LangChain...")
        paper_text = _fetch_via_langchain(arxiv_id)

    # Step 3: PDF bytes
    pdf_bytes = None
    if arxiv_id and key:
        if verbose: print(f"  [Deep verify] Downloading PDF...")
        pdf_bytes = _fetch_pdf(arxiv_id)

    # Step 4: Gemini judge
    if key:
        if verbose: print(f"  [Deep verify] Calling Gemini Flash...")
        result = _call_gemini(fact, arxiv_id or "",
                              paper_text, pdf_bytes, extra_ev, key)
        if result:
            return result

    # Step 5: no Gemini — still return enriched search result for pipeline
    if not key and (paper_text or extra_ev):
        # Try Groq judge with enriched evidence
        try:
            from modules.judge import judge_claim
            combined = ""
            if paper_text: combined += f"PAPER TEXT:\n{paper_text}\n\n"
            if extra_ev:   combined += f"WEB EVIDENCE:\n{extra_ev}"
            result = judge_claim(fact, combined)
            result["gemini_used"] = False
            result["pdf_used"]    = False
            if verbose:
                print(f"  [Deep verify] Groq re-judge on enriched evidence: {result.get('verdict')}")
            return result
        except Exception as e:
            if verbose: print(f"  [Deep verify] Groq re-judge failed: {e}")

    return None


if __name__ == "__main__":
    from modules.retriever import KNOWN_PAPERS, retrieve_evidence, format_evidence_block
    from modules.query_generator import generate_skeptical_queries

    fact    = "The paper demonstrated results with a rank of 8, achieving 91.3% on MNLI."
    context = "LoRA was proposed by Hu et al. from Microsoft in 2022."

    queries = generate_skeptical_queries(fact)
    ev      = retrieve_evidence(queries, fact=fact, context=context)
    ev_blk  = format_evidence_block(ev)

    key = os.environ.get("GEMINI_API_KEY", "")
    result = deep_verify(fact, context, ev_blk, KNOWN_PAPERS,
                         gemini_key=key, verbose=True)
    if result:
        print(f"\nVerdict: {result.get('verdict')}")
        print(f"Reason:  {result.get('reasoning')}")
        print(f"Quote:   {result.get('evidence_quote','')[:120]}")
    else:
        print("No result")