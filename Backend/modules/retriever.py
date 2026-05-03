"""
modules/retriever.py — Step 3: Hybrid Retrieval
-------------------------------------------------
Three-source retrieval strategy:

  Source 1 — arXiv DIRECT (new, most important)
    Extracts the paper/model name from the atomic fact,
    searches arXiv by title, fetches the FULL abstract of the
    top result. This is the authoritative ground truth — the
    original paper with exact numbers, authors, and dates.

  Source 2 — arXiv ADVERSARIAL
    Searches arXiv with the adversarial query string.
    Catches related papers that might contradict the claim.

  Source 3 — DuckDuckGo WEB
    General web search for blog posts, leaderboards, press
    releases — good for dates and announcements.

Why Source 1 solves the BERT problem:
  Adversarial query "BERT actual SQuAD 2.0 score vs claimed 80.5%"
  returns generic pages. But searching arXiv for "BERT Devlin
  bidirectional transformers" returns arXiv:1810.04805 whose
  abstract contains "86.7 F1 on SQuAD 2.0" — the exact number
  that proves the hallucination.
"""

import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_ARXIV_RESULTS, MAX_WEB_RESULTS

import arxiv

# Optional PDF extraction — used when abstract doesn't have the specific value
try:
    from modules.pdf_extractor import get_paper_results as _get_pdf_results
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# PDF fetcher for full paper content (results tables, experiments)
try:
    from modules.pdf_fetcher import fetch_paper_full_content
    PDF_FETCH_AVAILABLE = True
except ImportError:
    PDF_FETCH_AVAILABLE = False


# ── Known paper aliases for direct arXiv lookup ───────────────────────────────
# Maps common names → arXiv IDs for instant exact retrieval
KNOWN_PAPERS = {
    "bert":               "1810.04805",
    "gpt-4":              "2303.08774",
    "gpt-3":              "2005.14165",
    "gpt4":               "2303.08774",
    "gpt3":               "2005.14165",
    "llama 2":            "2307.09288",
    "llama2":             "2307.09288",
    "llama":              "2302.13971",
    "attention is all":   "1706.03762",
    "transformer":        "1706.03762",
    "roberta":            "1907.11692",
    "t5":                 "1910.10683",
    "instructgpt":        "2203.02155",
    "factscore":          "2305.14251",
    "rarr":               "2210.08726",
    "self-rag":           "2310.11511",
    "cove":               "2309.11495",
    "chain-of-thought":   "2201.11903",
    "chain of thought":   "2201.11903",
    "reflexion":          "2303.11366",
    "react":              "2210.03629",
    "toolformer":         "2302.04761",
    "lora":               "2106.09685",
    "qlora":              "2304.01196",
    "dpo":                "2305.18290",
    "clip":               "2103.00020",
    "stable diffusion":   "2112.10752",
    "mistral":            "2310.06825",
    "halueval":           "2305.11747",
    "truthfulqa":         "2109.07958",
    "squad":              "1606.05250",
    "squad 2":            "1806.03822",
    "webgpt":             "2112.09332",
    "blip":               "2301.13688",
    "llava":              "2304.08485",
}


def _fetch_by_arxiv_id(arxiv_id: str) -> dict | None:
    """Fetch a paper directly by its arXiv ID — guaranteed exact match."""
    try:
        results = list(arxiv.Search(
            id_list=[arxiv_id], max_results=1
        ).results())
        if not results:
            return None
        p = results[0]
        # Use FULL abstract (no truncation) — this is where the exact numbers are
        pub_date = p.published.strftime('%Y-%m-%d')
        pub_year = p.published.year
        snippet = (
            f"Title: {p.title}\n"
            f"Authors: {', '.join(a.name for a in p.authors[:6])}\n"
            f"Published: {pub_date} (Year: {pub_year})\n"
            f"FACT-CHECK NOTE: This paper was first published/introduced/released in {pub_year}, not any other year.\n"
            f"arXiv ID: {arxiv_id}\n"
            f"Abstract: {p.summary}"
        )
        return {
            "source":  "arxiv_direct",
            "url":     f"https://arxiv.org/abs/{arxiv_id}",
            "title":   p.title,
            "snippet": snippet,
        }
    except Exception as e:
        print(f"  [arXiv direct warn] {e}")
        return None


def _direct_arxiv_lookup(fact: str, context: str = "") -> list[dict]:
    """
    Try to find the exact paper being referenced in the fact.
    Uses both the atomic fact AND the full summary context to identify the paper.
    This fixes cases where the atomicizer strips the model name from a fact
    (e.g. "The model achieved 28.4 BLEU" loses "transformer" as the subject).
    """
    # Search both the fact and the full context for known paper keywords
    search_text = (fact + " " + context).lower()
    results = []
    seen_ids = set()

    # Step 1: check known paper aliases against combined text
    for keyword, arxiv_id in KNOWN_PAPERS.items():
        if keyword in search_text and arxiv_id not in seen_ids:
            seen_ids.add(arxiv_id)
            paper = _fetch_by_arxiv_id(arxiv_id)
            if paper:
                results.append(paper)

    # Step 2: extract capitalised terms from BOTH fact and context
    combined = fact + " " + context
    caps = re.findall(r'\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z0-9][A-Za-z0-9\-]*)?\b', combined)
    # Deduplicate while preserving order
    seen_caps = set()
    unique_caps = [c for c in caps if not (c in seen_caps or seen_caps.add(c))]

    for term in unique_caps[:5]:
        if len(term) < 3:
            continue
        try:
            search = arxiv.Search(
                query=f'ti:"{term}"',
                max_results=2,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for p in search.results():
                aid = p.entry_id.split("/abs/")[-1].split("v")[0]
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    pub_date = p.published.strftime('%Y-%m-%d')
                    pub_year = p.published.year
                    snippet = (
                        f"Title: {p.title}\n"
                        f"Authors: {', '.join(a.name for a in p.authors[:6])}\n"
                        f"Published: {pub_date} (Year: {pub_year})\n"
                        f"arXiv ID: {aid}\n"
                        f"NOTE: This paper was published in {pub_year}.\n"
                        f"Abstract: {p.summary}"
                    )
                    results.append({
                        "source":  "arxiv_direct",
                        "url":     p.entry_id,
                        "title":   p.title,
                        "snippet": snippet,
                    })
        except Exception:
            pass

    return results[:3]


def _search_arxiv(query: str) -> list[dict]:
    """Search arXiv with the adversarial query."""
    results = []
    try:
        search = arxiv.Search(
            query=query,
            max_results=MAX_ARXIV_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for paper in search.results():
            snippet = (
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors[:4])}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"Abstract: {paper.summary[:800]}"
            )
            results.append({
                "source":  "arxiv",
                "url":     paper.entry_id,
                "title":   paper.title,
                "snippet": snippet,
            })
    except Exception as e:
        print(f"  [arXiv warn] {e}")
    return results


def _search_web(query: str) -> list[dict]:
    """Search the web via DuckDuckGo."""
    results = []
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=MAX_WEB_RESULTS))
        for hit in hits:
            results.append({
                "source":  "web",
                "url":     hit.get("href", ""),
                "title":   hit.get("title", ""),
                "snippet": hit.get("body", ""),
            })
    except Exception as e:
        print(f"  [Web warn] {e}")
    return results


def retrieve_evidence(queries: list[str], fact: str = "", context: str = "") -> list[dict]:
    """
    Retrieve evidence using three sources IN PARALLEL:
      1. Direct arXiv lookup (by paper name / known ID)
      2. arXiv adversarial search
      3. DuckDuckGo web search
      4. Venue-specific web search (if fact mentions a conference/venue)

    All sources run concurrently — cuts retrieval time from ~4 min to ~20-30 sec.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

    tasks = []

    # Build all tasks
    if fact:
        tasks.append(("direct", None, fact))
    for q in queries:
        tasks.append(("arxiv", q, None))
        tasks.append(("web",   q, None))

    # Add PDF fetch tasks for known papers found in fact+context
    if PDF_FETCH_AVAILABLE:
        search_text = (fact + " " + context).lower()
        for keyword, arxiv_id in KNOWN_PAPERS.items():
            if keyword in search_text:
                tasks.append(("pdf", arxiv_id, fact))
                break  # one PDF fetch per call is enough

    fact_lower    = fact.lower()
    context_lower = context.lower()
    combined_lower = fact_lower + " " + context_lower

    # Extra: venue-specific search
    venue_keywords = ["icml","neurips","nips","acl","emnlp","naacl","iclr","cvpr",
                      "iccv","eccv","aaai","ijcai","conference","workshop","published at"]
    if any(v in fact_lower for v in venue_keywords):
        caps = re.findall(r'\b[A-Z][A-Za-z0-9\-]{2,}\b', context)[:3]
        if caps:
            venue_query = f"{caps[0]} paper published conference venue official proceedings"
            tasks.append(("web", venue_query, None))

    # PDF results extraction — triggered for any fact containing a number
    # Extracts results tables from the full paper, not just abstract
    if fact and re.search(r'\d', fact):
        tasks.append(("pdf", None, fact))

    # Extra: date/year-specific search
    year_keywords = ["published", "released", "introduced", "proposed", "presented", "year"]
    if any(w in fact_lower for w in year_keywords):
        caps = re.findall(r'\b[A-Z][A-Za-z0-9\-]{2,}\b', context)[:2]
        if caps:
            year_query = f"{caps[0]} paper when published year release date arxiv"
            tasks.append(("web", year_query, None))

    # Extra: method/training-specific search
    method_keywords = ["language modelling", "language modeling", "pre-training", "pretraining",
                       "training objective", "masked", "causal", "autoregressive", "architecture"]
    if any(w in fact_lower for w in method_keywords):
        caps = re.findall(r'\b[A-Z][A-Za-z0-9\-]{2,}\b', context)[:2]
        if caps:
            method_query = f"{caps[0]} training method pre-training objective technique actual"
            tasks.append(("web", method_query, None))

    raw_results = []
    _context = context  # captured for closure

    def run_task(task):
        kind, query, f = task
        try:
            if kind == "pdf":
                # query holds arxiv_id
                try:
                    from modules.pdf_extractor import get_paper_results
                    result = get_paper_results(query)
                    return [result] if result else []
                except Exception as e:
                    print(f"  [PDF warn] {e}")
                    return []
            elif kind == "direct":
                return _direct_arxiv_lookup(f, context=_context)
            elif kind == "arxiv":
                return _search_arxiv(query)
            else:
                return _search_web(query)
        except Exception:
            return []

    # Run all in parallel with a 30-second timeout per task
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_task, t): t for t in tasks}
        for future in as_completed(futures, timeout=45):
            try:
                raw_results.extend(future.result(timeout=30))
            except Exception:
                pass

    # Deduplicate — PDF first (most detailed), then direct, then rest
    all_evidence = []
    seen_urls    = set()

    # PDF content first — has results tables
    for ev in raw_results:
        if ev.get("source") == "arxiv_pdf":
            url = ev.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_evidence.append(ev)

    # Direct arXiv abstract second
    for ev in raw_results:
        if ev.get("source") == "arxiv_direct":
            url = ev.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_evidence.append(ev)

    # Everything else
    for ev in raw_results:
        if ev.get("source") not in ("arxiv_pdf", "arxiv_direct"):
            url = ev.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_evidence.append(ev)

    return all_evidence


def format_evidence_block(evidence_list: list[dict]) -> str:
    """Format evidence list into a readable block for the LLM judge."""
    if not evidence_list:
        return "No evidence retrieved."

    lines = []
    for i, ev in enumerate(evidence_list, 1):
        src_label = {
            "arxiv_pdf":    "ARXIV-PDF (full paper — results & experiments)",
            "arxiv_direct": "ARXIV-DIRECT (abstract — authoritative)",
            "arxiv":        "ARXIV",
            "web":          "WEB",
        }.get(ev["source"], ev["source"].upper())

        lines.append(
            f"[Evidence {i}] Source: {src_label} | {ev['title']}\n"
            f"URL: {ev['url']}\n"
            f"{ev['snippet']}\n"
        )
    return "\n---\n".join(lines)


if __name__ == "__main__":
    test_fact    = "BERT achieved 80.5% F1 on the SQuAD 2.0 benchmark."
    test_queries = [
        "BERT actual SQuAD 2.0 score official result",
        "SQuAD 2.0 leaderboard BERT correct performance",
    ]
    print("=== RETRIEVER TEST ===\n")
    print(f"Fact: {test_fact}\n")
    evidence = retrieve_evidence(test_queries, fact=test_fact)
    print(f"Retrieved {len(evidence)} evidence items.\n")
    for ev in evidence:
        print(f"[{ev['source'].upper()}] {ev['title']}")
        print(f"  {ev['snippet'][:200]}\n")