"""
modules/retriever.py — Step 3: Hybrid Retrieval
-------------------------------------------------
Three-source retrieval strategy with caching and rate limiting:

  Source 1 — arXiv DIRECT (most important)
    Extracts the paper/model name from the atomic fact,
    searches arXiv by title, fetches the FULL abstract.

  Source 2 — arXiv ADVERSARIAL
    Searches arXiv with the adversarial query string.

  Source 3 — DuckDuckGo WEB
    General web search for blog posts, leaderboards, etc.

All results are cached in SQLite to avoid hitting arXiv rate limits.
A per-run cap of MAX_ARXIV_CALLS_PER_RUN prevents 429 errors.
"""

import logging
import re
import threading
import time

import requests as _requests

from config import (
    ARXIV_DELAY_SECONDS,
    MAX_ARXIV_CALLS_PER_RUN,
    RESULTS_PER_QUERY,
    RETRIEVAL_TIMEOUT_SECONDS,
)
from modules.cache_layer import get_cached, set_cached

import arxiv

# Optional PDF extraction
try:
    from modules.pdf_extractor import get_paper_results as _get_pdf_results
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Thread-safe arXiv call counter
_arxiv_counter = threading.local()


def _get_arxiv_count() -> int:
    return getattr(_arxiv_counter, "count", 0)


def _increment_arxiv_count():
    _arxiv_counter.count = _get_arxiv_count() + 1


def reset_arxiv_counter():
    """Call this before each ablation config to reset the per-thread cap."""
    _arxiv_counter.count = 0

# ── Known paper aliases for direct arXiv lookup ───────────────────────────────
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


def _can_call_arxiv() -> bool:
    """Check if we're still under the per-run arXiv cap."""
    return _get_arxiv_count() < MAX_ARXIV_CALLS_PER_RUN


def _fetch_by_arxiv_id(arxiv_id: str) -> dict | None:
    """Fetch a paper directly by its arXiv ID — guaranteed exact match."""
    # Check cache first
    cached = get_cached(f"id:{arxiv_id}", "arxiv_direct")
    if cached is not None:
        logging.info("CACHE HIT [arxiv_direct] id:%s", arxiv_id)
        return cached[0] if cached else None

    if not _can_call_arxiv():
        logging.warning("arXiv cap reached (%d), skipping fetch for %s", MAX_ARXIV_CALLS_PER_RUN, arxiv_id)
        return None

    time.sleep(ARXIV_DELAY_SECONDS)
    _increment_arxiv_count()

    try:
        results = list(arxiv.Search(
            id_list=[arxiv_id], max_results=1
        ).results())
        if not results:
            return None
        p = results[0]
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
        result = {
            "source":  "arxiv_direct",
            "url":     f"https://arxiv.org/abs/{arxiv_id}",
            "title":   p.title,
            "snippet": snippet,
        }
        set_cached(f"id:{arxiv_id}", "arxiv_direct", [result])
        return result
    except Exception as e:
        print(f"  [arXiv direct warn] {e}")
        return None


def _direct_arxiv_lookup(fact: str, context: str = "") -> list[dict]:
    """
    Try to find the exact paper being referenced in the fact.
    """
    search_text = (fact + " " + context).lower()
    results = []
    seen_ids = set()

    for keyword, arxiv_id in KNOWN_PAPERS.items():
        if keyword in search_text and arxiv_id not in seen_ids:
            seen_ids.add(arxiv_id)
            paper = _fetch_by_arxiv_id(arxiv_id)
            if paper:
                results.append(paper)

    combined = fact + " " + context
    caps = re.findall(r'\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z0-9][A-Za-z0-9\-]*)?\b', combined)
    seen_caps = set()
    unique_caps = [c for c in caps if not (c in seen_caps or seen_caps.add(c))]

    for term in unique_caps[:5]:
        if len(term) < 3 or not _can_call_arxiv():
            continue

        # Check cache
        cache_key = f'ti:"{term}"'
        cached = get_cached(cache_key, "arxiv")
        if cached is not None:
            logging.info("CACHE HIT [arxiv] %s", cache_key)
            for paper in cached:
                aid = paper.get("url", "").split("/abs/")[-1].split("v")[0]
                if aid and aid not in seen_ids:
                    seen_ids.add(aid)
                    results.append(paper)
            continue

        time.sleep(ARXIV_DELAY_SECONDS)
        _increment_arxiv_count()

        try:
            search = arxiv.Search(
                query=cache_key,
                max_results=RESULTS_PER_QUERY,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            batch_results = []
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
                    paper = {
                        "source":  "arxiv_direct",
                        "url":     p.entry_id,
                        "title":   p.title,
                        "snippet": snippet,
                    }
                    batch_results.append(paper)
                    results.append(paper)
            set_cached(cache_key, "arxiv", batch_results)
        except Exception:
            pass

    return results[:3]


def _search_arxiv(query: str) -> list[dict]:
    """Search arXiv with the adversarial query (with caching + backoff)."""
    # Cache check
    cached = get_cached(query, "arxiv")
    if cached is not None:
        logging.info("CACHE HIT [arxiv] %s", query[:50])
        return cached

    if not _can_call_arxiv():
        logging.warning("arXiv cap reached, using web-only for: %s", query[:50])
        return []

    time.sleep(ARXIV_DELAY_SECONDS)
    _increment_arxiv_count()

    results = []
    for attempt in range(3):
        try:
            search = arxiv.Search(
                query=query,
                max_results=RESULTS_PER_QUERY,
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
            break
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = (2 ** attempt) * 3
                logging.warning("arXiv rate limit, waiting %ds (attempt %d/3)", wait, attempt + 1)
                time.sleep(wait)
            else:
                print(f"  [arXiv warn] {e}")
                break

    if results:
        set_cached(query, "arxiv", results)
    return results


def _search_web(query: str) -> list[dict]:
    """Search the web via DuckDuckGo (with caching)."""
    cached = get_cached(query, "web")
    if cached is not None:
        logging.info("CACHE HIT [web] %s", query[:50])
        return cached

    results = []
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=3))
        for hit in hits:
            results.append({
                "source":  "web",
                "url":     hit.get("href", ""),
                "title":   hit.get("title", ""),
                "snippet": hit.get("body", ""),
            })
    except Exception as e:
        print(f"  [Web warn] {e}")

    if results:
        set_cached(query, "web", results)
    return results


# Alias so precache_evidence.py can import _web_search
_web_search = _search_web


def retrieve_from_semantic_scholar(arxiv_id: str) -> list[dict]:
    """Fetch paper details from Semantic Scholar. No API key required."""
    cache_key = f"ss:{arxiv_id}"
    cached = get_cached(cache_key, "web")
    if cached:
        return cached
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        params = {"fields": "title,abstract,year,authors,tldr"}
        r = _requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            entry = [{
                "title": data.get("title", ""),
                "snippet": data.get("abstract", ""),
                "url": f"https://www.semanticscholar.org/paper/{data.get('paperId','')}",
                "source": "semantic_scholar",
            }]
            set_cached(cache_key, "web", entry)
            return entry
    except Exception:
        pass
    return []


def retrieve_evidence(queries: list[str], fact: str = "", context: str = "") -> list[dict]:
    """
    Retrieve evidence using three sources IN PARALLEL:
      1. Direct arXiv lookup (by paper name / known ID)
      2. arXiv adversarial search
      3. DuckDuckGo web search

    All sources run concurrently with caching and rate-limit protection.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

    tasks = []

    if fact:
        tasks.append(("direct", None, fact))
    for q in queries:
        tasks.append(("arxiv", q, None))
        tasks.append(("web",   q, None))

    fact_lower    = fact.lower()
    context_lower = context.lower()

    # Extra: venue-specific search
    venue_keywords = ["icml","neurips","nips","acl","emnlp","naacl","iclr","cvpr",
                      "iccv","eccv","aaai","ijcai","conference","workshop","published at"]
    if any(v in fact_lower for v in venue_keywords):
        caps = re.findall(r'\b[A-Z][A-Za-z0-9\-]{2,}\b', context)[:3]
        if caps:
            venue_query = f"{caps[0]} paper published conference venue official proceedings"
            tasks.append(("web", venue_query, None))

    # PDF results extraction
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
    _context = context

    def run_task(task):
        kind, query, f = task
        try:
            if kind == "pdf":
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

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_task, t): t for t in tasks}
        try:
            for future in as_completed(futures, timeout=RETRIEVAL_TIMEOUT_SECONDS):
                try:
                    raw_results.extend(future.result(timeout=RETRIEVAL_TIMEOUT_SECONDS))
                except Exception:
                    pass
        except TimeoutError:
            pass

    # Deduplicate — PDF first, then direct, then Semantic Scholar, then rest
    all_evidence = []
    seen_urls    = set()

    for ev in raw_results:
        if ev.get("source") == "arxiv_pdf":
            url = ev.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_evidence.append(ev)

    for ev in raw_results:
        if ev.get("source") == "arxiv_direct":
            url = ev.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_evidence.append(ev)

    # Try Semantic Scholar if we found an arxiv_id from direct lookup
    search_text = (fact + " " + context).lower()
    for keyword, aid in KNOWN_PAPERS.items():
        if keyword in search_text:
            ss_results = retrieve_from_semantic_scholar(aid)
            for ev in ss_results:
                url = ev.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_evidence.append(ev)
            break  # only need one paper match

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
