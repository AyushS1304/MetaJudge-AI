"""
Pre-caches all arXiv and Semantic Scholar evidence for benchmark papers.
Run once before any evaluation. After this, all evaluation runs hit cache only.
Usage: python build_dataset/precache_evidence.py
"""

import json, time, sys, os, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env_utils import load_env
load_env()

from modules.cache_layer import get_cached, set_cached
import arxiv as arxiv_lib

BENCHMARK_FILE = "data/skepticbench_sample.json"
SS_BASE = "https://api.semanticscholar.org/graph/v1"

with open(BENCHMARK_FILE) as f:
    samples = json.load(f)

arxiv_ids = [s["arxiv_id"] for s in samples if s.get("arxiv_id")]
print(f"Found {len(arxiv_ids)} arXiv IDs to pre-cache: {arxiv_ids}")

# --- arXiv direct fetch ---
for arxiv_id in arxiv_ids:
    cache_key = f"id:{arxiv_id}"
    if get_cached(cache_key, "arxiv_direct"):
        print(f"  ALREADY CACHED (arXiv): {arxiv_id}")
        continue
    try:
        print(f"  Fetching arXiv:{arxiv_id} ...", end=" ", flush=True)
        search = arxiv_lib.Search(id_list=[arxiv_id])
        results = list(search.results())
        if results:
            paper = results[0]
            pub_year = paper.published.year
            entry = [{
                "source": "arxiv_direct",
                "title": paper.title,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "snippet": (
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(a.name for a in paper.authors[:6])}\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d')} (Year: {pub_year})\n"
                    f"FACT-CHECK NOTE: This paper was first published/introduced/released in {pub_year}, not any other year.\n"
                    f"arXiv ID: {arxiv_id}\n"
                    f"Abstract: {paper.summary}"
                ),
            }]
            set_cached(cache_key, "arxiv_direct", entry)
            print(f"OK — {paper.title[:60]}")
        time.sleep(3)
    except Exception as e:
        print(f"FAILED: {e}")
        time.sleep(5)

# --- Semantic Scholar fetch ---
print("\nFetching Semantic Scholar entries...")
for arxiv_id in arxiv_ids:
    cache_key = f"ss:{arxiv_id}"
    if get_cached(cache_key, "web"):
        print(f"  ALREADY CACHED (SS): {arxiv_id}")
        continue
    try:
        url = f"{SS_BASE}/paper/arXiv:{arxiv_id}"
        params = {"fields": "title,abstract,year,authors,tldr"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            entry = [{
                "title": data.get("title", ""),
                "snippet": data.get("abstract", ""),
                "url": f"https://www.semanticscholar.org/paper/{data.get('paperId','')}",
                "source": "semantic_scholar"
            }]
            set_cached(cache_key, "web", entry)
            print(f"  SS cached: {arxiv_id}")
        else:
            print(f"  SS status {r.status_code} for {arxiv_id}")
        time.sleep(1.5)
    except Exception as e:
        print(f"  SS failed for {arxiv_id}: {e}")

# --- Pre-cache adversarial query evidence for each known atomic fact ---
print("\nPre-caching web evidence for atomic facts in benchmark...")
from modules.query_generator import generate_skeptical_queries
from modules.retriever import _search_web

for sample in samples:
    facts = sample.get("atomic_facts", [])
    print(f"\n  Paper: {sample.get('paper', 'unknown')[:60]}")
    for fact in facts:
        sentinel = f"prefetch:{fact[:80]}"
        if get_cached(sentinel, "web"):
            print(f"    CACHED: {fact[:50]}...")
            continue
        try:
            queries = generate_skeptical_queries(fact)
            for q in queries[:2]:
                if not get_cached(q, "web"):
                    res = _search_web(q)
                    if res:
                        set_cached(q, "web", res)
                    time.sleep(0.5)
            set_cached(sentinel, "web", [{"prefetched": True}])
            print(f"    DONE: {fact[:50]}...")
        except Exception as e:
            print(f"    Error on fact: {e}")
        time.sleep(0.3)

print("\n✓ Pre-caching complete. Run evaluation now.")
