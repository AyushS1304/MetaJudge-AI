"""
Run this from the project root to diagnose the BERT retrieval/judging path.

Run:
    python debug_test.py
"""

from __future__ import annotations

import os
import sys

from env_utils import load_env


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_env()

FACT = "BERT achieved 80.5% F1 on the SQuAD 2.0 benchmark."


def _print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    _print_header("DEBUG TEST - BERT metric hallucination")

    print("\n[TEST 1] Atomicizer")
    print(f"Input: {FACT}")
    from modules.atomicizer import atomicize

    facts = atomicize(FACT)
    print(f"Output ({len(facts)} facts):")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact}")

    if len(facts) == 1 and "80.5" in facts[0] and "SQuAD" in facts[0]:
        print("  PASS - kept metric and benchmark together")
    else:
        print("  FAIL - split the metric from benchmark")
        print("  -> The atomicizer prompt fix was not applied to your file")
        print("  -> Make sure you replaced modules/atomicizer.py")

    print("\n[TEST 2] Direct arXiv retrieval for BERT")
    from modules.retriever import _direct_arxiv_lookup, _fetch_by_arxiv_id

    print("Fetching arXiv:1810.04805 directly...")
    paper = _fetch_by_arxiv_id("1810.04805")
    if paper:
        snippet = paper["snippet"]
        print(f"  Found: {paper['title'][:60]}")
        if "86.7" in snippet:
            print("  PASS - abstract contains '86.7' (the real score)")
            idx = snippet.find("86.7")
            print(f"  Context: ...{snippet[max(0, idx - 50):idx + 80]}...")
        elif "86" in snippet:
            print("  PARTIAL - abstract contains '86' but not '86.7'")
            idx = snippet.find("86")
            print(f"  Context: ...{snippet[max(0, idx - 50):idx + 80]}...")
        else:
            print("  FAIL - abstract does not contain 86.7")
            print(f"  Abstract snippet: {snippet[:300]}")
    else:
        print("  FAIL - could not fetch BERT paper from arXiv")
        print("  -> Check your internet connection")

    print("\n[TEST 3] Direct lookup triggered from fact string")
    results = _direct_arxiv_lookup(FACT)
    print(f"  Found {len(results)} direct results")
    for result in results:
        print(f"  Source: {result['source']} | {result['title'][:55]}")
        if "86.7" in result["snippet"]:
            print("  PASS - contains 86.7; retriever is working")
        else:
            print("  FAIL - does not contain 86.7")

    print("\n[TEST 4] Full retrieve_evidence call")
    from modules.query_generator import generate_skeptical_queries
    from modules.retriever import format_evidence_block, retrieve_evidence

    queries = generate_skeptical_queries(FACT)
    print(f"  Queries: {queries}")
    evidence = retrieve_evidence(queries, fact=FACT)
    print(f"  Total evidence items: {len(evidence)}")

    found_score = False
    for item in evidence:
        if "86.7" in item["snippet"] or "86.7" in item.get("title", ""):
            found_score = True
            print(f"  FOUND 86.7 in: [{item['source']}] {item['title'][:55]}")

    if not found_score:
        print("  FAIL - 86.7 not found in any evidence item")
        print("  -> The retriever is not surfacing the real BERT score")
        print("  Sources retrieved:")
        for item in evidence[:5]:
            print(f"    [{item['source']}] {item['title'][:55]}")

    print("\n[TEST 5] Judge on the actual fact + evidence")
    from modules.judge import judge_claim

    evidence_block = format_evidence_block(evidence)
    result = judge_claim(FACT, evidence_block)
    print(f"  Verdict:   {result['verdict']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Quote:     {result.get('evidence_quote', '(none)')[:100]}")

    if result["verdict"] == "CONTRADICTED":
        print("  PASS - judge correctly identified the error")
    elif result["verdict"] == "SUPPORTED":
        print("  FAIL - judge wrongly said SUPPORTED")
        print("  -> Either judge prompt is outdated or the judge hallucinated")
    else:
        print("  PARTIAL - judge could not find enough relevant evidence")

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
