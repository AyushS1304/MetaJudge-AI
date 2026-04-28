"""
debug_test.py — Run this from project root to diagnose the BERT issue
Shows exactly what the atomicizer, retriever, and judge are doing.

Run: python debug_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

FACT = "BERT achieved 80.5% F1 on the SQuAD 2.0 benchmark."

print("="*60)
print("DEBUG TEST — BERT metric hallucination")
print("="*60)

# ── Test 1: Atomicizer ────────────────────────────────────────
print("\n[TEST 1] Atomicizer")
print(f"Input: {FACT}")
from modules.atomicizer import atomicize
facts = atomicize(FACT)
print(f"Output ({len(facts)} facts):")
for i, f in enumerate(facts, 1):
    print(f"  {i}. {f}")

if len(facts) == 1 and "80.5" in facts[0] and "SQuAD" in facts[0]:
    print("  ✓ PASS — kept metric and benchmark together")
else:
    print("  ✗ FAIL — split the metric from benchmark")
    print("  → The atomicizer prompt fix was not applied to your file")
    print("  → Make sure you replaced modules/atomicizer.py")

# ── Test 2: Retriever — direct arXiv lookup ───────────────────
print("\n[TEST 2] Direct arXiv retrieval for BERT")
from modules.retriever import _direct_arxiv_lookup, _fetch_by_arxiv_id

print("Fetching arXiv:1810.04805 directly...")
paper = _fetch_by_arxiv_id("1810.04805")
if paper:
    snippet = paper["snippet"]
    print(f"  ✓ Found: {paper['title'][:60]}")
    if "86.7" in snippet:
        print("  ✓ PASS — abstract contains '86.7' (the real score)")
        idx = snippet.find("86.7")
        print(f"  Context: ...{snippet[max(0,idx-50):idx+80]}...")
    elif "86" in snippet:
        print("  ~ PARTIAL — abstract contains '86' but not '86.7'")
        idx = snippet.find("86")
        print(f"  Context: ...{snippet[max(0,idx-50):idx+80]}...")
    else:
        print("  ✗ FAIL — abstract does NOT contain 86.7")
        print(f"  Abstract snippet: {snippet[:300]}")
else:
    print("  ✗ FAIL — could not fetch BERT paper from arXiv")
    print("  → Check your internet connection")

# ── Test 3: Direct arXiv lookup from fact ────────────────────
print("\n[TEST 3] Direct lookup triggered from fact string")
results = _direct_arxiv_lookup(FACT)
print(f"  Found {len(results)} direct results")
for r in results:
    print(f"  Source: {r['source']} | {r['title'][:55]}")
    if "86.7" in r["snippet"]:
        print("  ✓ Contains 86.7 — retriever is working")
    else:
        print("  ✗ Does NOT contain 86.7")

# ── Test 4: Full retrieval ─────────────────────────────────────
print("\n[TEST 4] Full retrieve_evidence call")
from modules.retriever import retrieve_evidence, format_evidence_block
from modules.query_generator import generate_skeptical_queries

queries = generate_skeptical_queries(FACT)
print(f"  Queries: {queries}")
evidence = retrieve_evidence(queries, fact=FACT)
print(f"  Total evidence items: {len(evidence)}")

found_score = False
for ev in evidence:
    if "86.7" in ev["snippet"] or "86.7" in ev.get("title",""):
        found_score = True
        print(f"  ✓ FOUND 86.7 in: [{ev['source']}] {ev['title'][:55]}")

if not found_score:
    print("  ✗ 86.7 NOT found in any evidence item")
    print("  → The retriever is not surfacing the real BERT score")
    print("  Sources retrieved:")
    for ev in evidence[:5]:
        print(f"    [{ev['source']}] {ev['title'][:55]}")

# ── Test 5: Judge ──────────────────────────────────────────────
print("\n[TEST 5] Judge on the actual fact + evidence")
ev_block = format_evidence_block(evidence)
from modules.judge import judge_claim
result = judge_claim(FACT, ev_block)
print(f"  Verdict:  {result['verdict']}")
print(f"  Reasoning: {result['reasoning']}")
print(f"  Quote:    {result.get('evidence_quote','(none)')[:100]}")

if result["verdict"] == "CONTRADICTED":
    print("  ✓ PASS — judge correctly identified the error")
elif result["verdict"] == "SUPPORTED":
    print("  ✗ FAIL — judge wrongly said SUPPORTED")
    print("  → Either judge prompt not updated OR judge is hallucinating")
else:
    print("  ~ INSUFFICIENT — judge could not find relevant evidence")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)