"""
pipeline.py — Skeptical CoVe-RAG: Full End-to-End Pipeline
===========================================================
Orchestrates all 6 steps:
  1. Atomicizer        → break summary into atomic facts
  2. Skeptical Queries → adversarial queries per fact
  3. Hybrid Retrieval  → arXiv + web evidence
  4. LLM Judge         → verdict per fact
  5. CoVe Loop         → meta-verify the judge's CONTRADICTED decisions
  6. RARR Editor       → surgically fix confirmed errors

Usage:
  python pipeline.py                          # runs sample text
  python pipeline.py --bench                  # runs on SkepticBench sample data
  python pipeline.py --text "Your text here"  # runs on custom text
"""

import sys
import json
import time
import argparse
import os

# Windows-safe path fix — works on all platforms
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from config import MAX_FACTS
from modules.atomicizer     import atomicize
from modules.query_generator import generate_skeptical_queries
from modules.retriever       import retrieve_evidence, format_evidence_block
from modules.judge           import judge_claim, VERDICT_CONTRADICTED, VERDICT_INSUFFICIENT
from modules.cove_loop       import run_cove_verification
from modules.editor          import edit_sentence, apply_corrections_to_summary
from evaluation.skeptic_score import BenchmarkReport, ClaimResult


SEPARATOR = "─" * 60


def run_pipeline(summary: str, verbose: bool = True) -> dict:
    """
    Run the full 6-step Skeptical CoVe-RAG pipeline on a summary.

    Args:
        summary: Input text (LLM-generated summary to verify).
        verbose: Print step-by-step progress.

    Returns:
        A dict with keys:
        {original, corrected, facts, corrections, report}
    """
    if verbose:
        print(f"\n{SEPARATOR}")
        print("  SKEPTICAL CoVe-RAG PIPELINE")
        print(SEPARATOR)
        print(f"\nInput summary:\n  {summary}\n")

    # ── STEP 1: ATOMICIZE ─────────────────────────────────────────────
    if verbose: print(f"[Step 1] Atomic decomposition...")
    facts = atomicize(summary)[:MAX_FACTS]
    if verbose: print(f"  → {len(facts)} atomic facts extracted.")

    corrections  = []
    all_results  = []
    report       = BenchmarkReport()

    for i, fact in enumerate(facts, 1):
        if verbose:
            print(f"\n{SEPARATOR}")
            print(f"  Fact {i}/{len(facts)}: {fact}")

        # ── STEP 2: SKEPTICAL QUERIES ──────────────────────────────────
        if verbose: print(f"  [Step 2] Generating adversarial queries...")
        queries = generate_skeptical_queries(fact)
        if verbose:
            for q in queries:
                print(f"    ↯  {q}")

        # ── STEP 3: HYBRID RETRIEVAL ───────────────────────────────────
        if verbose: print(f"  [Step 3] Retrieving evidence (arXiv + web)...")
        evidence      = retrieve_evidence(queries, fact=fact, context=summary)
        evidence_block = format_evidence_block(evidence)
        if verbose: print(f"    → {len(evidence)} evidence items retrieved.")

        # ── STEP 4: LLM JUDGE ─────────────────────────────────────────
        if verbose: print(f"  [Step 4] Judging claim...")
        judge_result = judge_claim(fact, evidence_block)
        if verbose:
            print(f"    → Raw verdict: {judge_result['verdict']}")
            print(f"    → Reasoning:   {judge_result['reasoning']}")

        # ── STEP 4b: SECOND OPINION — Gemini fallback for INSUFFICIENT ───
        if judge_result["verdict"] == VERDICT_INSUFFICIENT:
            try:
                from modules.second_opinion import get_second_opinion
                from modules.retriever import KNOWN_PAPERS
                second = get_second_opinion(
                    fact, summary, evidence_block,
                    KNOWN_PAPERS, verbose=verbose
                )
                if second:
                    # Merge Gemini result into judge_result
                    judge_result["verdict"]        = second.get("verdict", VERDICT_INSUFFICIENT)
                    judge_result["reasoning"]      = f"[Gemini 2nd opinion] {second.get('reasoning','')}"
                    judge_result["evidence_quote"] = second.get("evidence_quote","")
                    judge_result["evidence_source"]= second.get("evidence_source","")
                    judge_result["gemini_used"]    = True
                    judge_result["pdf_used"]       = second.get("pdf_used", False)
                    if verbose:
                        print(f"    → Gemini verdict: {judge_result['verdict']}")
                else:
                    # Mark as disputed — couldn't verify either way
                    judge_result["disputed"] = True
                    if verbose:
                        print(f"    → Could not verify — marked as DISPUTED")
            except Exception as e:
                judge_result["disputed"] = True
                if verbose:
                    print(f"    → Second opinion error: {e}")

        # ── STEP 5: CoVe VERIFICATION LOOP ────────────────────────────
        if judge_result["verdict"] == VERDICT_CONTRADICTED:
            if verbose: print(f"  [Step 5] ★ CoVe activated — verifying judge decision...")
            final_result = run_cove_verification(fact, judge_result, evidence_block)
            if verbose:
                print(f"    → CoVe meta-verdict: {final_result['cove_meta_verdict']}")
                print(f"    → Final verdict:     {final_result['verdict']}")
        else:
            final_result = dict(judge_result)
            final_result["cove_applied"]      = False
            final_result["cove_meta_verdict"] = None
            if verbose:
                print(f"  [Step 5] CoVe skipped (verdict is {judge_result['verdict']}).")

        final_result["fact"] = fact
        all_results.append(final_result)

        # ── STEP 6: TARGETED RARR EDITOR ──────────────────────────────
        if (
            final_result["verdict"] == VERDICT_CONTRADICTED
            and final_result.get("cove_meta_verdict") == "CONFIRMED_CONTRADICTION"
        ):
            if verbose: print(f"  [Step 6] Applying surgical correction...")

            # Find the sentence in the summary that contains this fact
            source_sentence = _find_source_sentence(summary, fact)

            edit_result = edit_sentence(
                original_sentence=source_sentence,
                wrong_fact=fact,
                cove_result=final_result,
                evidence_block=evidence_block,
            )
            if edit_result["changed"]:
                corrections.append({
                    "fact":           fact,
                    "source_sentence": source_sentence,
                    **edit_result,
                })
                if verbose:
                    print(f"    ✓ Fixed: '{edit_result['error_span']}' → '{edit_result['correction']}'")
                    print(f"    Source: {edit_result['source_url']}")
            else:
                if verbose: print(f"    ✗ Could not determine exact correction.")
        else:
            if verbose: print(f"  [Step 6] Editor skipped (no confirmed contradiction).")

        # Small delay to respect API rate limits
        time.sleep(0.3)

    # ── APPLY ALL CORRECTIONS TO SUMMARY ──────────────────────────────
    corrected_summary = apply_corrections_to_summary(summary, corrections)

    if verbose:
        print(f"\n{SEPARATOR}")
        print("  FINAL RESULTS")
        print(SEPARATOR)
        print(f"\nOriginal:  {summary}")
        print(f"Corrected: {corrected_summary}")
        print(f"\nCorrections applied: {len(corrections)}")
        for c in corrections:
            print(f"  • '{c['error_span']}' → '{c['correction']}' (source: {c['source_url']})")

    return {
        "original":   summary,
        "corrected":  corrected_summary,
        "facts":      facts,
        "results":    all_results,
        "corrections": corrections,
    }


def run_benchmark(bench_path: str = "data/skepticbench_sample.json") -> BenchmarkReport:
    """
    Run the pipeline on SkepticBench and compute evaluation metrics.
    Ground-truth labels come from the dataset's injected_errors field.
    """
    with open(bench_path) as f:
        bench = json.load(f)

    report = BenchmarkReport()

    for item in bench:
        print(f"\n{'='*60}")
        print(f"  Benchmark item: {item['id']}")
        print(f"  Summary: {item['summary']}")

        output   = run_pipeline(item["summary"], verbose=True)
        injected = {e["fact"] for e in item.get("injected_errors", [])}

        for res in output["results"]:
            ground_truth = "hallucinated" if res["fact"] in injected else "correct"
            claim_result = ClaimResult(
                fact              = res["fact"],
                ground_truth      = ground_truth,
                verdict           = res["verdict"],
                cove_applied      = res.get("cove_applied", False),
                cove_meta_verdict = res.get("cove_meta_verdict"),
                correction        = next(
                    (c["correction"] for c in output["corrections"] if c["fact"] == res["fact"]), ""
                ),
                source_url = res.get("evidence_source", ""),
            )
            report.add(claim_result)

    report.print_report()
    return report


def _find_source_sentence(text: str, fact: str) -> str:
    """Find the sentence in text most likely to contain the atomic fact."""
    sentences = [s.strip() for s in text.replace(".\n", ". ").split(". ") if s.strip()]
    # Find sentence with most word overlap with the fact
    fact_words = set(fact.lower().split())
    best, best_score = text, 0
    for sent in sentences:
        overlap = len(set(sent.lower().split()) & fact_words)
        if overlap > best_score:
            best_score = overlap
            best = sent + ("." if not sent.endswith(".") else "")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skeptical CoVe-RAG Pipeline")
    parser.add_argument("--text",  type=str, help="Custom summary text to verify")
    parser.add_argument("--bench", action="store_true", help="Run on SkepticBench sample")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠  GROQ_API_KEY not set. Create a .env file with:\n  GROQ_API_KEY=your_key_here\n")
        sys.exit(1)

    if args.bench:
        run_benchmark()
    elif args.text:
        run_pipeline(args.text)
    else:
        # Default demo
        demo_text = (
            "BERT, introduced by Google in 2018, uses a bidirectional transformer encoder. "
            "It was pre-trained on BookCorpus and English Wikipedia, and achieved 80.5% F1 "
            "on the SQuAD 2.0 benchmark. The paper was authored by Devlin et al. and "
            "published at NAACL 2019."
        )
        run_pipeline(demo_text)