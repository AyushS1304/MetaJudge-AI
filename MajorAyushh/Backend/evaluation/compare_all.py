"""
evaluation/compare_all.py — Full Comparative Evaluation
---------------------------------------------------------
Runs ALL systems on SkepticBench and produces a paper-ready
comparison table:

  System               | Precision | Recall |   F1  | Skeptic Score
  ---------------------|-----------|--------|-------|---------------
  Zero-Shot LLM        |    ...    |  ...   |  ...  |    ...
  Standard RAG         |    ...    |  ...   |  ...  |    ...
  Vanilla RARR         |    ...    |  ...   |  ...  |    ...
  Ours (full pipeline) |    ...    |  ...   |  ...  |    ...

Run this AFTER collecting results to generate your Table 1.
"""

import json
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.skeptic_score import BenchmarkReport, ClaimResult
from modules.atomicizer       import atomicize
from config import MAX_FACTS, GROQ_API_KEY


def _gt_lookup(results_list: list[dict], injected_facts: set) -> list[ClaimResult]:
    """Convert a results list to ClaimResult objects with ground-truth labels."""
    out = []
    for res in results_list:
        fact = res.get("fact") or res.get("sentence", "")
        gt   = "hallucinated" if fact in injected_facts else "correct"
        out.append(ClaimResult(
            fact              = fact,
            ground_truth      = gt,
            verdict           = res.get("verdict", "INSUFFICIENT_EVIDENCE"),
            cove_applied      = res.get("cove_applied", False),
            cove_meta_verdict = res.get("cove_meta_verdict"),
        ))
    return out


def run_full_comparison(bench_path: str = "data/skepticbench_sample.json"):
    """Run all four systems and print a comparison table."""

    # Import here to avoid circular imports
    from pipeline import run_pipeline
    from baselines.baseline_standard_rag import run_standard_rag
    from baselines.baseline_rarr         import run_vanilla_rarr
    from baselines.baseline_zeroshot     import run_zeroshot

    with open(bench_path) as f:
        bench = json.load(f)

    systems = {
        "Zero-Shot LLM":   (run_zeroshot,      "results"),
        "Standard RAG":    (run_standard_rag,  "results"),
        "Vanilla RARR":    (run_vanilla_rarr,  "results"),
        "Ours (full)":     (run_pipeline,      "results"),
    }

    reports = {name: BenchmarkReport() for name in systems}

    for item in bench:
        injected_facts = {e["fact"] for e in item.get("injected_errors", [])}
        print(f"\n{'─'*50}\nItem: {item['id']}")

        for sys_name, (runner, results_key) in systems.items():
            print(f"  [{sys_name}] running...")
            try:
                output = runner(item["summary"], verbose=False)
                results = output.get(results_key, [])
                for cr in _gt_lookup(results, injected_facts):
                    reports[sys_name].add(cr)
            except Exception as e:
                print(f"    ✗ Error: {e}")
            time.sleep(0.5)

    # ── Print paper-ready comparison table ────────────────────────────────
    width = 21
    print(f"\n\n{'='*66}")
    print("  TABLE 1: System Comparison on SkepticBench")
    print(f"{'='*66}")
    print(f"  {'System':<{width}} {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Skeptic↑':>8}")
    print(f"  {'-'*width} {'------':>6}  {'------':>6}  {'------':>6}  {'--------':>8}")

    for name, report in reports.items():
        marker = "★" if name == "Ours (full)" else " "
        print(
            f"{marker} {name:<{width}} "
            f"{report.precision():>6.3f}  "
            f"{report.recall():>6.3f}  "
            f"{report.f1():>6.3f}  "
            f"{report.skeptic_score():>8.3f}"
        )

    print(f"{'='*66}")
    print("  ★ = proposed system   Skeptic↑ = higher is better")
    print(f"\n  CoVe reversals (false positives caught): "
          f"{reports['Ours (full)'].cove_overturned_contradictions}")
    return reports


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("⚠  Set GROQ_API_KEY in .env first.")
        sys.exit(1)
    run_full_comparison()