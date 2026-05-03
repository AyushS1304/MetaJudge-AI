"""
evaluation/ablation.py — Ablation Study Runner (Legacy)
--------------------------------------------------------
See evaluation/ablation_runner.py for the updated 4-config version.

Runs the three ablation conditions described in the paper:
  Ablation 1: Remove CoVe Loop
  Ablation 2: Remove Skeptical Query Generator
  Ablation 3: Remove Atomic Decomposition
"""

import json
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAX_FACTS
from modules.atomicizer      import atomicize
from modules.query_generator import generate_skeptical_queries
from modules.retriever       import retrieve_evidence, format_evidence_block
from modules.judge           import judge_claim, VERDICT_CONTRADICTED
from modules.cove_loop       import run_cove_verification
from evaluation.skeptic_score import BenchmarkReport, ClaimResult

SEPARATOR = "─" * 55


def run_no_cove(summary: str, verbose: bool = False) -> list[dict]:
    """Full pipeline MINUS the CoVe verification loop."""
    facts   = atomicize(summary)[:MAX_FACTS]
    results = []
    for fact in facts:
        queries  = generate_skeptical_queries(fact)
        evidence = retrieve_evidence(queries)
        ev_block = format_evidence_block(evidence)
        verdict  = judge_claim(fact, ev_block)
        verdict["fact"]             = fact
        verdict["cove_applied"]     = False
        verdict["cove_meta_verdict"] = None
        results.append(verdict)
        time.sleep(0.3)
    return results


def run_no_adversarial(summary: str, verbose: bool = False) -> list[dict]:
    """Full pipeline MINUS adversarial query generation."""
    facts   = atomicize(summary)[:MAX_FACTS]
    results = []
    for fact in facts:
        queries  = [fact, f"information about {fact}"]
        evidence = retrieve_evidence(queries)
        ev_block = format_evidence_block(evidence)
        verdict  = judge_claim(fact, ev_block)
        if verdict["verdict"] == VERDICT_CONTRADICTED:
            verdict = run_cove_verification(fact, verdict, ev_block)
        else:
            verdict["cove_applied"]      = False
            verdict["cove_meta_verdict"] = None
        verdict["fact"] = fact
        results.append(verdict)
        time.sleep(0.3)
    return results


def run_no_atomic(summary: str, verbose: bool = False) -> list[dict]:
    """Full pipeline MINUS atomic decomposition (operates at sentence level)."""
    sentences = [s.strip() + "." for s in summary.replace(".\n", ". ").split(". ") if len(s.strip()) > 10]
    results   = []
    for sent in sentences[:MAX_FACTS]:
        queries  = generate_skeptical_queries(sent)
        evidence = retrieve_evidence(queries)
        ev_block = format_evidence_block(evidence)
        verdict  = judge_claim(sent, ev_block)
        if verdict["verdict"] == VERDICT_CONTRADICTED:
            verdict = run_cove_verification(sent, verdict, ev_block)
        else:
            verdict["cove_applied"]      = False
            verdict["cove_meta_verdict"] = None
        verdict["fact"] = sent
        results.append(verdict)
        time.sleep(0.3)
    return results


def run_ablation_study(bench_path: str = "data/skepticbench_sample.json"):
    """Run all ablation conditions on SkepticBench and compare."""
    with open(bench_path) as f:
        bench = json.load(f)

    configs = {
        "No CoVe":          run_no_cove,
        "No Adversarial":   run_no_adversarial,
        "No Atomic Decomp": run_no_atomic,
    }

    reports = {name: BenchmarkReport() for name in configs}

    for item in bench:
        injected_facts = {e["fact"] for e in item.get("injected_errors", [])}
        print(f"\nProcessing: {item['id']}")

        for name, runner in configs.items():
            print(f"  Running ablation: {name}...")
            results = runner(item["summary"])
            for res in results:
                gt = "hallucinated" if res["fact"] in injected_facts else "correct"
                reports[name].add(ClaimResult(
                    fact              = res["fact"],
                    ground_truth      = gt,
                    verdict           = res["verdict"],
                    cove_applied      = res.get("cove_applied", False),
                    cove_meta_verdict = res.get("cove_meta_verdict"),
                ))

    print(f"\n\n{'='*55}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*55}")
    print(f"  {'Condition':<22} {'F1':>6}  {'Precision':>9}  {'Recall':>6}  {'Skeptic':>7}")
    print(f"  {'-'*22} {'-'*6}  {'-'*9}  {'-'*6}  {'-'*7}")

    for name, report in reports.items():
        print(
            f"  {name:<22} "
            f"{report.f1():>6.3f}  "
            f"{report.precision():>9.3f}  "
            f"{report.recall():>6.3f}  "
            f"{report.skeptic_score():>7.3f}"
        )

    print(f"{'='*55}")
    return reports


if __name__ == "__main__":
    if not os.getenv("NVIDIA_API_KEY"):
        print("⚠  Set NVIDIA_API_KEY in .env first.")
        sys.exit(1)
    run_ablation_study()