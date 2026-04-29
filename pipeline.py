"""
MetaJudge AI end-to-end pipeline orchestration.

Usage:
  python pipeline.py
  python pipeline.py --bench
  python pipeline.py --text "Your text here"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from config import MAX_FACTS
from evaluation.skeptic_score import BenchmarkReport, ClaimResult
from modules.atomicizer import atomicize
from modules.cove_loop import run_cove_verification
from modules.deep_verifier import deep_verify
from modules.editor import apply_corrections_to_summary, edit_sentence
from modules.judge import VERDICT_CONTRADICTED, VERDICT_INSUFFICIENT, judge_claim
from modules.query_generator import generate_skeptical_queries
from modules.retriever import KNOWN_PAPERS, format_evidence_block, retrieve_evidence
from modules.text_utils import find_best_matching_sentence


SEPARATOR = "=" * 60
PipelineEventCallback = Callable[[dict[str, Any]], None]


def _emit(callback: PipelineEventCallback | None, event_type: str, **payload: Any) -> None:
    if callback is None:
        return
    callback({"type": event_type, **payload})


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _merge_deep_verification_result(
    judge_result: dict[str, Any],
    second_result: dict[str, Any] | None,
    gemini_requested: bool,
) -> dict[str, Any]:
    """
    Merge deep-verification output back into the judge result.
    """
    merged = dict(judge_result)

    if second_result and second_result.get("verdict") != VERDICT_INSUFFICIENT:
        merged["verdict"] = second_result.get("verdict", VERDICT_INSUFFICIENT)
        merged["reasoning"] = second_result.get("reasoning", "")
        merged["evidence_quote"] = second_result.get("evidence_quote", "")
        merged["evidence_source"] = second_result.get("evidence_source", "")
        merged["gemini_used"] = second_result.get("gemini_used", False)
        merged["pdf_used"] = second_result.get("pdf_used", False)
        merged["disputed"] = False
        return merged

    merged["disputed"] = True
    if gemini_requested:
        merged["gemini_used"] = True
        if second_result:
            merged["pdf_used"] = second_result.get("pdf_used", False)
    return merged


def run_pipeline(
    summary: str,
    verbose: bool = True,
    *,
    gemini_key: str | None = None,
    on_event: PipelineEventCallback | None = None,
    sleep_seconds: float = 0.3,
) -> dict[str, Any]:
    """
    Run the full 6-step MetaJudge pipeline on a summary.
    """
    summary = summary.strip()
    if not summary:
        raise ValueError("summary cannot be empty")

    if verbose:
        print(f"\n{SEPARATOR}")
        print("METAJUDGE AI PIPELINE")
        print(SEPARATOR)
        print(f"\nInput summary:\n  {summary}\n")

    _emit(on_event, "progress", value=0.05, label="Step 1 of 6 - Atomicizing...")
    _emit(on_event, "log", kind="step", message="Step 1: Atomicizing summary...")
    _log(verbose, "[Step 1] Atomic decomposition...")

    facts = atomicize(summary)[:MAX_FACTS]
    _emit(on_event, "log", kind="done", message=f"  -> {len(facts)} atomic facts extracted")
    _log(verbose, f"  -> {len(facts)} atomic facts extracted.")

    corrections: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    total_facts = len(facts)
    gemini_requested = bool(gemini_key or os.environ.get("GEMINI_API_KEY"))

    for index, fact in enumerate(facts, start=1):
        progress = 0.1 + ((index - 1) / max(total_facts, 1)) * 0.85
        _emit(on_event, "progress", value=progress, label=f"Processing fact {index} of {total_facts}...")
        _emit(on_event, "log", kind="info", message="")
        _emit(on_event, "log", kind="info", message=f"-- Fact {index}/{total_facts} -------------------------")
        _emit(on_event, "log", kind="fact", message=f"  {fact[:80]}{'...' if len(fact) > 80 else ''}")

        _log(verbose, f"\n{SEPARATOR}")
        _log(verbose, f"Fact {index}/{total_facts}: {fact}")

        _emit(on_event, "log", kind="step", message="Step 2: Generating adversarial queries...")
        _log(verbose, "  [Step 2] Generating adversarial queries...")
        queries = generate_skeptical_queries(fact)
        for query in queries:
            _emit(on_event, "log", kind="info", message=f"  -> {query[:75]}")
            _log(verbose, f"    -> {query}")

        _emit(on_event, "log", kind="step", message="Step 3: Retrieving evidence...")
        _log(verbose, "  [Step 3] Retrieving evidence (arXiv + web)...")
        evidence = retrieve_evidence(queries, fact=fact, context=summary)
        evidence_block = format_evidence_block(evidence)
        _emit(
            on_event,
            "log",
            kind="done",
            message=f"  -> {len(evidence)} evidence items (arXiv direct + adversarial + web)",
        )
        _log(verbose, f"    -> {len(evidence)} evidence items retrieved.")

        _emit(on_event, "log", kind="step", message="Step 4: Groq judge evaluating claim...")
        _log(verbose, "  [Step 4] Judging claim...")
        judge_result = judge_claim(fact, evidence_block)
        _emit(
            on_event,
            "log",
            kind=(
                "done"
                if judge_result["verdict"] == "SUPPORTED"
                else "err"
                if judge_result["verdict"] == VERDICT_CONTRADICTED
                else "info"
            ),
            message=f"  -> Groq verdict: {judge_result['verdict']}",
        )
        if judge_result.get("reasoning"):
            _emit(on_event, "log", kind="info", message=f"  -> {judge_result['reasoning'][:90]}")
        _log(verbose, f"    -> Raw verdict: {judge_result['verdict']}")
        _log(verbose, f"    -> Reasoning:   {judge_result.get('reasoning', '')}")

        if judge_result["verdict"] == VERDICT_INSUFFICIENT:
            _emit(on_event, "log", kind="gemini" if gemini_requested else "step", message="Step 4b: Deep verification...")
            _log(verbose, "  [Step 4b] Deep verification...")
            second_result = deep_verify(
                fact,
                summary,
                evidence_block,
                KNOWN_PAPERS,
                gemini_key=gemini_key or "",
                verbose=verbose,
            )
            judge_result = _merge_deep_verification_result(
                judge_result,
                second_result,
                gemini_requested=gemini_requested,
            )

            if judge_result.get("disputed"):
                _emit(
                    on_event,
                    "log",
                    kind="gemini" if gemini_requested else "info",
                    message="  -> Deep verification could not settle the claim; marked disputed",
                )
                _log(verbose, "    -> Could not verify; marked disputed.")
            else:
                source_suffix = (
                    " (PDF)"
                    if judge_result.get("pdf_used")
                    else " (Gemini)"
                    if judge_result.get("gemini_used")
                    else " (enriched search)"
                )
                _emit(
                    on_event,
                    "log",
                    kind="gemini" if judge_result["verdict"] == VERDICT_CONTRADICTED else "done",
                    message=f"  -> Deep verify{source_suffix}: {judge_result['verdict']}",
                )
                _log(verbose, f"    -> Deep verify verdict: {judge_result['verdict']}")

        if judge_result["verdict"] == VERDICT_CONTRADICTED:
            _emit(on_event, "log", kind="step", message="Step 5: CoVe meta-verification...")
            _log(verbose, "  [Step 5] CoVe activated - verifying judge decision...")
            final_result = run_cove_verification(fact, judge_result, evidence_block)
            _emit(
                on_event,
                "log",
                kind="done" if final_result.get("cove_meta_verdict") == "CONFIRMED_CONTRADICTION" else "err",
                message=f"  -> CoVe: {final_result.get('cove_meta_verdict')}",
            )
            _log(verbose, f"    -> CoVe meta-verdict: {final_result.get('cove_meta_verdict')}")
            _log(verbose, f"    -> Final verdict:     {final_result['verdict']}")
        else:
            final_result = dict(judge_result)
            final_result["cove_applied"] = False
            final_result["cove_meta_verdict"] = None
            _emit(on_event, "log", kind="step", message="Step 5: CoVe skipped")
            _log(verbose, f"  [Step 5] CoVe skipped (verdict is {judge_result['verdict']}).")

        final_result["fact"] = fact
        all_results.append(final_result)

        if (
            final_result["verdict"] == VERDICT_CONTRADICTED
            and final_result.get("cove_meta_verdict") == "CONFIRMED_CONTRADICTION"
        ):
            _emit(on_event, "log", kind="step", message="Step 6: Applying surgical correction...")
            _log(verbose, "  [Step 6] Applying surgical correction...")
            source_sentence = find_best_matching_sentence(summary, fact)
            edit_result = edit_sentence(
                original_sentence=source_sentence,
                wrong_fact=fact,
                cove_result=final_result,
                evidence_block=evidence_block,
            )
            if edit_result["changed"]:
                corrections.append(
                    {
                        "fact": fact,
                        "source_sentence": source_sentence,
                        **edit_result,
                    }
                )
                _emit(
                    on_event,
                    "log",
                    kind="done",
                    message=f"  OK Fixed: '{edit_result['error_span']}' -> '{edit_result['correction']}'",
                )
                _log(verbose, f"    OK Fixed: '{edit_result['error_span']}' -> '{edit_result['correction']}'")
                _log(verbose, f"    Source: {edit_result['source_url']}")
            else:
                _emit(on_event, "log", kind="info", message="  -> Could not extract an exact correction span")
                _log(verbose, "    -> Could not determine an exact correction.")
        else:
            _emit(on_event, "log", kind="step", message="Step 6: Editor skipped")
            _log(verbose, "  [Step 6] Editor skipped (no confirmed contradiction).")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    corrected_summary = apply_corrections_to_summary(summary, corrections)

    _emit(on_event, "progress", value=1.0, label="Pipeline complete")
    _emit(on_event, "log", kind="info", message="")
    contradicted_count = sum(1 for result in all_results if result["verdict"] == VERDICT_CONTRADICTED)
    disputed_count = sum(1 for result in all_results if result.get("disputed"))
    _emit(
        on_event,
        "log",
        kind="done",
        message=(
            f"Done - {len(corrections)} correction(s) | "
            f"{contradicted_count} contradiction(s) | {disputed_count} disputed"
        ),
    )

    if verbose:
        print(f"\n{SEPARATOR}")
        print("FINAL RESULTS")
        print(SEPARATOR)
        print(f"\nOriginal:  {summary}")
        print(f"Corrected: {corrected_summary}")
        print(f"\nCorrections applied: {len(corrections)}")
        for correction in corrections:
            print(
                f"  * '{correction['error_span']}' -> '{correction['correction']}' "
                f"(source: {correction['source_url']})"
            )

    return {
        "original": summary,
        "corrected": corrected_summary,
        "facts": facts,
        "results": all_results,
        "corrections": corrections,
    }


def run_benchmark(bench_path: str = "data/skepticbench_sample.json") -> BenchmarkReport:
    """
    Run the pipeline on SkepticBench and compute evaluation metrics.
    """
    with open(bench_path, encoding="utf-8") as handle:
        bench = json.load(handle)

    report = BenchmarkReport()

    for item in bench:
        print(f"\n{'=' * 60}")
        print(f"Benchmark item: {item['id']}")
        print(f"Summary: {item['summary']}")

        output = run_pipeline(item["summary"], verbose=True)
        injected = {error["fact"] for error in item.get("injected_errors", [])}

        for result in output["results"]:
            ground_truth = "hallucinated" if result["fact"] in injected else "correct"
            claim_result = ClaimResult(
                fact=result["fact"],
                ground_truth=ground_truth,
                verdict=result["verdict"],
                cove_applied=result.get("cove_applied", False),
                cove_meta_verdict=result.get("cove_meta_verdict"),
                correction=next(
                    (correction["correction"] for correction in output["corrections"] if correction["fact"] == result["fact"]),
                    "",
                ),
                source_url=result.get("evidence_source", ""),
            )
            report.add(claim_result)

    report.print_report()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MetaJudge AI pipeline")
    parser.add_argument("--text", type=str, help="Custom summary text to verify")
    parser.add_argument("--bench", action="store_true", help="Run on SkepticBench sample")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.getenv("GROQ_API_KEY"):
        print("\nGROQ_API_KEY not set. Create a .env file with:\n  GROQ_API_KEY=your_key_here\n")
        return 1

    if args.bench:
        run_benchmark()
        return 0

    if args.text:
        run_pipeline(args.text)
        return 0

    demo_text = (
        "BERT, introduced by Google in 2018, uses a bidirectional transformer encoder. "
        "It was pre-trained on BookCorpus and English Wikipedia, and achieved 80.5% F1 "
        "on the SQuAD 2.0 benchmark. The paper was authored by Devlin et al. and "
        "published at NAACL 2019."
    )
    run_pipeline(demo_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
