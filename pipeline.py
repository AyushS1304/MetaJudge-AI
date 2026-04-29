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
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import modules.judge as judge_module
from config import MAX_FACTS
from evaluation.skeptic_score import BenchmarkReport, ClaimResult
from modules.atomicizer import atomicize
from modules.confidence_scorer import attach_detection_confidence
from modules.consistency_checker import check_cross_claim_consistency
from modules.cove_loop import run_cove_verification
from modules.deep_verifier import deep_verify
from modules.editor import apply_corrections_to_summary, edit_sentence
from modules.judge import VERDICT_CONTRADICTED, VERDICT_INSUFFICIENT, judge_claim
from modules.query_generator import generate_skeptical_queries
from modules.retriever import KNOWN_PAPERS, format_evidence_block, retrieve_evidence
from modules.text_utils import find_best_matching_sentence


SEPARATOR = "=" * 60
VERDICT_INTERNAL_CONTRADICTION = "INTERNAL_CONTRADICTION"
PipelineEventCallback = Callable[[dict[str, Any]], None]


def _emit(callback: PipelineEventCallback | None, event_type: str, **payload: Any) -> None:
    if callback is None:
        return
    callback({"type": event_type, **payload})


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _standard_queries(fact: str) -> list[str]:
    return [fact, f"official information about {fact}"]


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


def _build_internal_contradiction_result(
    fact: str,
    claim_index: int,
    linked_claims: list[tuple[int, str, str]],
    *,
    used_consistency_checker: bool,
) -> dict[str, Any]:
    related_reasons = [
        f"claim {other_index + 1} ({contradiction_type}): {reason or 'internal contradiction detected'}"
        for other_index, reason, contradiction_type in linked_claims
    ]
    reasoning = (
        "Clearly internally contradictory with "
        + "; ".join(related_reasons)
        if related_reasons
        else "Clearly internally contradictory with another claim in the same summary."
    )

    return attach_detection_confidence(
        {
            "fact": fact,
            "claim_index": claim_index,
            "verdict": VERDICT_INTERNAL_CONTRADICTION,
            "reasoning": reasoning,
            "evidence_quote": "",
            "evidence_source": "",
            "cove_applied": False,
            "cove_meta_verdict": None,
            "disputed": False,
            "gemini_used": False,
            "pdf_used": False,
            "queries": [],
            "evidence_count": 0,
            "evidence_sources": [],
            "used_adversarial_queries": False,
            "used_deep_verifier": False,
            "used_consistency_checker": used_consistency_checker,
        }
    )


def _should_apply_editor(result: dict[str, Any], *, use_cove: bool) -> bool:
    if result.get("verdict") != VERDICT_CONTRADICTED:
        return False
    if not use_cove:
        return True
    return result.get("cove_meta_verdict") == "CONFIRMED_CONTRADICTION"


def run_pipeline(
    summary: str,
    verbose: bool = True,
    *,
    gemini_key: str | None = None,
    on_event: PipelineEventCallback | None = None,
    sleep_seconds: float = 0.3,
    use_consistency_checker: bool = True,
    use_adversarial_queries: bool = True,
    use_deep_verifier: bool = True,
    use_cove: bool = True,
    facts_override: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the full MetaJudge pipeline on a summary.
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

    facts = [fact.strip() for fact in (facts_override or atomicize(summary)) if str(fact).strip()][:MAX_FACTS]
    _emit(on_event, "log", kind="done", message=f"  -> {len(facts)} atomic facts extracted")
    _log(verbose, f"  -> {len(facts)} atomic facts extracted.")

    _emit(on_event, "log", kind="step", message="Step 1.5: Cross-Claim Consistency Check")
    _log(verbose, "  [Step 1.5] Cross-Claim Consistency Check...")

    internal_contradictions = {
        "contradicting_pairs": [],
        "consistency_score": 1.0,
        "graph_edges": [],
        "summary": "Cross-claim consistency check skipped.",
    }
    contradiction_map: dict[int, list[tuple[int, str, str]]] = defaultdict(list)

    if use_consistency_checker and len(facts) > 1:
        internal_contradictions = check_cross_claim_consistency(facts, judge_module.client)
        for i, j, _, _, reason, contradiction_type in internal_contradictions["contradicting_pairs"]:
            contradiction_map[i].append((j, reason, contradiction_type))
            contradiction_map[j].append((i, reason, contradiction_type))
    elif len(facts) <= 1:
        internal_contradictions["summary"] = "Not enough atomic claims for cross-claim consistency checking."

    _emit(on_event, "log", kind="done", message=f"  -> {internal_contradictions['summary']}")
    _log(verbose, f"  -> {internal_contradictions['summary']}")

    corrections: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    total_facts = len(facts)
    gemini_requested = bool((gemini_key or os.environ.get("GEMINI_API_KEY")) and use_deep_verifier)

    for index, fact in enumerate(facts, start=1):
        claim_index = index - 1
        progress = 0.1 + (claim_index / max(total_facts, 1)) * 0.85
        _emit(on_event, "progress", value=progress, label=f"Processing fact {index} of {total_facts}...")
        _emit(on_event, "log", kind="info", message="")
        _emit(on_event, "log", kind="info", message=f"-- Fact {index}/{total_facts} -------------------------")
        _emit(on_event, "log", kind="fact", message=f"  {fact[:80]}{'...' if len(fact) > 80 else ''}")

        _log(verbose, f"\n{SEPARATOR}")
        _log(verbose, f"Fact {index}/{total_facts}: {fact}")

        if claim_index in contradiction_map:
            internal_result = _build_internal_contradiction_result(
                fact,
                claim_index,
                contradiction_map[claim_index],
                used_consistency_checker=use_consistency_checker,
            )
            all_results.append(internal_result)
            _emit(
                on_event,
                "log",
                kind="err",
                message="  -> Retrieval skipped due to internal contradiction with another claim",
            )
            _log(verbose, "  [Steps 2-6] Skipped due to internal contradiction.")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        step_2_message = (
            "Step 2: Generating adversarial queries..."
            if use_adversarial_queries
            else "Step 2: Generating standard search queries..."
        )
        _emit(on_event, "log", kind="step", message=step_2_message)
        _log(
            verbose,
            "  [Step 2] Generating adversarial queries..."
            if use_adversarial_queries
            else "  [Step 2] Generating standard search queries...",
        )
        queries = generate_skeptical_queries(fact) if use_adversarial_queries else _standard_queries(fact)
        for query in queries:
            _emit(on_event, "log", kind="info", message=f"  -> {query[:75]}")
            _log(verbose, f"    -> {query}")

        _emit(on_event, "log", kind="step", message="Step 3: Retrieving evidence...")
        _log(verbose, "  [Step 3] Retrieving evidence (arXiv + web)...")
        evidence = retrieve_evidence(queries, fact=fact, context=summary)
        evidence_block = format_evidence_block(evidence)
        evidence_sources = [item.get("source", "") for item in evidence]
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
        deep_verifier_used = False
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

        if judge_result["verdict"] == VERDICT_INSUFFICIENT and use_deep_verifier:
            deep_verifier_used = True
            _emit(
                on_event,
                "log",
                kind="gemini" if gemini_requested else "step",
                message="Step 4b: Deep verification...",
            )
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
        else:
            judge_result.setdefault("disputed", False)
            judge_result.setdefault("gemini_used", False)
            judge_result.setdefault("pdf_used", False)
            if judge_result["verdict"] == VERDICT_INSUFFICIENT:
                _emit(on_event, "log", kind="step", message="Step 4b: Deep verification skipped")
                _log(verbose, "  [Step 4b] Deep verification skipped.")

        if judge_result["verdict"] == VERDICT_CONTRADICTED and use_cove:
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
            _emit(
                on_event,
                "log",
                kind="step",
                message="Step 5: CoVe skipped" if use_cove else "Step 5: CoVe disabled",
            )
            _log(
                verbose,
                f"  [Step 5] CoVe skipped (verdict is {judge_result['verdict']})."
                if use_cove
                else "  [Step 5] CoVe disabled by configuration.",
            )

        final_result.update(
            {
                "fact": fact,
                "claim_index": claim_index,
                "queries": queries,
                "evidence_count": len(evidence),
                "evidence_sources": evidence_sources,
                "used_adversarial_queries": use_adversarial_queries,
                "used_deep_verifier": deep_verifier_used,
                "used_consistency_checker": use_consistency_checker,
            }
        )
        final_result = attach_detection_confidence(final_result)
        all_results.append(final_result)

        if _should_apply_editor(final_result, use_cove=use_cove):
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
                        "claim_index": claim_index,
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
    contradicted_count = sum(
        1
        for result in all_results
        if result["verdict"] in {VERDICT_CONTRADICTED, VERDICT_INTERNAL_CONTRADICTION}
    )
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
        "internal_contradictions": internal_contradictions,
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

        output = run_pipeline(
            item["summary"],
            verbose=True,
            facts_override=item.get("corrupted_facts") or item.get("atomic_facts"),
        )
        injected_by_fact = {error["fact"]: error for error in item.get("injected_errors", [])}

        for result in output["results"]:
            injected = injected_by_fact.get(result["fact"])
            claim_result = ClaimResult(
                fact=result["fact"],
                ground_truth="hallucinated" if injected else "correct",
                verdict=result["verdict"],
                cove_applied=result.get("cove_applied", False),
                cove_meta_verdict=result.get("cove_meta_verdict"),
                correction=next(
                    (
                        correction["correction"]
                        for correction in output["corrections"]
                        if correction["fact"] == result["fact"]
                    ),
                    "",
                ),
                source_url=result.get("evidence_source", ""),
                ground_truth_correction=(injected or {}).get("ground_truth_correction", ""),
                detection_confidence=float(result.get("detection_confidence", 0.65)),
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
        result = run_pipeline(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    demo_text = (
        "BERT, introduced by Google in 2018, uses a bidirectional transformer encoder. "
        "It was pre-trained on BookCorpus and English Wikipedia, and achieved 80.5% F1 "
        "on the SQuAD 2.0 benchmark. The paper was authored by Devlin et al. and "
        "published at NAACL 2019."
    )
    result = run_pipeline(demo_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
