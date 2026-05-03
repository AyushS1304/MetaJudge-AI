"""
Generate system-comparison and ablation tables for MetaJudge.
"""

from __future__ import annotations

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baseline_rarr import run_vanilla_rarr
from baselines.baseline_standard_rag import run_standard_rag
from baselines.baseline_zeroshot import run_zeroshot
from evaluation.skeptic_score import compute_metrics
from pipeline import run_pipeline


BENCH_PATH = "data/skepticbench_full.json"
RESULTS_DIR = Path("results")
TABLE1_PATH = RESULTS_DIR / "table1.txt"
TABLE2_PATH = RESULTS_DIR / "table2.txt"
FULL_SYSTEM_ARTIFACTS_PATH = RESULTS_DIR / "full_system_eval.json"


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_benchmark(bench_path: str) -> list[dict[str, Any]]:
    with open(bench_path, encoding="utf-8") as handle:
        return json.load(handle)


def _extract_correction_value(correct: str, wrong: str) -> str:
    correct_tokens = correct.split()
    wrong_tokens = wrong.split()
    matcher = SequenceMatcher(a=wrong_tokens, b=correct_tokens)
    replacement_tokens: list[str] = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            replacement_tokens.extend(correct_tokens[j1:j2])
    extracted = " ".join(replacement_tokens).strip(" ,.;:()")
    return extracted or correct


def _claim_annotations(entry: dict[str, Any]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    injected_by_fact = {item.get("fact"): item for item in entry.get("injected_errors", [])}
    atomic_facts = entry.get("atomic_facts", [])
    corrupted_facts = entry.get("corrupted_facts", atomic_facts)
    labels = entry.get("labels", [])

    for index, summary_claim in enumerate(corrupted_facts):
        correct_claim = atomic_facts[index] if index < len(atomic_facts) else summary_claim
        is_corrupted = index < len(labels) and str(labels[index]).lower() == "false"
        injected = injected_by_fact.get(summary_claim, {})
        annotation = {
            "claim_index": index,
            "summary_claim": summary_claim,
            "ground_truth_claim": correct_claim,
            "ground_truth": "hallucinated" if is_corrupted else "correct",
            "is_corrupted": is_corrupted,
            "error_type": injected.get("type", "clean_no_error" if not is_corrupted else "metric"),
        }
        if is_corrupted:
            annotation["ground_truth_correction"] = _extract_correction_value(correct_claim, summary_claim)
        annotations.append(annotation)

    return annotations


def _ground_truth_for_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(annotation) for annotation in _claim_annotations(entry)]


def _normalise_predictions(
    results: list[dict[str, Any]],
    corrections_lookup: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    corrections_lookup = corrections_lookup or {}

    for fallback_index, result in enumerate(results):
        claim_index = int(result.get("claim_index", fallback_index))
        predictions.append(
            {
                "claim_index": claim_index,
                "fact": result.get("fact") or result.get("sentence", ""),
                "verdict": result.get("verdict", "INSUFFICIENT_EVIDENCE"),
                "cove_applied": result.get("cove_applied", False),
                "cove_meta_verdict": result.get("cove_meta_verdict"),
                "correction": result.get("correction", corrections_lookup.get(claim_index, "")),
                "reasoning": result.get("reasoning", ""),
                "detection_confidence": float(result.get("detection_confidence", 0.65)),
            }
        )

    return sorted(predictions, key=lambda item: item["claim_index"])


def _placeholder_predictions(truths: list[dict[str, Any]], error_message: str) -> list[dict[str, Any]]:
    return [
        {
            "claim_index": truth["claim_index"],
            "fact": truth["summary_claim"],
            "verdict": "INSUFFICIENT_EVIDENCE",
            "cove_applied": False,
            "cove_meta_verdict": None,
            "correction": "",
            "reasoning": error_message,
            "detection_confidence": 0.0,
        }
        for truth in truths
    ]


def _run_full_system(entry: dict[str, Any], **pipeline_flags: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = [annotation["summary_claim"] for annotation in _claim_annotations(entry)]
    output = run_pipeline(
        entry["summary"],
        verbose=False,
        sleep_seconds=0.0,
        facts_override=facts,
        **pipeline_flags,
    )
    corrections_lookup = {
        int(correction.get("claim_index", -1)): correction.get("correction", "")
        for correction in output.get("corrections", [])
    }
    return _normalise_predictions(output.get("results", []), corrections_lookup), output


def _run_zeroshot_system(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = [annotation["summary_claim"] for annotation in _claim_annotations(entry)]
    output = run_zeroshot(entry["summary"], verbose=False, facts_override=facts, sleep_seconds=0.0)
    return _normalise_predictions(output.get("results", [])), output


def _run_standard_rag_system(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = [annotation["summary_claim"] for annotation in _claim_annotations(entry)]
    output = run_standard_rag(entry["summary"], verbose=False, sentences_override=facts, sleep_seconds=0.0)
    return _normalise_predictions(output.get("results", [])), output


def _run_vanilla_rarr_system(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = [annotation["summary_claim"] for annotation in _claim_annotations(entry)]
    output = run_vanilla_rarr(entry["summary"], verbose=False, facts_override=facts, sleep_seconds=0.0)
    return _normalise_predictions(output.get("results", [])), output


def _evaluate_configuration(
    bench: list[dict[str, Any]],
    runner: Callable[[dict[str, Any]], tuple[list[dict[str, Any]], dict[str, Any]]],
    label: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    all_predictions: list[dict[str, Any]] = []
    all_truths: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    print(f"\n[{label}] evaluating {len(bench)} entries")
    for index, entry in enumerate(bench, start=1):
        truths = _ground_truth_for_entry(entry)
        try:
            predictions, raw_output = runner(entry)
            if len(predictions) != len(truths):
                raise ValueError(
                    f"Prediction/ground-truth length mismatch for {entry.get('id', '?')}: "
                    f"{len(predictions)} vs {len(truths)}"
                )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"  - {entry.get('id', '?')}: {error_message}")
            predictions = _placeholder_predictions(truths, error_message)
            raw_output = {"error": error_message}

        all_predictions.extend(predictions)
        all_truths.extend(truths)
        artifacts.append(
            {
                "entry_id": entry.get("id"),
                "paper": entry.get("paper"),
                "predictions": predictions,
                "ground_truth": truths,
                "raw_output": raw_output,
            }
        )
        print(f"  - {index:02d}/{len(bench):02d} {entry.get('id', '?')} complete")

    return compute_metrics(all_predictions, all_truths), artifacts


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [separator, render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _save_text(path: Path, content: str) -> None:
    path.write_text(content + "\n", encoding="utf-8")


def run_full_comparison(bench_path: str = BENCH_PATH) -> dict[str, Any]:
    ensure_results_dir()
    bench = _load_benchmark(bench_path)

    systems = {
        "Zero-Shot LLM": _run_zeroshot_system,
        "Standard RAG": _run_standard_rag_system,
        "Vanilla RARR": _run_vanilla_rarr_system,
        "MetaJudge": _run_full_system,
    }

    system_metrics: dict[str, dict[str, float]] = {}
    system_artifacts: dict[str, list[dict[str, Any]]] = {}
    table1_rows: list[list[str]] = []

    for system_name, runner in systems.items():
        metrics, artifacts = _evaluate_configuration(bench, runner, system_name)
        system_metrics[system_name] = metrics
        system_artifacts[system_name] = artifacts
        table1_rows.append(
            [
                system_name,
                _format_metric(metrics["precision"]),
                _format_metric(metrics["recall"]),
                _format_metric(metrics["f1"]),
                _format_metric(metrics["correction_acc"]),
                _format_metric(metrics["false_positive_rate"]),
            ]
        )

    FULL_SYSTEM_ARTIFACTS_PATH.write_text(
        json.dumps(system_artifacts["MetaJudge"], indent=2),
        encoding="utf-8",
    )

    table1 = _ascii_table(
        ["System", "Precision", "Recall", "F1", "Correction-Acc", "FP-Rate"],
        table1_rows,
    )
    print("\nTABLE 1: System Comparison")
    print(table1)
    _save_text(TABLE1_PATH, table1)

    full_metrics = system_metrics["MetaJudge"]
    ablations = {
        "Full system": {},
        "- Adversarial queries (standard search only)": {"use_adversarial_queries": False},
        "- CoVe meta-judge (no Step 5)": {"use_cove": False},
        "- Deep verifier / escalation (no Step 4b)": {"use_deep_verifier": False},
        "- Consistency checker (no Step 1.5)": {"use_consistency_checker": False},
    }

    ablation_metrics: dict[str, dict[str, float]] = {"Full system": full_metrics}
    table2_rows: list[list[str]] = [
        ["Full system", _format_metric(full_metrics["f1"]), f"{0.0:+.3f}"]
    ]

    for label, flags in ablations.items():
        if label == "Full system":
            continue
        metrics, _ = _evaluate_configuration(
            bench,
            lambda entry, flags=flags: _run_full_system(entry, **flags),
            label,
        )
        ablation_metrics[label] = metrics
        table2_rows.append(
            [
                label,
                _format_metric(metrics["f1"]),
                f"{metrics['f1'] - full_metrics['f1']:+.3f}",
            ]
        )

    table2 = _ascii_table(
        ["Configuration", "F1", "Delta-vs-Full"],
        table2_rows,
    )
    print("\nTABLE 2: Ablation")
    print(table2)
    _save_text(TABLE2_PATH, table2)

    return {
        "table1_metrics": system_metrics,
        "table2_metrics": ablation_metrics,
        "full_system_artifacts": system_artifacts["MetaJudge"],
    }


if __name__ == "__main__":
    if not os.path.exists(BENCH_PATH):
        raise SystemExit(f"Missing benchmark file: {BENCH_PATH}")
    run_full_comparison()
