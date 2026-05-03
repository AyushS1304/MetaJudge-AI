"""
Failure analysis for MetaJudge benchmark runs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


RESULTS_PATH = Path("results/full_system_eval.json")
OUTPUT_PATH = Path("results/error_analysis.json")
ERROR_TYPES = ["metric", "author", "date", "architecture", "venue"]


def _prediction_is_positive(prediction: dict[str, Any]) -> bool:
    verdict = str(prediction.get("verdict", "")).upper()
    if verdict == "INTERNAL_CONTRADICTION":
        return True
    if verdict != "CONTRADICTED":
        return False
    if prediction.get("cove_applied"):
        return prediction.get("cove_meta_verdict") == "CONFIRMED_CONTRADICTION"
    return True


def _raw_results_by_index(raw_output: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(result.get("claim_index", index)): result
        for index, result in enumerate(raw_output.get("results", []))
    }


def _diagnose_false_negative(prediction: dict[str, Any], raw_result: dict[str, Any]) -> str:
    evidence_count = int(raw_result.get("evidence_count", 0))
    evidence_sources = set(raw_result.get("evidence_sources", []))

    if evidence_count == 0:
        return "RETRIEVAL_FAILURE"
    if raw_result.get("used_deep_verifier") and raw_result.get("disputed"):
        return "EVIDENCE_INSUFFICIENT"
    if "arxiv_direct" in evidence_sources and "arxiv_pdf" not in evidence_sources and prediction.get("verdict") == "INSUFFICIENT_EVIDENCE":
        return "EVIDENCE_INSUFFICIENT"
    return "JUDGE_FAILURE"


def analyze_failures(results, ground_truth=None) -> dict[str, Any]:
    """
    Diagnose missed detections and summarize them by corruption type.
    """
    summary: dict[str, dict[str, Any]] = {
        error_type: {
            "total": 0,
            "detected": 0,
            "missed": 0,
            "failure_causes": Counter(),
        }
        for error_type in ERROR_TYPES
    }
    missed_examples: list[dict[str, Any]] = []

    for artifact in results:
        truths = artifact.get("ground_truth", ground_truth or [])
        predictions = {item["claim_index"]: item for item in artifact.get("predictions", [])}
        raw_results = _raw_results_by_index(artifact.get("raw_output", {}))

        for truth in truths:
            error_type = truth.get("error_type")
            if error_type not in ERROR_TYPES or not truth.get("is_corrupted"):
                continue

            claim_index = int(truth["claim_index"])
            prediction = predictions.get(claim_index, {})
            raw_result = raw_results.get(claim_index, {})
            detected = _prediction_is_positive(prediction)

            summary[error_type]["total"] += 1
            if detected:
                summary[error_type]["detected"] += 1
                continue

            summary[error_type]["missed"] += 1
            cause = _diagnose_false_negative(prediction, raw_result)
            summary[error_type]["failure_causes"][cause] += 1
            missed_examples.append(
                {
                    "entry_id": artifact.get("entry_id"),
                    "paper": artifact.get("paper"),
                    "claim_index": claim_index,
                    "claim": truth.get("summary_claim"),
                    "error_type": error_type,
                    "failure_cause": cause,
                }
            )

    by_error_type = {}
    for error_type, stats in summary.items():
        total = stats["total"]
        detected = stats["detected"]
        missed = stats["missed"]
        dominant_cause = stats["failure_causes"].most_common(1)[0][0] if stats["failure_causes"] else "NONE"
        by_error_type[error_type] = {
            "detection_rate": detected / total if total else 0.0,
            "miss_rate": missed / total if total else 0.0,
            "dominant_failure_cause": dominant_cause,
            "counts": {
                "total": total,
                "detected": detected,
                "missed": missed,
                **dict(stats["failure_causes"]),
            },
        }

    return {
        "by_error_type": by_error_type,
        "missed_examples": missed_examples,
    }


def _render_table(analysis: dict[str, Any]) -> str:
    headers = ["Error Type", "Detection Rate", "Miss Rate", "Dominant Failure Cause"]
    rows = []
    for error_type in ERROR_TYPES:
        stats = analysis["by_error_type"].get(error_type, {})
        rows.append(
            [
                error_type,
                f"{stats.get('detection_rate', 0.0):.3f}",
                f"{stats.get('miss_rate', 0.0):.3f}",
                stats.get("dominant_failure_cause", "NONE"),
            ]
        )

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


if __name__ == "__main__":
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing evaluation artifact: {RESULTS_PATH}")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    analysis = analyze_failures(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(_render_table(analysis))
