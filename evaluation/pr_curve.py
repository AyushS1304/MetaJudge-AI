"""
Precision-recall curve generation from MetaJudge detection confidences.
"""

from __future__ import annotations

import json
from pathlib import Path


RESULTS_PATH = Path("results/full_system_eval.json")
OUTPUT_PATH = Path("results/pr_curve.png")


def _flatten_scores(artifacts: list[dict]) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for artifact in artifacts:
        truths = {item["claim_index"]: item for item in artifact.get("ground_truth", [])}
        for prediction in artifact.get("predictions", []):
            truth = truths.get(prediction["claim_index"])
            if truth is None:
                continue
            scores.append(float(prediction.get("detection_confidence", 0.0)))
            labels.append(1 if truth.get("is_corrupted") else 0)
    return scores, labels


def _precision_recall_points(scores: list[float], labels: list[int]) -> tuple[list[float], list[float]]:
    thresholds = sorted(set(scores), reverse=True)
    if not thresholds:
        return [1.0], [0.0]
    if thresholds[-1] != 0.0:
        thresholds.append(0.0)

    precision_points: list[float] = []
    recall_points: list[float] = []
    positives = sum(labels)

    for threshold in thresholds:
        predicted_positive = [score >= threshold for score in scores]
        true_positive = sum(
            1 for predicted, label in zip(predicted_positive, labels) if predicted and label == 1
        )
        false_positive = sum(
            1 for predicted, label in zip(predicted_positive, labels) if predicted and label == 0
        )
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
        recall = true_positive / positives if positives else 0.0
        precision_points.append(precision)
        recall_points.append(recall)

    return precision_points, recall_points


def generate_pr_curve() -> Path:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing evaluation artifact: {RESULTS_PATH}")

    import matplotlib.pyplot as plt

    artifacts = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    scores, labels = _flatten_scores(artifacts)
    precision, recall = _precision_recall_points(scores, labels)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker="o", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("MetaJudge Precision-Recall Curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    return OUTPUT_PATH


if __name__ == "__main__":
    path = generate_pr_curve()
    print(f"Saved PR curve to {path}")
