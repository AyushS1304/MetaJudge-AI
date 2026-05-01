"""
Evaluation metrics for MetaJudge AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


POSITIVE_VERDICTS = {"CONTRADICTED", "INTERNAL_CONTRADICTION"}


def _normalise_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    if hasattr(item, key):
        return getattr(item, key)
    if isinstance(item, dict):
        return item.get(key, default)
    return default


def _prediction_is_positive(item: Any) -> bool:
    verdict = str(_get_value(item, "verdict", "")).upper()
    if verdict == "INTERNAL_CONTRADICTION":
        return True
    if verdict != "CONTRADICTED":
        return False

    cove_applied = bool(_get_value(item, "cove_applied", False))
    cove_meta_verdict = _get_value(item, "cove_meta_verdict")
    if cove_applied:
        return cove_meta_verdict == "CONFIRMED_CONTRADICTION"
    return True




def _ground_truth_correction(item: Any) -> str:
    return str(_get_value(item, "ground_truth_correction", "") or "")


def _prediction_correction(item: Any) -> str:
    return str(_get_value(item, "correction", "") or "")


def compute_metrics(predictions: list[bool], ground_truth: list[bool]) -> dict:
    TP = sum(p and g for p, g in zip(predictions, ground_truth))
    FP = sum(p and not g for p, g in zip(predictions, ground_truth))
    FN = sum(not p and g for p, g in zip(predictions, ground_truth))
    TN = sum(not p and not g for p, g in zip(predictions, ground_truth))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "false_positive_rate": round(fpr, 3),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "sample_count": len(predictions)
    }


@dataclass
class ClaimResult:
    """Result for a single atomic fact."""

    fact: str
    ground_truth: Literal["correct", "hallucinated"]
    verdict: str
    cove_applied: bool
    cove_meta_verdict: str | None
    correction: str = ""
    source_url: str = ""
    ground_truth_correction: str = ""
    detection_confidence: float = 0.65


@dataclass
class BenchmarkReport:
    results: list[ClaimResult] = field(default_factory=list)

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    cove_confirmed_contradictions: int = 0
    cove_overturned_contradictions: int = 0
    total_claims: int = 0

    def add(self, result: ClaimResult) -> None:
        self.results.append(result)
        self.total_claims += 1

        is_hallucinated = result.ground_truth == "hallucinated"
        is_detected = _prediction_is_positive(result)

        if is_hallucinated and is_detected:
            self.true_positive += 1
        elif is_hallucinated and not is_detected:
            self.false_negative += 1
        elif not is_hallucinated and is_detected:
            self.false_positive += 1
        else:
            self.true_negative += 1

        if result.cove_applied and result.cove_meta_verdict == "CONFIRMED_CONTRADICTION":
            self.cove_confirmed_contradictions += 1
        if result.cove_applied and result.cove_meta_verdict == "OVERTURNED":
            self.cove_overturned_contradictions += 1

    def precision(self) -> float:
        preds = [_prediction_is_positive(r) for r in self.results]
        truths = [r.ground_truth == "hallucinated" for r in self.results]
        return compute_metrics(preds, truths)["precision"]

    def recall(self) -> float:
        preds = [_prediction_is_positive(r) for r in self.results]
        truths = [r.ground_truth == "hallucinated" for r in self.results]
        return compute_metrics(preds, truths)["recall"]

    def f1(self) -> float:
        preds = [_prediction_is_positive(r) for r in self.results]
        truths = [r.ground_truth == "hallucinated" for r in self.results]
        return compute_metrics(preds, truths)["f1"]

    def correction_accuracy(self) -> float:
        total = matches = 0
        for r in self.results:
            if _prediction_is_positive(r) and r.ground_truth == "hallucinated":
                if r.ground_truth_correction:
                    total += 1
                    if _normalise_text(r.correction) == _normalise_text(r.ground_truth_correction):
                        matches += 1
        return matches / total if total else 0.0

    def false_positive_rate(self) -> float:
        preds = [_prediction_is_positive(r) for r in self.results]
        truths = [r.ground_truth == "hallucinated" for r in self.results]
        return compute_metrics(preds, truths)["false_positive_rate"]

    def skeptic_score(self) -> float:
        return self.cove_confirmed_contradictions / self.total_claims if self.total_claims else 0.0

    def cove_precision_gain(self) -> float:
        total_before = self.cove_confirmed_contradictions + self.cove_overturned_contradictions
        return self.cove_overturned_contradictions / total_before if total_before else 0.0

    def print_report(self) -> None:
        print("\n" + "=" * 55)
        print("  METAJUDGE BENCHMARK REPORT")
        print("=" * 55)
        print(f"  Total claims evaluated : {self.total_claims}")
        print(f"  True Positives  (TP)   : {self.true_positive}")
        print(f"  False Positives (FP)   : {self.false_positive}")
        print(f"  True Negatives  (TN)   : {self.true_negative}")
        print(f"  False Negatives (FN)   : {self.false_negative}")
        print("-" * 55)
        print(f"  Precision              : {self.precision():.3f}")
        print(f"  Recall                 : {self.recall():.3f}")
        print(f"  Detection F1           : {self.f1():.3f}")
        print(f"  Correction Accuracy    : {self.correction_accuracy():.3f}")
        print(f"  False Positive Rate    : {self.false_positive_rate():.3f}")
        print("-" * 55)
        print(f"  Skeptic Score          : {self.skeptic_score():.3f}")
        print(f"  CoVe Reversals         : {self.cove_overturned_contradictions}")
        print(f"  CoVe Precision Gain    : {self.cove_precision_gain():.3f}")
        print("=" * 55)


if __name__ == "__main__":
    report = BenchmarkReport()
    synthetic = [
        ClaimResult(
            "BERT got 80.5% on SQuAD 2.0",
            "hallucinated",
            "CONTRADICTED",
            True,
            "CONFIRMED_CONTRADICTION",
            "86.7%",
            "https://arxiv.org/abs/1810.04805",
            "86.7%",
        ),
        ClaimResult(
            "GPT-4 released in 2022",
            "hallucinated",
            "CONTRADICTED",
            True,
            "CONFIRMED_CONTRADICTION",
            "2023",
            "https://arxiv.org/abs/2303.08774",
            "2023",
        ),
        ClaimResult(
            "Attention is All You Need, 2017",
            "correct",
            "SUPPORTED",
            False,
            None,
        ),
        ClaimResult(
            "FActScore by Min et al.",
            "correct",
            "CONTRADICTED",
            True,
            "OVERTURNED",
        ),
    ]
    for item in synthetic:
        report.add(item)
    report.print_report()
