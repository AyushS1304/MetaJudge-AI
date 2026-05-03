"""
evaluation/skeptic_score.py — Evaluation Metrics
-------------------------------------------------
Computes the two primary metrics for benchmarking on SkepticBench:

1. Detection F1:
   Standard precision/recall/F1 of identifying injected hallucinations.

2. Skeptic Score (novel metric):
   Ratio of successful ADVERSARIAL falsifications to total claims.
   Unlike F1, this rewards proactive skepticism — finding errors through
   evidence-backed falsification, not just pattern matching.

   Skeptic Score = (Correctly_Contradicted_and_Confirmed_by_CoVe) / (Total_Claims)

   A higher Skeptic Score means the system finds errors through
   real evidence, not hallucinated reasoning.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ClaimResult:
    """Result for a single atomic fact."""
    fact:             str
    ground_truth:     Literal["correct", "hallucinated"]  # from SkepticBench label
    verdict:          str    # SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
    cove_applied:     bool
    cove_meta_verdict: str | None
    correction:       str = ""
    source_url:       str = ""


@dataclass
class BenchmarkReport:
    results:        list[ClaimResult] = field(default_factory=list)

    # --- Detection F1 counters ---
    true_positive:  int = 0   # Hallucination correctly detected (CONTRADICTED + CoVe confirmed)
    false_positive: int = 0   # Correct claim incorrectly flagged as hallucination
    true_negative:  int = 0   # Correct claim correctly passed
    false_negative: int = 0   # Hallucination missed (SUPPORTED or INSUFFICIENT)

    # --- Skeptic Score counters ---
    cove_confirmed_contradictions: int = 0   # Contradictions that survived CoVe
    cove_overturned_contradictions: int = 0  # Hallucinated judge decisions caught by CoVe
    total_claims:   int = 0

    def add(self, result: ClaimResult):
        self.results.append(result)
        self.total_claims += 1

        is_hallucinated = result.ground_truth == "hallucinated"
        # A detection counts only if CoVe confirmed it (prevents gaming with hallucinated verdicts)
        is_detected = (
            result.verdict == "CONTRADICTED"
            and result.cove_meta_verdict == "CONFIRMED_CONTRADICTION"
        )

        if is_hallucinated and is_detected:
            self.true_positive += 1
            self.cove_confirmed_contradictions += 1
        elif is_hallucinated and not is_detected:
            self.false_negative += 1
        elif not is_hallucinated and is_detected:
            self.false_positive += 1
        elif not is_hallucinated and not is_detected:
            self.true_negative += 1

        if result.cove_applied and result.cove_meta_verdict == "OVERTURNED":
            self.cove_overturned_contradictions += 1

    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else 0.0

    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def skeptic_score(self) -> float:
        """
        Novel metric: ratio of evidence-backed falsifications to total claims.
        Rewards finding errors through CoVe-confirmed adversarial retrieval.
        """
        return self.cove_confirmed_contradictions / self.total_claims if self.total_claims > 0 else 0.0

    def cove_precision_gain(self) -> float:
        """
        How much did CoVe improve precision?
        = overturned_contradictions / total_contradictions_before_cove
        """
        total_before = self.cove_confirmed_contradictions + self.cove_overturned_contradictions
        return self.cove_overturned_contradictions / total_before if total_before > 0 else 0.0

    def print_report(self):
        print("\n" + "="*55)
        print("  SKEPTICAL CoVe-RAG — BENCHMARK REPORT")
        print("="*55)
        print(f"  Total claims evaluated : {self.total_claims}")
        print(f"  True Positives  (TP)   : {self.true_positive}")
        print(f"  False Positives (FP)   : {self.false_positive}")
        print(f"  True Negatives  (TN)   : {self.true_negative}")
        print(f"  False Negatives (FN)   : {self.false_negative}")
        print("-"*55)
        print(f"  Precision              : {self.precision():.3f}")
        print(f"  Recall                 : {self.recall():.3f}")
        print(f"  Detection F1           : {self.f1():.3f}")
        print("-"*55)
        print(f"  Skeptic Score  (novel) : {self.skeptic_score():.3f}")
        print(f"  CoVe Reversals         : {self.cove_overturned_contradictions}")
        print(f"  CoVe Precision Gain    : {self.cove_precision_gain():.3f}")
        print("="*55)


if __name__ == "__main__":
    # Demo with synthetic results
    report = BenchmarkReport()

    synthetic = [
        ClaimResult("BERT got 80.5% on SQuAD 2.0",   "hallucinated", "CONTRADICTED",      True,  "CONFIRMED_CONTRADICTION", "86.7% F1", "arxiv.org/abs/1810.04805"),
        ClaimResult("GPT-4 released in 2022",         "hallucinated", "CONTRADICTED",      True,  "CONFIRMED_CONTRADICTION", "2023",     "openai.com/gpt-4"),
        ClaimResult("Attention is All You Need, 2017","correct",      "SUPPORTED",         False, None,                      "",         ""),
        ClaimResult("LLaMA 2 uses RLHF",              "correct",      "SUPPORTED",         False, None,                      "",         ""),
        ClaimResult("FActScore by Min et al.",         "correct",      "CONTRADICTED",      True,  "OVERTURNED",              "",         ""),
        ClaimResult("RARR by Gao et al. 2022",        "hallucinated", "INSUFFICIENT_EVIDENCE", False, None,                  "",         ""),
    ]
    for r in synthetic:
        report.add(r)

    report.print_report()