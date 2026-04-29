"""
Confidence calibration helpers for MetaJudge detections.
"""

from __future__ import annotations

from typing import Any


HIGH_CONFIDENCE_KEYWORDS = ("clearly", "definitely", "unambiguously")
LOW_CONFIDENCE_KEYWORDS = ("possibly", "might", "unclear")


def score_detection_confidence(reasoning: str, verdict: str | None = None) -> float:
    """
    Convert judge-style reasoning into a coarse hallucination-detection score.
    """
    text = (reasoning or "").lower()
    has_high = any(keyword in text for keyword in HIGH_CONFIDENCE_KEYWORDS)
    has_low = any(keyword in text for keyword in LOW_CONFIDENCE_KEYWORDS)

    if has_high and not has_low:
        score = 0.9
    elif has_low and not has_high:
        score = 0.4
    elif has_high and has_low:
        score = 0.55
    else:
        score = 0.65

    if verdict == "SUPPORTED":
        score = 1.0 - score
    elif verdict == "INTERNAL_CONTRADICTION":
        score = max(score, 0.9)
    elif verdict == "INSUFFICIENT_EVIDENCE":
        score = min(score, 0.5)

    return round(min(1.0, max(0.0, score)), 3)


def attach_detection_confidence(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copied result dict with `detection_confidence` attached.
    """
    enriched = dict(result)
    enriched["detection_confidence"] = score_detection_confidence(
        str(enriched.get("reasoning", "")),
        str(enriched.get("verdict", "")) or None,
    )
    return enriched
