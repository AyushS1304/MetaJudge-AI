"""
Cross-claim consistency checking for MetaJudge AI.

This module checks internal contradictions between atomic claims from the same
summary without any retrieval. That cross-claim consistency graph is a novel
addition here and is absent from RARR, FActScore, and SAFE.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any

from modules.nvidia_client import nvidia_chat, MODELS

CONSISTENCY_MODEL = MODELS["fast"]
VALID_CONTRADICTION_TYPES = {"numerical", "temporal", "categorical", "relational"}


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _parse_model_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from consistency checker.")
    return parsed


def check_cross_claim_consistency(claims: list[str], _groq_client=None) -> dict[str, Any]:
    """
    Check all atomic-claim pairs for internal contradictions.

    Uses `itertools.combinations` over all N*(N-1)/2 pairs, processes them in
    batches of five pairs, and silently skips failed pair checks.

    The _groq_client parameter is kept for API compatibility but is ignored;
    all calls now go through nvidia_chat.
    """
    indexed_claims = list(enumerate(claims))
    claim_pairs = list(combinations(indexed_claims, 2))
    contradicting_pairs: list[tuple[int, int, str, str, str, str]] = []
    graph_edges: list[tuple[int, int]] = []

    if not claim_pairs:
        return {
            "contradicting_pairs": contradicting_pairs,
            "consistency_score": 1.0,
            "graph_edges": graph_edges,
            "summary": "Not enough atomic claims for cross-claim consistency checking.",
        }

    for batch in _batched(claim_pairs, 5):
        for (index_a, claim_a), (index_b, claim_b) in batch:
            prompt = (
                "You are checking only INTERNAL consistency between two claims from the same summary. "
                "Use no external knowledge. Ask only whether Claim A and Claim B cannot both be true at the same time. "
                "If both claims could coexist, even if one or both might be false in the real world, return contradicts=false. "
                "Different topics are not contradictions. "
                f'Do Claim A: {claim_a} and Claim B: {claim_b} CONTRADICT? '
                'JSON only: {"contradicts":bool,"reason":str,'
                '"contradiction_type":"numerical|temporal|categorical|relational"}'
            )
            try:
                raw = nvidia_chat(
                    [{"role": "user", "content": prompt}],
                    role="fast",
                    temperature=0.0,
                    max_tokens=180,
                )
                parsed = _parse_model_json(raw)
            except Exception:
                continue

            if bool(parsed.get("contradicts")):
                contradiction_type = str(parsed.get("contradiction_type", "relational")).strip().lower()
                if contradiction_type not in VALID_CONTRADICTION_TYPES:
                    contradiction_type = "relational"
                reason = str(parsed.get("reason", "")).strip()
                contradicting_pairs.append(
                    (index_a, index_b, claim_a, claim_b, reason, contradiction_type)
                )
                graph_edges.append((index_a, index_b))

    total_pairs = len(claim_pairs)
    contradiction_count = len(graph_edges)
    consistency_score = 1.0 - (contradiction_count / total_pairs if total_pairs else 0.0)

    if contradiction_count == 0:
        summary = f"No internal contradictions found across {total_pairs} claim pairs."
    else:
        involved_claims = len({node for edge in graph_edges for node in edge})
        summary = (
            f"Found {contradiction_count} contradicting claim pairs across {total_pairs} comparisons; "
            f"{involved_claims} claims participate in at least one contradiction."
        )

    return {
        "contradicting_pairs": contradicting_pairs,
        "consistency_score": round(max(0.0, consistency_score), 4),
        "graph_edges": graph_edges,
        "summary": summary,
    }
