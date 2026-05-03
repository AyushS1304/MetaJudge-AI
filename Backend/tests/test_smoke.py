from __future__ import annotations

import sys
import types

try:
    import arxiv  # noqa: F401
except ModuleNotFoundError:
    sys.modules["arxiv"] = types.SimpleNamespace(
        Search=lambda *args, **kwargs: types.SimpleNamespace(results=lambda: []),
        SortCriterion=types.SimpleNamespace(Relevance="relevance"),
    )

try:
    import ddgs  # noqa: F401
except ModuleNotFoundError:
    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=3):
            return []

    sys.modules["ddgs"] = types.SimpleNamespace(DDGS=_FakeDDGS)

from fastapi.testclient import TestClient

import fastapi_app
import pipeline
from modules.editor import apply_corrections_to_summary
from modules.text_utils import find_best_matching_sentence


def test_find_best_matching_sentence_prefers_best_overlap() -> None:
    text = (
        "BERT was introduced by Google in 2018. "
        "It achieved 86.7% F1 on SQuAD 2.0. "
        "The paper was authored by Devlin et al."
    )
    fragment = "BERT achieved 80.5% F1 on the SQuAD 2.0 benchmark."
    assert find_best_matching_sentence(text, fragment) == "It achieved 86.7% F1 on SQuAD 2.0."


def test_apply_corrections_to_summary_replaces_one_occurrence() -> None:
    summary = "BERT achieved 80.5% F1. Another claim says 80.5% again."
    corrections = [
        {
            "changed": True,
            "error_span": "80.5%",
            "correction": "86.7%",
        }
    ]
    corrected = apply_corrections_to_summary(summary, corrections)
    assert corrected == "BERT achieved 86.7% F1. Another claim says 80.5% again."


def test_run_pipeline_emits_events_and_applies_correction(monkeypatch) -> None:
    events: list[dict] = []

    monkeypatch.setattr(pipeline, "atomicize", lambda summary: ["BERT achieved 80.5% F1 on SQuAD 2.0."])
    monkeypatch.setattr(pipeline, "generate_skeptical_queries", lambda fact: ["query 1", "query 2"])
    monkeypatch.setattr(
        pipeline,
        "retrieve_evidence",
        lambda queries, fact="", context="": [{"source": "arxiv_direct", "title": "BERT", "url": "https://arxiv.org/abs/1810.04805", "snippet": "BERT achieves 86.7 F1 on SQuAD 2.0 dev set."}],
    )
    monkeypatch.setattr(pipeline, "format_evidence_block", lambda evidence: "evidence block")
    monkeypatch.setattr(
        pipeline,
        "judge_claim",
        lambda fact, evidence_block: {
            "verdict": "CONTRADICTED",
            "reasoning": "Claim says 80.5, evidence says 86.7.",
            "evidence_quote": "BERT achieves 86.7 F1 on SQuAD 2.0 dev set.",
            "evidence_source": "https://arxiv.org/abs/1810.04805",
            "model_used": "nvidia/llama-3.1-nemotron-70b-instruct",
        },
    )
    monkeypatch.setattr(pipeline, "deep_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "run_cove_verification",
        lambda fact, judge_result, evidence_block: {
            **judge_result,
            "verdict": "CONTRADICTED",
            "cove_applied": True,
            "cove_meta_verdict": "CONFIRMED_CONTRADICTION",
        },
    )
    monkeypatch.setattr(pipeline, "find_best_matching_sentence", lambda text, fragment: "BERT achieved 80.5% F1 on SQuAD 2.0.")
    monkeypatch.setattr(
        pipeline,
        "edit_sentence",
        lambda **kwargs: {
            "corrected_text": "BERT achieved 86.7% F1 on SQuAD 2.0.",
            "error_span": "80.5%",
            "correction": "86.7%",
            "source_url": "https://arxiv.org/abs/1810.04805",
            "changed": True,
        },
    )
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)

    result = pipeline.run_pipeline(
        "BERT achieved 80.5% F1 on SQuAD 2.0.",
        verbose=False,
        on_event=events.append,
        sleep_seconds=0.0,
    )

    assert result["corrected"] == "BERT achieved 86.7% F1 on SQuAD 2.0."
    assert result["corrections"][0]["correction"] == "86.7%"
    assert any(event["type"] == "progress" for event in events)
    assert any(event["type"] == "log" and "verdict" in event.get("message", "").lower() for event in events)


def test_fastapi_analyze_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    monkeypatch.setattr(
        fastapi_app,
        "run_pipeline",
        lambda text, verbose=False, mode="fast": {
            "original": text,
            "corrected": text,
            "facts": [text],
            "results": [],
            "corrections": [],
            "internal_contradictions": {},
        },
    )

    # Need to reimport to pick up the monkeypatched run_pipeline
    client = TestClient(fastapi_app.app)
    response = client.post("/analyze", json={"text": "Paris is the capital of France.", "mode": "fast"})

    assert response.status_code == 200
    body = response.json()
    assert body["original"] == "Paris is the capital of France."
