"""
baselines/baseline_standard_rag.py — Baseline 1: Standard RAG
--------------------------------------------------------------
The naive approach this project improves upon.

Key differences from Skeptical CoVe-RAG:
  ✗ Queries ask "Is X true?" (supportive, not adversarial)
  ✗ No CoVe verification — judge's word is final
  ✗ No atomic decomposition — verifies full sentences
  ✗ No surgical editor — no correction step
"""

import re
import json
import time

from modules.nvidia_client import nvidia_chat
from modules.retriever import retrieve_evidence, format_evidence_block

JUDGE_PROMPT = """You are a fact-checker. Given a claim and evidence, decide:
- SUPPORTED: evidence confirms the claim
- CONTRADICTED: evidence clearly contradicts the claim
- INSUFFICIENT_EVIDENCE: cannot determine from evidence

Return ONLY JSON: {"verdict": "...", "reasoning": "one sentence"}
"""

def _supportive_query(sentence: str) -> str:
    """Standard RAG: generate a query to CONFIRM the claim."""
    return sentence

def _judge_sentence(sentence: str, evidence_block: str) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"CLAIM: {sentence}\n\nEVIDENCE: {evidence_block}\n\nReturn JSON."},
    ]
    try:
        raw = nvidia_chat(messages, role="judge", temperature=0.0, max_tokens=256)
    except Exception:
        raw = nvidia_chat(messages, role="fast", temperature=0.0, max_tokens=256)

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reasoning": "Parse error."}

def run_standard_rag(
    summary: str,
    verbose: bool = True,
    *,
    sentences_override: list[str] | None = None,
    sleep_seconds: float = 0.3,
) -> dict:
    """
    Run Standard RAG pipeline (baseline).
    Works at sentence level, no adversarial queries, no CoVe.
    """
    sentences = (
        [sentence.strip() for sentence in sentences_override if str(sentence).strip()]
        if sentences_override
        else [s.strip() + "." for s in summary.replace(".\n", ". ").split(". ") if len(s.strip()) > 10]
    )
    results = []

    if verbose:
        print("\n[Baseline: Standard RAG]")

    for index, sent in enumerate(sentences):
        query    = _supportive_query(sent)
        evidence = retrieve_evidence([query])
        ev_block = format_evidence_block(evidence)
        verdict  = _judge_sentence(sent, ev_block)

        results.append({
            "claim_index": index,
            "sentence": sent,
            "verdict":  verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
            "reasoning": verdict.get("reasoning", ""),
            "correction": "",
        })
        if verbose:
            print(f"  {verdict.get('verdict','?'):28s} | {sent[:70]}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    contradictions = [r for r in results if r["verdict"] == "CONTRADICTED"]
    return {
        "summary":        summary,
        "results":        results,
        "n_contradicted": len(contradictions),
        "n_total":        len(sentences),
    }


if __name__ == "__main__":
    demo = (
        "BERT, introduced by Google in 2018, achieved 80.5% F1 on SQuAD 2.0. "
        "GPT-4 was released by OpenAI in March 2022."
    )
    run_standard_rag(demo)
