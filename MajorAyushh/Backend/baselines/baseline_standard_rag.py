"""
baselines/baseline_standard_rag.py — Baseline 1: Standard RAG
--------------------------------------------------------------
The naive approach this project improves upon.

Key differences from Skeptical CoVe-RAG:
  ✗ Queries ask "Is X true?" (supportive, not adversarial)
  ✗ No CoVe verification — judge's word is final
  ✗ No atomic decomposition — verifies full sentences
  ✗ No surgical editor — no correction step

Used in ablation: removing BOTH adversarial retrieval AND CoVe loop.
"""

import re
import json
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from modules.retriever import retrieve_evidence, format_evidence_block
from config import GROQ_API_KEY, STRONG_MODEL

client = Groq(api_key=GROQ_API_KEY)

JUDGE_PROMPT = """You are a fact-checker. Given a claim and evidence, decide:
- SUPPORTED: evidence confirms the claim
- CONTRADICTED: evidence clearly contradicts the claim
- INSUFFICIENT_EVIDENCE: cannot determine from evidence

Return ONLY JSON: {"verdict": "...", "reasoning": "one sentence"}
"""

def _supportive_query(sentence: str) -> str:
    """Standard RAG: generate a query to CONFIRM the claim."""
    return sentence  # Just use the claim itself as the query

def _judge_sentence(sentence: str, evidence_block: str) -> dict:
    raw = client.chat.completions.create(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",   "content": f"CLAIM: {sentence}\n\nEVIDENCE: {evidence_block}\n\nReturn JSON."}
        ],
        temperature=0.0,
        max_tokens=256,
    ).choices[0].message.content.strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reasoning": "Parse error."}

def run_standard_rag(summary: str, verbose: bool = True) -> dict:
    """
    Run Standard RAG pipeline (baseline).
    Works at sentence level, no adversarial queries, no CoVe.
    """
    sentences = [s.strip() + "." for s in summary.replace(".\n", ". ").split(". ") if len(s.strip()) > 10]
    results = []

    if verbose:
        print("\n[Baseline: Standard RAG]")

    for sent in sentences:
        query    = _supportive_query(sent)
        evidence = retrieve_evidence([query])
        ev_block = format_evidence_block(evidence)
        verdict  = _judge_sentence(sent, ev_block)

        results.append({
            "sentence": sent,
            "verdict":  verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
            "reasoning": verdict.get("reasoning", ""),
        })
        if verbose:
            print(f"  {verdict.get('verdict','?'):28s} | {sent[:70]}")
        time.sleep(0.3)

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