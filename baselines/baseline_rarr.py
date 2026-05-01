"""
baselines/baseline_rarr.py — Baseline 2: Vanilla RARR
-------------------------------------------------------
Retrieval Augmented Revision and Rewriting (Gao et al., 2023).

Improvements over Standard RAG:
  ✓ Atomic decomposition
  ✓ Web retrieval per fact
  ✓ Rewrites incorrect sentences

Still missing vs Skeptical CoVe-RAG:
  ✗ Queries are still supportive, not adversarial
  ✗ No CoVe — judge is trusted blindly
  ✗ Destructive correction — rewrites full sentences, not surgical
"""

import re
import json
import time

from modules.nvidia_client import nvidia_chat
from modules.atomicizer  import atomicize
from modules.retriever   import retrieve_evidence, format_evidence_block
from modules.text_utils import find_best_matching_sentence
from config import MAX_FACTS

JUDGE_PROMPT = """You are a fact-checker. Given an atomic claim and evidence, judge:
- SUPPORTED: evidence confirms the claim
- CONTRADICTED: evidence contradicts the claim
- INSUFFICIENT_EVIDENCE: cannot determine

Return ONLY JSON: {"verdict": "...", "reasoning": "one sentence", "correction": "corrected fact or empty string"}
"""

REWRITE_PROMPT = """You are a text editor. A sentence contains a factual error. Rewrite the ENTIRE sentence to be correct.
Use only information from the provided evidence. Return the corrected sentence only — no explanation.
"""

def _supportive_query(fact: str) -> str:
    """RARR uses the fact itself as the query (no adversarial framing)."""
    return fact

def _judge_fact(fact: str, evidence_block: str) -> dict:
    raw = nvidia_chat(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",   "content": f"CLAIM: {fact}\n\nEVIDENCE:\n{evidence_block}\n\nReturn JSON."},
        ],
        role="judge",
        temperature=0.0,
        max_tokens=256,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reasoning": "Parse error.", "correction": ""}

def _rewrite_sentence(original: str, evidence_block: str) -> str:
    """Vanilla RARR: rewrite entire sentence (destructive)."""
    raw = nvidia_chat(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user",   "content": f"ORIGINAL: {original}\n\nEVIDENCE:\n{evidence_block[:1000]}\n\nRewritten sentence:"},
        ],
        role="editor",
        temperature=0.0,
        max_tokens=256,
    )
    return raw.strip()

def run_vanilla_rarr(
    summary: str,
    verbose: bool = True,
    *,
    facts_override: list[str] | None = None,
    sleep_seconds: float = 0.3,
) -> dict:
    """
    Run Vanilla RARR pipeline (baseline).
    Atomic decomposition + blind judge + full-sentence rewrite.
    """
    facts   = [fact.strip() for fact in (facts_override or atomicize(summary)) if str(fact).strip()][:MAX_FACTS]
    results = []
    rewrites = []

    if verbose:
        print(f"\n[Baseline: Vanilla RARR] {len(facts)} facts extracted.")

    for index, fact in enumerate(facts):
        query    = _supportive_query(fact)
        evidence = retrieve_evidence([query])
        ev_block = format_evidence_block(evidence)
        verdict  = _judge_fact(fact, ev_block)

        result = {
            "claim_index": index,
            "fact":      fact,
            "verdict":   verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
            "reasoning": verdict.get("reasoning", ""),
            "correction": verdict.get("correction", ""),
        }

        if result["verdict"] == "CONTRADICTED":
            source = find_best_matching_sentence(summary, fact)
            rewritten = _rewrite_sentence(source, ev_block)
            result["original_sentence"] = source
            result["rewritten_sentence"] = rewritten
            rewrites.append(result)
            if verbose:
                print(f"  CONTRADICTED | {fact[:60]}")
                print(f"    Rewrite: {rewritten[:80]}")
        else:
            if verbose:
                print(f"  {result['verdict']:28s} | {fact[:60]}")

        results.append(result)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    corrected = summary
    for r in rewrites:
        corrected = corrected.replace(
            r["original_sentence"].rstrip("."),
            r["rewritten_sentence"].rstrip("."),
            1
        )

    return {
        "summary":   summary,
        "corrected": corrected,
        "results":   results,
        "rewrites":  rewrites,
        "n_contradicted": len(rewrites),
        "n_total":        len(facts),
    }


if __name__ == "__main__":
    demo = (
        "BERT, introduced by Google in 2018, achieved 80.5% F1 on SQuAD 2.0. "
        "GPT-4 was released by OpenAI in March 2022."
    )
    out = run_vanilla_rarr(demo)
    print(f"\nOriginal:  {out['summary']}")
    print(f"Corrected: {out['corrected']}")
