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

This baseline isolates the contribution of:
  (a) Adversarial retrieval, and
  (b) The CoVe meta-verification loop
"""

import re
import json
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from modules.atomicizer  import atomicize
from modules.retriever   import retrieve_evidence, format_evidence_block
from config import GROQ_API_KEY, FAST_MODEL, STRONG_MODEL, MAX_FACTS

client = Groq(api_key=GROQ_API_KEY)

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
    raw = client.chat.completions.create(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",   "content": f"CLAIM: {fact}\n\nEVIDENCE:\n{evidence_block}\n\nReturn JSON."}
        ],
        temperature=0.0,
        max_tokens=256,
    ).choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reasoning": "Parse error.", "correction": ""}

def _rewrite_sentence(original: str, evidence_block: str) -> str:
    """Vanilla RARR: rewrite entire sentence (destructive)."""
    response = client.chat.completions.create(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user",   "content": f"ORIGINAL: {original}\n\nEVIDENCE:\n{evidence_block[:1000]}\n\nRewritten sentence:"}
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()

def run_vanilla_rarr(summary: str, verbose: bool = True) -> dict:
    """
    Run Vanilla RARR pipeline (baseline).
    Atomic decomposition + blind judge + full-sentence rewrite.
    """
    facts   = atomicize(summary)[:MAX_FACTS]
    results = []
    rewrites = []

    if verbose:
        print(f"\n[Baseline: Vanilla RARR] {len(facts)} facts extracted.")

    for fact in facts:
        query    = _supportive_query(fact)
        evidence = retrieve_evidence([query])
        ev_block = format_evidence_block(evidence)
        verdict  = _judge_fact(fact, ev_block)

        result = {
            "fact":      fact,
            "verdict":   verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
            "reasoning": verdict.get("reasoning", ""),
        }

        if result["verdict"] == "CONTRADICTED":
            # Vanilla RARR: rewrite full sentence (destructive)
            source = _find_sentence(summary, fact)
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
        time.sleep(0.3)

    # Apply rewrites (full-sentence replacement)
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

def _find_sentence(text: str, fact: str) -> str:
    sentences = [s.strip() for s in text.replace(".\n", ". ").split(". ") if s.strip()]
    fact_words = set(fact.lower().split())
    best, best_score = text, 0
    for s in sentences:
        score = len(set(s.lower().split()) & fact_words)
        if score > best_score:
            best_score, best = score, s
    return best + ("." if not best.endswith(".") else "")


if __name__ == "__main__":
    demo = (
        "BERT, introduced by Google in 2018, achieved 80.5% F1 on SQuAD 2.0. "
        "GPT-4 was released by OpenAI in March 2022."
    )
    out = run_vanilla_rarr(demo)
    print(f"\nOriginal:  {out['summary']}")
    print(f"Corrected: {out['corrected']}")