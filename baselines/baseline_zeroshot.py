"""
baselines/baseline_zeroshot.py — Baseline 3: Zero-Shot LLM
-----------------------------------------------------------
Simulates asking a strong LLM to fact-check using only its
parametric memory — no retrieval, no tools.

This represents the "just ask the model" approach.

Missing vs Skeptical CoVe-RAG:
  ✗ No retrieval — relies entirely on training knowledge
  ✗ No adversarial framing
  ✗ No CoVe meta-verification
  ✗ Subject to all the biases the model was trained on

In the paper, this baseline shows that LLM parametric memory
alone cannot reliably catch subtle, technical hallucinations
(wrong dates, metric values, author attributions).
"""

import re
import json
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from modules.atomicizer import atomicize
from config import GROQ_API_KEY, STRONG_MODEL, MAX_FACTS

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a knowledgeable fact-checker with expertise in AI/ML research.
Using ONLY your training knowledge (no search), evaluate each claim.

For each claim, return a JSON object:
{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "UNCERTAIN",
  "reasoning": "Brief explanation",
  "correction": "Correct value if contradicted, else empty string"
}
"""

def run_zeroshot(summary: str, verbose: bool = True) -> dict:
    """
    Run Zero-Shot fact-checking (baseline).
    No retrieval — pure LLM parametric memory.
    """
    facts   = atomicize(summary)[:MAX_FACTS]
    results = []

    if verbose:
        print(f"\n[Baseline: Zero-Shot LLM] {len(facts)} facts to check.")

    for fact in facts:
        response = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Fact-check this claim using your knowledge:\n\n{fact}"}
            ],
            temperature=0.0,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$",          "", raw)

        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"verdict": "UNCERTAIN", "reasoning": "Parse error.", "correction": ""}

        # Map "UNCERTAIN" to INSUFFICIENT_EVIDENCE for unified comparison
        verdict = parsed.get("verdict", "UNCERTAIN")
        if verdict == "UNCERTAIN":
            verdict = "INSUFFICIENT_EVIDENCE"

        result = {
            "fact":       fact,
            "verdict":    verdict,
            "reasoning":  parsed.get("reasoning", ""),
            "correction": parsed.get("correction", ""),
        }
        results.append(result)

        if verbose:
            print(f"  {result['verdict']:28s} | {fact[:65]}")
        time.sleep(0.2)

    contradictions = [r for r in results if r["verdict"] == "CONTRADICTED"]

    # Apply simple string corrections (no surgical editing)
    corrected = summary
    for c in contradictions:
        if c.get("correction"):
            corrected = corrected.replace(c["fact"], c["correction"], 1)

    return {
        "summary":        summary,
        "corrected":      corrected,
        "results":        results,
        "n_contradicted": len(contradictions),
        "n_total":        len(facts),
    }


if __name__ == "__main__":
    demo = (
        "BERT, introduced by Google in 2018, achieved 80.5% F1 on SQuAD 2.0. "
        "GPT-4 was released by OpenAI in March 2022."
    )
    out = run_zeroshot(demo)
    print(f"\nOriginal:  {out['summary']}")
    print(f"Corrected: {out['corrected']}")