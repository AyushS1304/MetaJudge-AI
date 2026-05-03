"""
modules/cove_loop.py — Step 5: CoVe Verification Loop ★ THE NOVELTY
----------------------------------------------------------------------
This is the core contribution of the paper: Meta-Verification of the Judge.

Problem: The LLM Judge (Step 4) can hallucinate its own reasoning.
It might declare a claim "CONTRADICTED" without actual contradicting evidence.

Solution: Chain-of-Verification (CoVe) applied to the JUDGE, not the text.
When the judge says CONTRADICTED, a second agent asks:
  "Show me the EXACT sentence from the evidence that proves this."

If the judge cannot produce a real verbatim quote from the evidence,
the CONTRADICTED verdict is REVERSED to INSUFFICIENT_EVIDENCE.

This transforms the judge from an opaque black-box into a
verifiable, accountable fact-checker.
"""

import re
import json
from groq import Groq
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, STRONG_MODEL, MIN_EVIDENCE_CHARS
from modules.judge import VERDICT_CONTRADICTED, VERDICT_INSUFFICIENT

client = Groq(api_key=GROQ_API_KEY)

COVE_SYSTEM_PROMPT = """You are a verification auditor checking a fact-checker's decision.

A fact-checker says a claim is FALSE and provides a quote from evidence.
Your job: confirm the quote is real AND that it contradicts the claim.

IMPORTANT — CONTRADICTION LOGIC:
A contradiction does NOT require the claimed value to appear in the evidence.
If the claim says "X achieved 80.5%" and the evidence says "X achieves 86.7%" for the
same benchmark — this IS a contradiction. Different values for the same thing = contradiction.

Also accept: if the evidence is from an AUTHORITATIVE source (arXiv paper) and gives
a different specific number for the same metric/benchmark — that is sufficient.

Verdict rules:
- CONFIRMED_CONTRADICTION: Quote is present in evidence AND gives a different value for the same thing.
  Also confirm if: the quote comes from ARXIV-DIRECT (the original paper) and gives a specific value
  that directly contradicts the claimed value — even if the quote refers to a closely related
  component (e.g. encoder vs decoder in the same architecture, or a different but related benchmark).
  The original paper is authoritative — if it says "six" and the claim says "8", that is a contradiction.
- OVERTURNED: Quote is completely fabricated OR refers to a totally different model/paper/domain.
  Do NOT overturn just because the quote mentions a related component instead of the exact one claimed.

Respond ONLY with JSON:
{
  "meta_verdict": "CONFIRMED_CONTRADICTION" | "OVERTURNED",
  "reason": "One sentence explaining why."
}
"""

def _verify_judge_decision(
    fact: str,
    judge_result: dict,
    evidence_block: str,
) -> dict:
    """
    Meta-verify a CONTRADICTED judgment from the judge.
    Returns the final verdict after verification.
    """
    user_msg = (
        f"ORIGINAL CLAIM: {fact}\n\n"
        f"JUDGE'S VERDICT: {judge_result['verdict']}\n"
        f"JUDGE'S REASONING: {judge_result.get('reasoning', '')}\n"
        f"JUDGE'S QUOTED EVIDENCE: \"{judge_result.get('evidence_quote', '')}\"\n\n"
        f"FULL EVIDENCE PASSAGES:\n{evidence_block}\n\n"
        "Is the judge's contradiction claim grounded in the actual evidence? Respond with JSON."
    )

    response = client.chat.completions.create(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": COVE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except Exception:
        return {"meta_verdict": "OVERTURNED", "reason": "CoVe parser failed — defaulting to safe reversal."}


def _has_minimal_quote(judge_result: dict) -> bool:
    """Fast check: did the judge even provide a quote of minimum length?"""
    quote = judge_result.get("evidence_quote", "").strip()
    return len(quote) >= MIN_EVIDENCE_CHARS


def run_cove_verification(
    fact: str,
    judge_result: dict,
    evidence_block: str,
) -> dict:
    """
    Run the CoVe verification loop on a judge's decision.

    Only activates if the verdict is CONTRADICTED.
    If the judge's evidence quote is missing or hallucinated → reverses the verdict.

    Args:
        fact:          The atomic fact being verified.
        judge_result:  The dict returned by judge.judge_claim().
        evidence_block: The raw evidence string used by the judge.

    Returns:
        Final result dict with keys:
        {verdict, reasoning, evidence_quote, evidence_source, cove_applied, cove_meta_verdict}
    """
    result = dict(judge_result)
    result["cove_applied"]      = False
    result["cove_meta_verdict"] = None

    # CoVe only activates on CONTRADICTED verdicts
    if result["verdict"] != VERDICT_CONTRADICTED:
        return result

    result["cove_applied"] = True

    # Stage 1: Fast gate — does the judge even have a quote?
    if not _has_minimal_quote(judge_result):
        result["verdict"]          = VERDICT_INSUFFICIENT
        result["cove_meta_verdict"] = "OVERTURNED"
        result["reasoning"] = (
            f"[CoVe OVERTURNED] Judge declared contradiction but provided "
            f"no evidence quote (min {MIN_EVIDENCE_CHARS} chars required). "
            f"Original reasoning: {judge_result.get('reasoning', '')}"
        )
        return result

    # Stage 2: Deep verification — is the quote real and actually contradictory?
    meta = _verify_judge_decision(fact, judge_result, evidence_block)
    result["cove_meta_verdict"] = meta.get("meta_verdict", "OVERTURNED")

    if result["cove_meta_verdict"] == "OVERTURNED":
        result["verdict"]   = VERDICT_INSUFFICIENT
        result["reasoning"] = (
            f"[CoVe OVERTURNED] {meta.get('reason', '')} "
            f"Original reasoning: {judge_result.get('reasoning', '')}"
        )
    else:
        # CONFIRMED — keep the contradiction verdict, annotate
        result["reasoning"] = (
            f"[CoVe CONFIRMED] {meta.get('reason', '')} "
            f"Original: {judge_result.get('reasoning', '')}"
        )

    return result


if __name__ == "__main__":
    # Simulate a judge hallucinating a contradiction
    fake_judge_result = {
        "verdict": "CONTRADICTED",
        "reasoning": "The evidence says BERT got 85.1%, not 80.5%.",
        "evidence_quote": "",   # <-- no quote provided
        "evidence_source": "",
    }

    fake_evidence = "BERT was evaluated on multiple benchmarks including GLUE and SQuAD."

    print("=== CoVe LOOP TEST ===")
    print(f"Simulated judge verdict: {fake_judge_result['verdict']}")
    print(f"Quote provided: '{fake_judge_result['evidence_quote']}'\n")

    final = run_cove_verification(
        fact="BERT achieved 80.5% on SQuAD 2.0.",
        judge_result=fake_judge_result,
        evidence_block=fake_evidence,
    )
    print(f"Final verdict after CoVe: {final['verdict']}")
    print(f"CoVe applied: {final['cove_applied']}")
    print(f"CoVe meta-verdict: {final['cove_meta_verdict']}")
    print(f"Reasoning: {final['reasoning']}")