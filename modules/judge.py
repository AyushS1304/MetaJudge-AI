"""
modules/judge.py — Step 4: LLM Judge
--------------------------------------
Compares an atomic fact against the retrieved evidence and returns
a verdict: SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.

Uses the stronger Groq model for better reasoning accuracy.
"""

import re
import json
from groq import Groq
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, FAST_MODEL, STRONG_MODEL

client = Groq(api_key=GROQ_API_KEY)

# Use fast model for judge to save daily token budget
# Switch to STRONG_MODEL only if accuracy needs improvement
JUDGE_MODEL = STRONG_MODEL

VERDICT_SUPPORTED    = "SUPPORTED"
VERDICT_CONTRADICTED = "CONTRADICTED"
VERDICT_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = """You are a precise fact-checker for AI/ML research paper claims.

You will be given a CLAIM and EVIDENCE passages. Your job: find errors.

EVIDENCE STRUCTURE:
- "ARXIV-DIRECT" items are the original paper — highest authority, always check first
- Each ARXIV-DIRECT item has: Title, Authors, Published date, Year note, Abstract

STEP 1 — CHECK THE YEAR (if claim mentions a year):
Look for the line "NOTE: This paper was published in YYYY" in ARXIV-DIRECT evidence.
If claim year ≠ evidence year → CONTRADICTED. Quote that NOTE line.

STEP 2 — CHECK NUMBERS/METRICS (if claim has a number):
Find the same metric in the evidence. If numbers differ even slightly → CONTRADICTED.
91.2 ≠ 93.2. Rank 8 ≠ Rank 4. Six ≠ 8. Do NOT say SUPPORTED if numbers differ.

STEP 3 — CHECK METHOD NAMES (if claim names a technique):
Find the technique in the abstract. Compare word by word.
"causal language modelling" ≠ "masked language modelling" → CONTRADICTED.
"rank 8" ≠ "rank 4" → CONTRADICTED.

STEP 4 — CHECK AUTHORS/INSTITUTIONS:
If claim names an author or institution, verify against the Authors line in evidence.

STEP 5 - CHECK FOR ANY CONTRADICTIONS AT ALL IN THE PASSAGE AGAIN THOROUGHLY OF ANY KIND IF YOU FIND ANYTHING SIGNIFICANT -> CONTRADICTED

RULES:
- For CONTRADICTED: quote the exact evidence sentence with the conflicting value.
- For SUPPORTED: quote the exact evidence sentence that confirms the claim value.
- For INSUFFICIENT_EVIDENCE: evidence does not discuss this specific claim at all.
- Never invent quotes. Only quote text literally present in the evidence.
- Prefer CONTRADICTED over INSUFFICIENT when evidence discusses the same topic but gives a different value.

Respond ONLY with JSON:
{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
  "reasoning": "Step X check: claim says [X], evidence says [Y] — CONTRADICTED/SUPPORTED",
  "evidence_quote": "Exact verbatim text from evidence",
  "evidence_source": "URL"
}
"""

def judge_claim(fact: str, evidence_block: str) -> dict:
    """
    Judge whether an atomic fact is supported or contradicted by evidence.

    Args:
        fact:           The atomic fact to verify.
        evidence_block: Formatted string of all retrieved evidence.

    Returns:
        A dict with keys: verdict, reasoning, evidence_quote, evidence_source
    """
    user_msg = (
        f"CLAIM TO VERIFY:\n{fact}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        "Return your judgment as JSON."
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        # Normalise verdict
        result["verdict"] = result.get("verdict", VERDICT_INSUFFICIENT).upper()
        if result["verdict"] not in (VERDICT_SUPPORTED, VERDICT_CONTRADICTED, VERDICT_INSUFFICIENT):
            result["verdict"] = VERDICT_INSUFFICIENT
        return result
    except Exception:
        return {
            "verdict": VERDICT_INSUFFICIENT,
            "reasoning": "Failed to parse judge response.",
            "evidence_quote": "",
            "evidence_source": "",
        }


if __name__ == "__main__":
    from retriever import retrieve_evidence, format_evidence_block
    from query_generator import generate_skeptical_queries

    fact = "BERT achieved 80.5% on the SQuAD 2.0 benchmark."
    print(f"=== JUDGE TEST ===\nFact: {fact}\n")

    queries  = generate_skeptical_queries(fact)
    evidence = retrieve_evidence(queries)
    ev_block = format_evidence_block(evidence)

    result = judge_claim(fact, ev_block)
    print(f"Verdict:  {result['verdict']}")
    print(f"Reason:   {result['reasoning']}")
    if result.get("evidence_quote"):
        print(f"Quote:    {result['evidence_quote']}")
        print(f"Source:   {result['evidence_source']}")