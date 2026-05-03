"""
modules/query_generator.py — Step 2: Skeptical Query Generator
---------------------------------------------------------------
Generates adversarial, conflict-seeking search queries for each atomic fact.
This is the core philosophical shift from standard RAG:
  - Standard RAG asks: "Find evidence that X is true."
  - Skeptical CoVe-RAG asks: "Find evidence that X is FALSE or WRONG."

This breaks confirmation bias by inverting the retrieval objective.
"""

import re
from groq import Groq
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, FAST_MODEL

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a skeptical fact-checker. Your job is to generate search queries
that will DISPROVE or find CONTRADICTIONS to a given claim.

For each claim, generate 2 adversarial search queries:
1. One targeting the specific number/date/name that could be wrong
2. One broader query checking the general claim from an authoritative source

Rules:
- Queries must be designed to FALSIFY the claim, not confirm it
- Use phrases like "actual", "correct", "official", "vs claimed", "real"
- Be specific — include the exact value being challenged
- Return ONLY a JSON array with exactly 2 query strings. No explanation.

Example claim: "BERT achieved 80.5% on SQuAD 2.0"
Example output:
["BERT actual SQuAD 2.0 score official result", "SQuAD 2.0 leaderboard BERT correct performance"]
"""

def generate_skeptical_queries(fact: str) -> list[str]:
    """
    Generate adversarial search queries designed to falsify a given atomic fact.

    Args:
        fact: A single atomic fact string.

    Returns:
        A list of 2 skeptical search query strings.
    """
    response = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Generate skeptical queries to DISPROVE this claim:\n\n{fact}"}
        ],
        temperature=0.2,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        import json
        queries = json.loads(raw)
        if isinstance(queries, list):
            return [str(q).strip() for q in queries if str(q).strip()][:3]
    except Exception:
        pass

    # Fallback: extract quoted strings
    queries = re.findall(r'"([^"]+)"', raw)
    if queries:
        return queries[:3]

    # Last resort: use the fact itself as a query with skeptical framing
    return [f"correct value {fact}", f"fact check {fact}"]


if __name__ == "__main__":
    test_facts = [
        "BERT was introduced by Google in 2018.",
        "GPT-4 achieved 86.4% on the MMLU benchmark.",
        "The paper 'Attention is All You Need' was authored by Vaswani et al.",
    ]

    print("=== SKEPTICAL QUERY GENERATOR TEST ===\n")
    for fact in test_facts:
        queries = generate_skeptical_queries(fact)
        print(f"Fact: {fact}")
        for i, q in enumerate(queries, 1):
            print(f"  Query {i}: {q}")
        print()