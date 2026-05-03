"""
modules/atomicizer.py — Step 1: Atomic Decomposition
-----------------------------------------------------
Takes a paragraph/summary and breaks it into independent, verifiable
atomic facts. Inspired by FActScore (Min et al., 2023).

Each atomic fact should be:
  - Self-contained (understandable without surrounding context)
  - A single verifiable claim (not compound)
  - Grounded in the original text (no invented content)
"""

import json
import re
from groq import Groq
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, FAST_MODEL

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an expert at decomposing text into atomic facts for fact-checking.

Given a passage, extract every distinct, verifiable claim as a separate atomic fact.

Rules:
1. Each fact must be self-contained — include the subject explicitly (no pronouns).
2. One claim per fact. Never combine two unrelated claims with "and".
3. Only extract what is stated — do not infer or add information.
4. Focus on: names, numbers, dates, model names, percentages, affiliations, methods, results.
5. CRITICAL: Never separate a metric/score from its benchmark. "achieved 86.4% on MMLU" must stay as ONE fact. Never split into "achieved 86.4%" and "evaluated on MMLU" — the number and benchmark together are the verifiable claim.
6. Return a JSON array of strings. Nothing else. No explanation, no markdown.

Good example:
Input:  "GPT-4 achieved 86.4% on the MMLU benchmark."
Output: ["GPT-4 achieved 86.4% on the MMLU benchmark."]

Bad example (never do this):
["GPT-4 achieved 86.4%.", "GPT-4 was evaluated on MMLU."]  <- WRONG: splits metric from benchmark
"""

def atomicize(text: str) -> list[str]:
    """
    Decompose a text passage into atomic facts.

    Args:
        text: The summary or paragraph to decompose.

    Returns:
        A list of atomic fact strings.
    """
    response = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Decompose this into atomic facts:\n\n{text}"}
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code blocks if model wrapped the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        facts = json.loads(raw)
        if isinstance(facts, list):
            return [str(f).strip() for f in facts if str(f).strip()]
    except json.JSONDecodeError:
        # Fallback: extract quoted strings line by line
        facts = re.findall(r'"([^"]+)"', raw)
        if facts:
            return facts

    # Last resort: split on newlines and strip bullets
    lines = [re.sub(r"^[\d\-\*\.\)]+\s*", "", l).strip() for l in raw.splitlines()]
    return [l for l in lines if len(l) > 10]


if __name__ == "__main__":
    sample = (
        "BERT, introduced by Google in 2018, uses a transformer encoder architecture "
        "and was pre-trained on BooksCorpus and English Wikipedia. "
        "It achieved a score of 80.5% on the SQuAD 2.0 benchmark, "
        "outperforming all previous models at the time of publication."
    )
    print("=== ATOMICIZER TEST ===")
    print(f"Input:\n{sample}\n")
    facts = atomicize(sample)
    print(f"Extracted {len(facts)} atomic facts:")
    for i, f in enumerate(facts, 1):
        print(f"  {i}. {f}")