"""
modules/editor.py — Step 6: Targeted RARR Editor
--------------------------------------------------
Surgically corrects only the specific tokens proven to be wrong.

Unlike vanilla RARR which rewrites entire sentences, this editor:
1. Identifies the exact span that's incorrect (the "error token")
2. Replaces only that span with the correct value from evidence
3. Preserves all surrounding text (minimizes Levenshtein distance)

This matters for legal, medical, and academic contexts where
original phrasing must be maintained as much as possible.
"""

import re
import json
from groq import Groq
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, STRONG_MODEL

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a surgical text editor. A specific claim in a text has been proven incorrect.
Your job is to make the MINIMAL edit to fix it.

Rules:
1. Only change the specific incorrect value — do NOT rewrite the whole sentence.
2. Use ONLY the corrected value from the provided evidence. Do not invent corrections.
3. If the correction is an author name from arXiv (e.g. "Min, Sewon"), format it as "Min et al." — use "et al." style, not a full author list.
4. If the correction is a year, number, or model name — use it exactly as stated in the evidence.
5. If you cannot find the correct value in the evidence, return the original text unchanged.
6. Preserve all original phrasing, punctuation, and style except the error.

Respond ONLY with JSON:
{
  "corrected_text": "The full original sentence with only the error fixed",
  "error_span": "The exact wrong text that was replaced",
  "correction": "The replacement value used",
  "source_url": "URL of the evidence used for correction"
}
"""

def edit_sentence(
    original_sentence: str,
    wrong_fact: str,
    cove_result: dict,
    evidence_block: str,
) -> dict:
    """
    Surgically correct a sentence based on the evidence that contradicted it.

    Args:
        original_sentence: The full sentence from the original summary.
        wrong_fact:        The atomic fact that was proven wrong.
        cove_result:       The final CoVe-verified result dict.
        evidence_block:    The retrieved evidence used to prove the error.

    Returns:
        A dict with keys: corrected_text, error_span, correction, source_url, changed
    """
    quote   = cove_result.get("evidence_quote", "")
    src_url = cove_result.get("evidence_source", "")

    user_msg = (
        f"ORIGINAL SENTENCE:\n{original_sentence}\n\n"
        f"WRONG ATOMIC FACT:\n{wrong_fact}\n\n"
        f"CONTRADICTING EVIDENCE QUOTE:\n{quote}\n\n"
        f"FULL EVIDENCE (for context):\n{evidence_block[:1500]}\n\n"
        f"Make the minimal surgical correction. Return JSON."
    )

    response = client.chat.completions.create(
        model=STRONG_MODEL,
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
        result["changed"] = (
            result.get("corrected_text", "").strip() != original_sentence.strip()
        )
        if not result.get("source_url"):
            result["source_url"] = src_url
        return result
    except Exception:
        return {
            "corrected_text": original_sentence,
            "error_span":     "",
            "correction":     "",
            "source_url":     src_url,
            "changed":        False,
        }


def apply_corrections_to_summary(
    original_summary: str,
    corrections: list[dict],
) -> str:
    """
    Apply all confirmed corrections back to the original summary.

    Args:
        original_summary: The full original text.
        corrections:      List of dicts from edit_sentence(), each with
                          {wrong_fact, corrected_text, error_span, correction}.

    Returns:
        The corrected summary string.
    """
    corrected = original_summary
    for corr in corrections:
        if corr.get("changed") and corr.get("error_span") and corr.get("correction"):
            corrected = corrected.replace(
                corr["error_span"],
                corr["correction"],
                1,  # Replace only first occurrence
            )
    return corrected


if __name__ == "__main__":
    sentence = "BERT, introduced by Google in 2018, achieved 80.5% on the SQuAD 2.0 benchmark."
    wrong_fact = "BERT achieved 80.5% on the SQuAD 2.0 benchmark."
    fake_cove = {
        "verdict": "CONTRADICTED",
        "evidence_quote": "BERT achieves 86.7 F1 on SQuAD 2.0 dev set.",
        "evidence_source": "https://arxiv.org/abs/1810.04805",
    }
    evidence = "BERT achieves 86.7 F1 on SQuAD 2.0 dev set according to the original paper."

    print("=== EDITOR TEST ===")
    print(f"Original: {sentence}")
    result = edit_sentence(sentence, wrong_fact, fake_cove, evidence)
    print(f"Corrected: {result['corrected_text']}")
    print(f"Changed: {result['changed']}")
    print(f"Error span: '{result['error_span']}' → '{result['correction']}'")
    print(f"Source: {result['source_url']}")