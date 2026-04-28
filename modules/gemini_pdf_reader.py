"""
modules/gemini_pdf_reader.py — Gemini Flash PDF Reading
---------------------------------------------------------
Uses Gemini Flash's native PDF understanding to answer specific
factual questions from full papers including results tables.

This solves the core limitation of abstract-only retrieval:
  - Abstract: "We use a rank decomposition approach..."  (no number)
  - Full paper table: "r=4 achieves 90.7% on MNLI"      (exact number)

Gemini Flash accepts the PDF directly and answers targeted questions
about specific claims — no parsing, no section detection needed.

Requires: GEMINI_API_KEY in .env
Install:  pip install google-genai
"""

import os
import re
import sys
import requests
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Key read fresh inside each function — never cached at import time

# Cache downloaded PDFs to avoid re-downloading
_PDF_BYTES_CACHE: dict[str, bytes] = {}


def _download_pdf_bytes(arxiv_id: str) -> bytes | None:
    """Download PDF bytes from arXiv."""
    if arxiv_id in _PDF_BYTES_CACHE:
        return _PDF_BYTES_CACHE[arxiv_id]
    try:
        url      = f"https://arxiv.org/pdf/{arxiv_id}"
        response = requests.get(url, timeout=30,
                                headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            _PDF_BYTES_CACHE[arxiv_id] = response.content
            return response.content
    except Exception as e:
        print(f"  [Gemini PDF] Download failed for {arxiv_id}: {e}")
    return None


def query_paper_pdf(arxiv_id: str, claim: str) -> dict | None:
    """
    Upload a paper PDF to Gemini Flash and ask a targeted question
    about a specific claim to find the ground truth value.

    Args:
        arxiv_id: arXiv paper ID (e.g. "2106.09685" for LoRA)
        claim:    The atomic fact to verify (e.g. "LoRA uses rank 8")

    Returns:
        Evidence dict with source, url, title, snippet — or None if failed.
    """
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        print("  [Gemini PDF] No GEMINI_API_KEY set — skipping")
        return None

    pdf_bytes = _download_pdf_bytes(arxiv_id)
    if not pdf_bytes:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build a targeted question from the claim
        question = f"""You are a fact-checker. Read this paper carefully including all tables and results sections.

I need to verify this specific claim: "{claim}"

Please:
1. Find the actual value(s) in the paper for the metric/fact being claimed
2. State whether the claim is correct or incorrect
3. Quote the exact sentence or table entry that confirms the real value

Be specific — include exact numbers, percentages, or names from the paper."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
                question,
            ]
        )

        answer = response.text.strip()

        return {
            "source":  "gemini_pdf",
            "url":     f"https://arxiv.org/abs/{arxiv_id}",
            "title":   f"Full paper (Gemini PDF analysis) arXiv:{arxiv_id}",
            "snippet": (
                f"[GEMINI FLASH — FULL PAPER ANALYSIS]\n"
                f"Claim verified: {claim}\n\n"
                f"{answer}"
            ),
        }

    except Exception as e:
        print(f"  [Gemini PDF] Query failed: {e}")
        return None


def get_gemini_evidence_for_fact(
    fact: str,
    context: str,
    known_papers: dict,
) -> list[dict]:
    """
    Try to get Gemini PDF evidence for a fact.
    Identifies the paper from fact+context, downloads PDF,
    asks Gemini to verify the specific claim.

    Args:
        fact:          The atomic fact to verify
        context:       Full original summary (for paper identification)
        known_papers:  The KNOWN_PAPERS dict from retriever

    Returns:
        List of evidence dicts (0 or 1 items)
    """
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        return []

    search_text = (fact + " " + context).lower()
    arxiv_id    = None

    # Find which paper this fact is about
    for keyword, aid in known_papers.items():
        if keyword in search_text:
            arxiv_id = aid
            break

    if not arxiv_id:
        return []

    print(f"  [Gemini PDF] Querying full paper arXiv:{arxiv_id}...")
    result = query_paper_pdf(arxiv_id, fact)
    return [result] if result else []


if __name__ == "__main__":
    # Test with LoRA
    if not GEMINI_API_KEY:
        print("Set GEMINI_API_KEY in .env first")
    else:
        print("Testing Gemini PDF reader on LoRA (2106.09685)...")
        result = query_paper_pdf(
            "2106.09685",
            "The paper demonstrated results on GPT-3 with a rank of 8, achieving 91.3% on MNLI"
        )
        if result:
            print(f"\n{result['snippet'][:500]}")
        else:
            print("Failed")