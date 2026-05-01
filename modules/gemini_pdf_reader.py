"""
modules/gemini_pdf_reader.py — Full Paper PDF Reading via NVIDIA NIM
----------------------------------------------------------------------
Uses NVIDIA NIM to answer specific factual questions from full papers
including results tables. Previously used Gemini Flash.

This solves the core limitation of abstract-only retrieval:
  - Abstract: "We use a rank decomposition approach..."  (no number)
  - Full paper table: "r=4 achieves 90.7% on MNLI"      (exact number)

Uses pdf_extractor.py to get the paper text, then NVIDIA NIM to answer.
"""

import os
import requests

from modules.nvidia_client import nvidia_chat

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
        print(f"  [PDF Reader] Download failed for {arxiv_id}: {e}")
    return None


def query_paper_pdf(arxiv_id: str, claim: str) -> dict | None:
    """
    Extract paper text from PDF and ask NVIDIA NIM to verify a claim.

    Args:
        arxiv_id: arXiv paper ID (e.g. "2106.09685" for LoRA)
        claim:    The atomic fact to verify (e.g. "LoRA uses rank 8")

    Returns:
        Evidence dict with source, url, title, snippet — or None if failed.
    """
    try:
        from modules.pdf_extractor import extract_results_section
    except ImportError:
        print("  [PDF Reader] pdf_extractor not available")
        return None

    paper_text = extract_results_section(arxiv_id)
    if not paper_text:
        return None

    question = (
        f"You are a fact-checker. Read this paper excerpt carefully.\n\n"
        f"PAPER TEXT:\n{paper_text[:4000]}\n\n"
        f"I need to verify this specific claim: \"{claim}\"\n\n"
        f"Please:\n"
        f"1. Find the actual value(s) in the paper for the metric/fact being claimed\n"
        f"2. State whether the claim is correct or incorrect\n"
        f"3. Quote the exact sentence or table entry that confirms the real value\n\n"
        f"Be specific — include exact numbers, percentages, or names from the paper."
    )

    try:
        answer = nvidia_chat(
            [{"role": "user", "content": question}],
            role="judge",
            temperature=0.0,
            max_tokens=512,
        )

        return {
            "source":  "nvidia_pdf",
            "url":     f"https://arxiv.org/abs/{arxiv_id}",
            "title":   f"Full paper (NIM analysis) arXiv:{arxiv_id}",
            "snippet": (
                f"[NVIDIA NIM — FULL PAPER ANALYSIS]\n"
                f"Claim verified: {claim}\n\n"
                f"{answer}"
            ),
        }

    except Exception as e:
        print(f"  [PDF Reader] Query failed: {e}")
        return None


def get_gemini_evidence_for_fact(
    fact: str,
    context: str,
    known_papers: dict,
) -> list[dict]:
    """
    Try to get PDF evidence for a fact via NVIDIA NIM.
    Identifies the paper from fact+context, downloads PDF,
    asks NIM to verify the specific claim.
    """
    search_text = (fact + " " + context).lower()
    arxiv_id    = None

    for keyword, aid in known_papers.items():
        if keyword in search_text:
            arxiv_id = aid
            break

    if not arxiv_id:
        return []

    print(f"  [PDF Reader] Querying full paper arXiv:{arxiv_id}...")
    result = query_paper_pdf(arxiv_id, fact)
    return [result] if result else []


if __name__ == "__main__":
    print("Testing PDF reader on LoRA (2106.09685)...")
    result = query_paper_pdf(
        "2106.09685",
        "The paper demonstrated results on GPT-3 with a rank of 8, achieving 91.3% on MNLI"
    )
    if result:
        print(f"\n{result['snippet'][:500]}")
    else:
        print("Failed")
