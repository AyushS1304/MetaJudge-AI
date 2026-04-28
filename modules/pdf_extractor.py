"""
modules/pdf_extractor.py — Full Paper PDF Extraction
------------------------------------------------------
Downloads the PDF from arXiv and extracts the results/experiments
section text. This gives the judge access to actual numbers from
results tables, not just the abstract.

Used by retriever.py when the abstract does not contain the
specific metric being verified.
"""

import re
import os
import sys
import tempfile
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cache to avoid re-downloading the same paper
_PDF_CACHE: dict[str, str] = {}

# Keywords that indicate results/experiments sections
RESULTS_KEYWORDS = [
    "results", "experiments", "evaluation", "performance",
    "benchmark", "baseline", "accuracy", "f1", "bleu",
    "table", "figure", "compared", "outperform", "achieve"
]


def _download_pdf_text(arxiv_id: str, max_pages: int = 12) -> str | None:
    """
    Download a paper PDF from arXiv and extract its full text.
    Returns the extracted text or None if download fails.
    Caches results to avoid repeated downloads.
    """
    if arxiv_id in _PDF_CACHE:
        return _PDF_CACHE[arxiv_id]

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    try:
        import fitz  # PyMuPDF
        response = requests.get(pdf_url, timeout=30,
                                headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return None

        # Write to temp file and extract
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(response.content)
            tmp_path = f.name

        try:
            doc   = fitz.open(tmp_path)
            pages = min(len(doc), max_pages)
            text  = "\n".join(doc[i].get_text() for i in range(pages))
            doc.close()
        finally:
            os.unlink(tmp_path)

        _PDF_CACHE[arxiv_id] = text
        return text

    except Exception as e:
        print(f"  [PDF warn] {arxiv_id}: {e}")
        return None


def extract_results_section(arxiv_id: str) -> str | None:
    """
    Extract the results/experiments section from a paper.
    Returns a focused snippet containing tables and numbers.
    """
    text = _download_pdf_text(arxiv_id)
    if not text:
        return None

    # Find results section
    lines      = text.split("\n")
    result_lines = []
    in_results = False
    results_start = -1

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Detect start of results/experiments section
        if re.match(r'^(4|5|6|7|results?|experiments?|evaluation)',
                    line_lower) and any(k in line_lower for k in
                    ["result", "experiment", "evaluation", "performance"]):
            in_results = True
            results_start = i

        # Stop at related work or conclusion
        if in_results and re.match(
                r'^(related work|conclusion|discussion|limitation|future)',
                line_lower):
            break

        if in_results:
            result_lines.append(line)

    if not result_lines:
        # Fallback: find lines with numbers and percentages
        result_lines = [
            line for line in lines
            if re.search(r'\d+\.?\d*\s*%|\b\d{1,3}\.\d\b', line)
            and any(k in line.lower() for k in RESULTS_KEYWORDS)
        ][:50]

    if not result_lines:
        return None

    # Take first 100 lines of results section
    snippet = "\n".join(result_lines[:100])

    # Clean up whitespace
    snippet = re.sub(r'\n{3,}', '\n\n', snippet)
    return snippet[:3000]  # cap at 3000 chars


def get_paper_results(arxiv_id: str) -> dict | None:
    """
    Get a formatted evidence item from the full paper PDF.
    Returns in the same format as retriever evidence dicts.
    """
    snippet = extract_results_section(arxiv_id)
    if not snippet:
        return None

    return {
        "source":  "arxiv_pdf",
        "url":     f"https://arxiv.org/abs/{arxiv_id}",
        "title":   f"Full paper results section (arXiv:{arxiv_id})",
        "snippet": f"[FROM FULL PAPER — RESULTS/EXPERIMENTS SECTION]\n{snippet}",
    }


if __name__ == "__main__":
    # Test with LoRA paper
    print("Testing PDF extraction on LoRA (2106.09685)...")
    result = get_paper_results("2106.09685")
    if result:
        print(f"✓ Extracted {len(result['snippet'])} chars")
        print("\nSample (looking for rank/MNLI):")
        lines = result['snippet'].split('\n')
        for line in lines:
            if any(k in line.lower() for k in ['rank', 'mnli', '90', '91', '4 ']):
                print(f"  {line[:120]}")
    else:
        print("✗ Could not extract")