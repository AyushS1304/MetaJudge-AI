"""
modules/deep_verifier.py — Deep Fact Verification
--------------------------------------------------
Called when Groq returns INSUFFICIENT_EVIDENCE.
Uses escalating strategies:

1. Broader web searches (Papers With Code, Semantic Scholar, direct queries)
2. Direct arXiv full-paper text via PDF extraction
3. NVIDIA NIM 405B escalation — reads the enriched context and verifies the specific claim

Key design: escalation tier (405B) is only used when confidence < ESCALATION_CONFIDENCE_GATE.
"""

import os
import re
import json

from config import ESCALATION_CONFIDENCE_GATE
from modules.nvidia_client import nvidia_chat, MODELS

_TEXT_CACHE: dict[str, str] = {}

ESCALATION_SYSTEM_PROMPT = (
    "You are a scientific fact-checker. Given an atomic claim and "
    "retrieved evidence, determine if the claim is factually correct. "
    "Be precise about numbers, dates, and author names."
)


# ── 1. Broader searches ────────────────────────────────────────────────────

def _broader_searches(fact: str, context: str) -> str:
    """Run 4 targeted search strategies to find metric evidence."""
    try:
        from modules.retriever import _search_web, _search_arxiv, format_evidence_block
    except Exception:
        return ""

    all_ev, seen = [], set()

    def add(evs):
        for e in evs:
            u = e.get("url", "")
            if u and u not in seen:
                seen.add(u); all_ev.append(e)

    nums  = re.findall(r'\d+\.?\d*\s*%?', fact)
    words = re.findall(r'\b[A-Z][a-zA-Z0-9\-]{2,}\b', fact + " " + context)

    # Strategy 1 — direct fact
    add(_search_web(fact[:100]))

    # Strategy 2 — model + benchmark
    if words and nums:
        add(_search_web(f"{words[0]} benchmark results {' '.join(nums[:2])}"))

    # Strategy 3 — Papers With Code leaderboard
    bench_kw = ["squad","mnli","mmlu","glue","bleu","wmt","imagenet","hellaswag","arc"]
    bench = next((b for b in bench_kw if b in fact.lower()), "")
    if words and bench:
        add(_search_web(f"{words[0]} {bench} paperswithcode leaderboard state of the art"))

    # Strategy 4 — arXiv
    add(_search_arxiv(" ".join(words[:3])))

    return format_evidence_block(all_ev[:8]) if all_ev else ""


# ── 2. Full paper text retrieval (direct arXiv, no LangChain) ──────────────

def _fetch_full_paper_text(arxiv_id: str) -> str:
    """Fetch full paper text by extracting from PDF."""
    if arxiv_id in _TEXT_CACHE:
        return _TEXT_CACHE[arxiv_id]
    try:
        from modules.pdf_extractor import extract_results_section
        text = extract_results_section(arxiv_id)
        if text and len(text) > 200:
            _TEXT_CACHE[arxiv_id] = text
            print(f"  [PDF Extract] Got {len(text)} chars for arXiv:{arxiv_id}")
            return text
    except Exception as e:
        print(f"  [PDF Extract] {e}")
    return ""


# ── 3. NVIDIA NIM escalation judge ────────────────────────────────────────

def _call_escalation(fact: str, arxiv_id: str,
                     paper_text: str, extra_ev: str) -> dict | None:
    """Call 405B escalation model with all available evidence."""
    evidence_parts = []
    if paper_text:
        evidence_parts.append(f"PAPER TEXT:\n{paper_text[:4000]}")
    if extra_ev:
        evidence_parts.append(f"WEB EVIDENCE:\n{extra_ev[:2000]}")
    evidence_text = "\n\n".join(evidence_parts) or "No additional evidence."

    user_msg = (
        f"Claim: {fact}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        'Respond in JSON: '
        '{"verdict": "SUPPORTED"|"CONTRADICTED"|"INSUFFICIENT_EVIDENCE",'
        '"confidence": float,'
        '"reasoning": str,'
        '"evidence_quote": str or null}'
    )

    messages = [
        {"role": "system", "content": ESCALATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = nvidia_chat(messages, role="escalation", temperature=0.0, max_tokens=512)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw, re.DOTALL)
            if m:
                result = json.loads(m.group())
            else:
                v = ("CONTRADICTED" if "CONTRADICTED" in raw.upper()
                     else "SUPPORTED" if "SUPPORTED" in raw.upper()
                     else "INSUFFICIENT_EVIDENCE")
                result = {
                    "verdict": v,
                    "reasoning": raw[:200],
                    "evidence_quote": "",
                    "evidence_source": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                }

        result["gemini_used"] = False
        result["pdf_used"]    = bool(paper_text)
        result["model_used"]  = MODELS["escalation"]
        print(f"  [Escalation] {result.get('verdict')} — {result.get('reasoning','')[:80]}")
        return result

    except Exception as e:
        print(f"  [Escalation] {type(e).__name__}: {e}")
        return None


# ── Main entry point ───────────────────────────────────────────────────────

def deep_verify(
    fact: str,
    context: str,
    original_evidence: str,
    known_papers: dict,
    gemini_key: str = "",
    verbose: bool = True,
    confidence_score: float = 0.5,
) -> dict | None:
    """
    Full deep verification pipeline.
    Uses NVIDIA NIM 405B as the escalation model.
    Only escalates when confidence_score < ESCALATION_CONFIDENCE_GATE.

    Returns verdict dict or None.
    """
    # Find arXiv ID
    search_text = (fact + " " + context).lower()
    arxiv_id    = None
    for keyword, aid in known_papers.items():
        if keyword in search_text:
            arxiv_id = aid
            break

    if verbose:
        print(f"  [Deep verify] Paper: arXiv:{arxiv_id or 'unknown'}")

    # Step 1: broader searches
    if verbose: print("  [Deep verify] Running broader searches...")
    extra_ev = _broader_searches(fact, context)
    if original_evidence:
        extra_ev = (extra_ev + "\n\n---\n\n" + original_evidence) if extra_ev else original_evidence

    # Step 2: full paper text (from PDF extraction)
    paper_text = ""
    if arxiv_id:
        if verbose: print(f"  [Deep verify] Extracting full text from PDF...")
        paper_text = _fetch_full_paper_text(arxiv_id)

    # Step 3: Try Groq-equivalent judge with enriched evidence first
    if paper_text or extra_ev:
        try:
            from modules.judge import judge_claim
            combined = ""
            if paper_text: combined += f"PAPER TEXT:\n{paper_text}\n\n"
            if extra_ev:   combined += f"WEB EVIDENCE:\n{extra_ev}"
            result = judge_claim(fact, combined)
            result["gemini_used"] = False
            result["pdf_used"]    = bool(paper_text)
            if verbose:
                print(f"  [Deep verify] Judge re-judge on enriched evidence: {result.get('verdict')}")
            if result.get("verdict") != "INSUFFICIENT_EVIDENCE":
                return result
        except Exception as e:
            if verbose: print(f"  [Deep verify] Re-judge failed: {e}")

    # Step 4: escalation to 405B (only if confidence below gate)
    if confidence_score < ESCALATION_CONFIDENCE_GATE:
        if verbose: print(f"  [Deep verify] Escalating to 405B (confidence={confidence_score:.2f})...")
        result = _call_escalation(fact, arxiv_id or "", paper_text, extra_ev)
        if result:
            return result
    elif verbose:
        print(f"  [Deep verify] Skipping escalation (confidence={confidence_score:.2f} >= {ESCALATION_CONFIDENCE_GATE})")

    return None


if __name__ == "__main__":
    from modules.retriever import KNOWN_PAPERS, retrieve_evidence, format_evidence_block
    from modules.query_generator import generate_skeptical_queries

    fact    = "The paper demonstrated results with a rank of 8, achieving 91.3% on MNLI."
    context = "LoRA was proposed by Hu et al. from Microsoft in 2022."

    queries = generate_skeptical_queries(fact)
    ev      = retrieve_evidence(queries, fact=fact, context=context)
    ev_blk  = format_evidence_block(ev)

    result = deep_verify(fact, context, ev_blk, KNOWN_PAPERS, verbose=True)
    if result:
        print(f"\nVerdict: {result.get('verdict')}")
        print(f"Reason:  {result.get('reasoning')}")
        print(f"Quote:   {result.get('evidence_quote','')[:120]}")
    else:
        print("No result")
