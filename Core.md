# FactGuard Core Guide

This file is a practical walkthrough of how this project works, so you can quickly understand, run, and modify it.

## 1) What this project does

FactGuard checks AI-generated summaries of AI/ML papers and fixes factual mistakes.

High-level idea:
1. Break summary into small factual claims.
2. Try to find evidence that disproves each claim.
3. Judge each claim against evidence.
4. Verify the judge itself (to avoid false corrections).
5. Apply minimal, token-level edits only where errors are confirmed.

Main goal: catch subtle hallucinations (wrong number, year, author, etc.) with low false positives.

## 2) Main entry points

- `pipeline.py`: CLI pipeline orchestrator (core 6-step flow).
- `streamlit_app.py`: interactive UI with live logs and results.
- `config.py`: models, API keys, and constants.

Note:
- CLI path (`pipeline.py`) uses `modules/second_opinion.py` for fallback checks.
- Streamlit path (`streamlit_app.py`) uses `modules/deep_verifier.py` for fallback checks.
- Both serve the same purpose (handle `INSUFFICIENT_EVIDENCE`) but are different implementations.

## 3) End-to-end architecture (6 steps)

### Step 1: Atomicizer
- File: `modules/atomicizer.py`
- Function: `atomicize(text) -> list[str]`
- Uses Groq (`FAST_MODEL`) to split summary into self-contained, verifiable claims.

### Step 2: Skeptical Query Generator
- File: `modules/query_generator.py`
- Function: `generate_skeptical_queries(fact) -> list[str]`
- Generates adversarial queries designed to falsify claims, not confirm them.

### Step 3: Hybrid Retriever
- File: `modules/retriever.py`
- Function: `retrieve_evidence(queries, fact, context) -> list[dict]`
- Runs parallel retrieval from:
1. Direct arXiv lookup (authoritative source, often best signal).
2. arXiv query search.
3. DuckDuckGo web search.
4. Optional PDF extraction for numeric claims (`modules/pdf_extractor.py`).

### Step 4: LLM Judge
- File: `modules/judge.py`
- Function: `judge_claim(fact, evidence_block) -> dict`
- Verdicts:
1. `SUPPORTED`
2. `CONTRADICTED`
3. `INSUFFICIENT_EVIDENCE`

### Step 4b: Deep fallback when evidence is weak
- CLI: `modules/second_opinion.py` (`get_second_opinion`)
- Streamlit: `modules/deep_verifier.py` (`deep_verify`)
- Uses broader retrieval and optionally Gemini for PDF-level verification.

### Step 5: CoVe loop (meta-verification of the judge)
- File: `modules/cove_loop.py`
- Function: `run_cove_verification(fact, judge_result, evidence_block) -> dict`
- Only runs when Step 4 says `CONTRADICTED`.
- If judge has weak/fabricated quote, CoVe overturns contradiction to `INSUFFICIENT_EVIDENCE`.

### Step 6: Surgical editor
- File: `modules/editor.py`
- Functions:
1. `edit_sentence(...)`
2. `apply_corrections_to_summary(...)`
- Applies minimal correction span instead of rewriting full text.

## 4) Data flow and key data structures

`run_pipeline(summary)` returns:
- `original`: original input summary
- `corrected`: corrected summary
- `facts`: atomic claims
- `results`: per-fact verdict objects
- `corrections`: applied edits

Typical `results[i]` fields:
- `fact`
- `verdict`
- `reasoning`
- `evidence_quote`
- `evidence_source`
- `cove_applied`
- `cove_meta_verdict`
- Optional: `disputed`, `gemini_used`, `pdf_used`

Typical `corrections[i]` fields:
- `fact`
- `source_sentence`
- `corrected_text`
- `error_span`
- `correction`
- `source_url`
- `changed`

## 5) Evaluation logic

- File: `evaluation/skeptic_score.py`
- Defines:
1. `ClaimResult`
2. `BenchmarkReport`

Metrics:
1. Precision / Recall / F1 (hallucination detection)
2. Skeptic Score = CoVe-confirmed contradictions / total claims
3. CoVe precision gain (how often CoVe overturns weak contradictions)

`pipeline.py --bench` runs benchmark against `data/skepticbench_sample.json`.

## 6) Configuration you should know first

In `config.py`:
- `GROQ_API_KEY` (required)
- `FAST_MODEL` (used by atomicizer/query generator)
- `STRONG_MODEL` (used by judge/editor/CoVe)
- `MAX_FACTS` (claims per run)
- `MAX_ARXIV_RESULTS`, `MAX_WEB_RESULTS`
- `MIN_EVIDENCE_CHARS` (quote threshold for CoVe gate)

Environment variables:
- `GROQ_API_KEY` required
- `GEMINI_API_KEY` optional (enables deeper verification)

## 7) How to run

CLI:
```bash
python pipeline.py --text "Your summary here"
python pipeline.py --bench
```

Streamlit:
```bash
streamlit run streamlit_app.py
```

## 8) Folder map (mental model)

- `modules/`: core pipeline components.
- `evaluation/`: metrics and benchmark scripts.
- `baselines/`: comparison baselines (zero-shot, standard rag, rarr).
- `build_dataset/`: synthetic dataset generation and merging tools.
- `data/`: sample benchmark data.

## 9) First things to modify (recommended)

If you want to improve accuracy:
1. Tune `SYSTEM_PROMPT` in `modules/judge.py`.
2. Expand `KNOWN_PAPERS` in `modules/retriever.py`.
3. Improve retrieval heuristics in `retrieve_evidence(...)`.
4. Align CLI and Streamlit to one fallback module (`second_opinion` or `deep_verifier`) for consistent behavior.

If you want faster runs:
1. Reduce `MAX_FACTS`.
2. Reduce retrieval result counts.
3. Use faster model in judge/editor only if quality remains acceptable.

## 10) Known project quirks

1. `pyproject.toml` says `requires-python >=3.12`, while README says `3.10+`.
2. `main.py` is just a placeholder and not part of the real pipeline.
3. Some modules are optional-path imports (for graceful fallback), so behavior can change based on installed packages.

---

If you are onboarding, start by reading in this order:
1. `pipeline.py`
2. `modules/retriever.py`
3. `modules/judge.py`
4. `modules/cove_loop.py`
5. `modules/editor.py`
