# ⚡ FactGuard — Skeptical CoVe-RAG

> **Meta-Verification of LLM Judges through Adversarial Falsification for Hallucination Correction**

FactGuard is a hallucination detection and correction pipeline for AI/ML research paper summaries. It breaks any summary into atomic claims, generates adversarial queries designed to *disprove* each claim, retrieves authoritative evidence from arXiv and the web, and uses a dual-LLM verification loop to detect and surgically correct factual errors — with zero false positives.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange)](https://groq.com)
[![Gemini](https://img.shields.io/badge/2nd%20Judge-Gemini%202.0%20Flash-purple)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem

LLMs frequently hallucinate subtle factual errors when summarising research papers — wrong benchmark scores, incorrect author names, wrong years, wrong methods. These errors are hard to detect because they are plausible and well-formatted. Standard RAG systems make this worse by retrieving evidence that *confirms* whatever the LLM already believes.

**Example:** An LLM summarises BERT as achieving "80.5% F1 on SQuAD 2.0". The real score is 83.1%. The error is subtle, numerically plausible, and a standard RAG system will not catch it because it searches for supporting evidence, not contradicting evidence.

FactGuard catches it.

---

## Architecture — 6 Steps

```
Input Summary
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Atomicizer                                             │
│  Breaks summary into self-contained atomic facts               │
│  "BERT achieved 80.5% F1 on SQuAD 2.0" — one verifiable claim │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Adversarial Query Generator                           │
│  Generates queries designed to DISPROVE the claim              │
│  ↯ "BERT actual SQuAD 2.0 F1 score vs claimed 80.5%"          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Hybrid Retriever (parallel)                           │
│  ├── arXiv Direct: fetches original paper abstract by ID       │
│  ├── arXiv Adversarial: searches with adversarial query        │
│  └── DuckDuckGo Web: broad web search for corroborating data   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: LLM Judge (Groq LLaMA-3.3-70B)                       │
│  Compares claim vs evidence — SUPPORTED / CONTRADICTED /       │
│  INSUFFICIENT_EVIDENCE                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                  ┌─────────▼──────────┐
                  │ INSUFFICIENT?      │
                  │ Step 4b: Gemini    │
                  │ reads full PDF +   │
                  │ results tables via │
                  │ LangChain          │
                  └─────────┬──────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: CoVe Loop ★ (Chain-of-Verification)                  │
│  Meta-judge audits the judge's decision                        │
│  Requires verbatim evidence quote — prevents false corrections │
│  CONFIRMED_CONTRADICTION → proceed / OVERTURNED → revert      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: RARR Editor                                           │
│  Surgical token replacement — only the wrong value is changed  │
│  "80.5%" → "83.1%" with source URL attached                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| Precision | **1.000** |
| False Positives | **0** |
| CoVe Reversals | 0 (judge was accurate) |
| Claims evaluated | 26 (SkepticBench-5) |
| Dataset | 66 papers, 874 facts, 443 corruptions |

---

## What It Catches

| Error Type | Example | Detection |
|---|---|---|
| Metric errors | BERT 80.5% → real 83.1% on SQuAD 2.0 | ✅ High |
| Author errors | Lee et al. → real Min et al. for FActScore | ✅ High |
| Date errors | GPT-4 March 2022 → real March 2023 | ✅ High |
| Architecture errors | 8 layers → real 6 layers in Transformer | ✅ High |
| Venue errors | ICML → real NeurIPS | ⚠️ Low (not in abstracts) |
| Hyperparameter errors | LoRA rank=8 → real rank=4 | ⚠️ Requires Gemini PDF |

---

## Installation

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (required)
- A free [Gemini API key](https://aistudio.google.com) (optional — enables PDF reading)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Aniket200424/FactGuard.git
cd FactGuard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file
cp .env.example .env
# Edit .env and add your API keys

# 4. Run the Streamlit demo
streamlit run streamlit_app.py
```

### .env file
```
GROQ_API_KEY=gsk_your_key_here
GEMINI_API_KEY=AIza_your_key_here   # optional but recommended
```

---

## How to Use

### Option 1 — Streamlit Demo (recommended)

```bash
streamlit run streamlit_app.py
```

1. Enter your Groq API key in the sidebar and click **Confirm**
2. Optionally add your Gemini key for deeper PDF analysis
3. Paste any AI-generated summary about an AI/ML paper
4. Click **Run Pipeline**
5. Watch the live console as each step processes
6. See the corrected output, disputed facts, and full chain-of-thought

**Built-in examples to try:**
- BERT — metric error (80.5% → 83.1%)
- FActScore — author error (Lee → Min et al.)
- GPT-4 — date error (2022 → 2023)
- Transformer — architecture error (8 → 6 layers)

---

### Option 2 — Command Line

```bash
# Run on a custom text
python pipeline.py --text "BERT achieved 80.5% F1 on the SQuAD 2.0 benchmark."

# Run on the sample benchmark (5 entries)
python pipeline.py --bench

# Save results to file (for the full dataset)
python pipeline.py --bench > results.txt 2>&1
```

---

### Option 3 — Python API

```python
from pipeline import run_pipeline

summary = """
BERT, introduced by Google in 2018, achieved 80.5% F1 on the SQuAD 2.0 benchmark.
The model uses a bidirectional transformer encoder pre-trained on BookCorpus and English Wikipedia.
"""

result = run_pipeline(summary, verbose=True)

print(result["corrected"])       # corrected summary
print(result["corrections"])     # list of changes made
print(result["results"])         # per-fact verdicts with reasoning
```

---

## Project Structure

```
FactGuard/
├── pipeline.py                  ← Main orchestrator
├── streamlit_app.py             ← Web demo UI
├── config.py                    ← API keys, model names, constants
├── requirements.txt
├── .env.example
│
├── modules/
│   ├── atomicizer.py            ← Step 1: atomic decomposition
│   ├── query_generator.py       ← Step 2: adversarial queries
│   ├── retriever.py             ← Step 3: hybrid retrieval (parallel)
│   ├── judge.py                 ← Step 4: LLM judge (Groq 70B)
│   ├── deep_verifier.py         ← Step 4b: LangChain + Gemini fallback
│   ├── cove_loop.py             ← Step 5: CoVe meta-verification ★
│   ├── editor.py                ← Step 6: surgical text editor
│   └── gemini_pdf_reader.py     ← Gemini Flash PDF reader
│
├── baselines/
│   ├── baseline_zeroshot.py     ← Zero-shot LLM baseline
│   ├── baseline_standard_rag.py ← Standard RAG baseline
│   └── baseline_rarr.py         ← Vanilla RARR baseline
│
├── evaluation/
│   ├── skeptic_score.py         ← Detection F1 + Skeptic Score metric
│   ├── compare_all.py           ← Paper Table 1 generator
│   └── ablation.py              ← Ablation study
│
├── build_dataset/
│   ├── groq_utils.py            ← Rate-limit retry wrapper
│   ├── complete_skepticbench.py ← Generate from CSV
│   └── generate_new_entries.py  ← Auto-generate from arXiv
│
└── data/
    ├── skepticbench_sample.json ← 5 entries for quick testing
    └── skepticbench_full.json   ← Full 66-entry benchmark
```

---

## Building the Dataset

```bash
# Generate from your own CSV of papers
python build_dataset/complete_skepticbench.py \
    --csv data/skeptic_dataset.csv \
    --out data/skepticbench_25.json

# Auto-generate new entries from arXiv
python build_dataset/generate_new_entries.py \
    --count 75 \
    --out data/skepticbench_new75.json

# Merge datasets
python build_dataset/merge_datasets.py \
    --inputs data/skepticbench_25.json data/skepticbench_new75.json \
    --out data/skepticbench_full.json

# Verify
python build_dataset/verify_dataset.py --json data/skepticbench_full.json
```

---

## Running Evaluation

```bash
# Full system vs 3 baselines (generates paper Table 1)
python evaluation/compare_all.py

# Ablation study (Step 3.5 of paper)
python evaluation/ablation.py
```

---

## Models Used

| Component | Model | Why |
|---|---|---|
| Judge | `llama-3.3-70b-versatile` (Groq) | Strongest free-tier reasoning |
| Atomicizer | `llama-3.1-8b-instant` (Groq) | Fast, sufficient for decomposition |
| Query generator | `llama-3.1-8b-instant` (Groq) | Fast, saves token budget |
| Second opinion | `gemini-2.0-flash` (Google) | Native PDF reading, large context |
| Dataset generation | `llama-3.1-8b-instant` (Groq) | High token limit (131k/min) |

---

## Limitations

- Venue/conference errors (ICML vs NeurIPS) are rarely detectable — conference names do not appear in arXiv abstracts
- Hyperparameter errors require Gemini PDF reading — values like LoRA rank appear only in results tables, not abstracts
- The system is domain-specific to AI/ML papers — optimised for arXiv papers, not general web content
- Groq free tier: 100k tokens/day on the 70B model — the full 66-entry benchmark requires overnight running or multiple API keys

---

## Citation

```bibtex
@misc{shah2026factguard,
  title   = {FactGuard: Meta-Verification of LLM Judges through Adversarial Falsification for Hallucination Correction},
  author  = {Shah, Aniket},
  year    = {2026},
  note    = {B.Tech Final Year Project, Bharati Vidyapeeth's College of Engineering},
  url     = {https://github.com/Aniket200424/FactGuard}
}
```

---

## References

1. Dhuliawala et al. (2023) — Chain-of-Verification Reduces Hallucination in LLMs
2. Gao et al. (2022) — RARR: Researching and Revising What Language Models Say
3. Min et al. (2023) — FActScore: Fine-grained Atomic Evaluation of Factual Precision
4. Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

---

*Built as B.Tech Final Year Project — Bharati Vidyapeeth's College of Engineering, Delhi | Feb 2026*
