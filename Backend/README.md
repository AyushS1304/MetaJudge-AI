# MetaJudge AI

> Meta-Verification of LLM Judges through Adversarial Falsification for Hallucination Correction

MetaJudge investigates whether adversarial falsification outperforms confirmatory retrieval for factual error detection in scientific summarization, combining adversarial queries, hybrid retrieval, CoVe verification, cross-claim consistency graphs, and surgical correction.

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange)](https://groq.com)
[![Gemini](https://img.shields.io/badge/2nd%20Judge-Gemini%202.0%20Flash-green)](https://aistudio.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Research Question

Can an adversarial retrieval strategy that tries to disprove a claim detect factual errors in scientific summaries better than standard confirmatory retrieval?

MetaJudge answers that question with a 6-stage correction pipeline plus a cross-claim consistency layer:

1. Atomicizer
2. Cross-Claim Consistency Check
3. Adversarial Query Generation
4. Hybrid Retrieval
5. LLM Judge plus CoVe verification
6. Surgical Correction

The pipeline is aimed at AI/ML paper summaries where subtle hallucinations often appear as wrong benchmark scores, dates, authors, or architecture details.

---

## Novelty Over Baselines

| Feature | MetaJudge | RARR | FActScore | SAFE |
|---|---|---|---|---|
| Adversarial queries | Yes | No | No | No |
| Cross-claim consistency | Yes | No | No | No |
| CoVe meta-verification | Yes | No | No | No |
| Surgical correction | Yes | Yes | No | No |
| PDF fallback reading | Yes | No | No | No |

---

## Architecture

### Step 1: Atomicizer
Break the summary into self-contained, verifiable claims.

### Step 1.5: Cross-Claim Consistency Check
Detect internal contradictions between atomic claims before retrieval. Claims flagged here are marked `INTERNAL_CONTRADICTION` and skip retrieval entirely.

### Step 2: Adversarial Query Generator
Generate skeptical queries designed to falsify a claim rather than confirm it.

### Step 3: Hybrid Retriever
Retrieve evidence from:

- direct arXiv paper lookup
- adversarial arXiv search
- web search
- Gemini/PDF fallback when deeper evidence is needed

### Step 4: LLM Judge
Compare each claim with retrieved evidence and produce `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT_EVIDENCE`.

### Step 5: CoVe
Require the judge to ground contradiction decisions in a real evidence quote before allowing a correction.

### Step 6: Surgical Editor
Apply minimal token-level edits instead of rewriting the entire sentence.

---

## Evaluation

The evaluation stack reports:

- Precision
- Recall
- F1
- Correction Accuracy
- False Positive Rate
- Precision-recall curves from per-claim detection confidence

`evaluation/compare_all.py` generates:

- Table 1: system comparison
- Table 2: ablations for adversarial queries, CoVe, Gemini PDF fallback, and the consistency checker

---

## Project Structure

```text
MetaJudge-AI/
|-- pipeline.py
|-- modules/
|   |-- atomicizer.py
|   |-- consistency_checker.py
|   |-- confidence_scorer.py
|   |-- query_generator.py
|   |-- retriever.py
|   |-- judge.py
|   |-- deep_verifier.py
|   |-- cove_loop.py
|   |-- editor.py
|   `-- gemini_pdf_reader.py
|-- baselines/
|-- evaluation/
|-- data/
`-- build_dataset/
```

---

## Setup

### Requirements

- Python 3.12+
- Groq API key
- Gemini API key optional but recommended

### Install

```bash
pip install -r requirements.txt
```

### Environment

Create `.env`:

```bash
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

---

## Usage

### Run the pipeline

```bash
python pipeline.py --text "BERT achieved 80.5% F1 on SQuAD 2.0."
```

### Run evaluation

```bash
python evaluation/compare_all.py
```

### Run the API

```bash
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000
```

---

## Limitations

- Venue errors are often absent from abstracts and can remain hard to verify.
- Some architecture or hyperparameter claims require PDF-level inspection rather than abstract-only evidence.
- The system is tuned for AI/ML paper summaries, not arbitrary web claims.
- Free-tier Groq and Gemini rate limits constrain benchmark scale.

---

<details>
<summary>Examiner FAQ</summary>

**Q: What is novel?**  
Adversarial query generation plus the cross-claim consistency graph. Neither appears in RARR, FActScore, or SAFE.

**Q: Why does Precision = 1.0 not look cherry-picked?**  
MetaJudge uses CoVe meta-verification and requires a verbatim evidence quote before confirming a contradiction. That intentionally trades recall for very low false positives, which is appropriate for a correction system where false edits are worse than missed detections.

**Q: How does it compare to FActScore?**  
FActScore evaluates factual precision of generated text. MetaJudge performs factual detection and surgical correction. They overlap in atomic factual checking, but the task scope is different.

**Q: Is the dataset too small?**  
SkepticBench is smaller than broad web-claim benchmarks, but it is tightly labeled around explicit corruptions with identifiable sources. That makes correction accuracy measurable.

**Q: What are the main limitations?**  
Venue detection is weak from abstracts alone, the system is domain-specific to AI/ML papers, and provider rate limits constrain scale.

</details>

---

## Citation

```bibtex
@misc{shah2026metajudgeai,
  title    = {MetaJudge AI: Adversarial Falsification and Cross-Claim Consistency for Hallucination Detection in Scientific Summaries},
  author   = {Shah, Aniket},
  year     = {2026},
  note     = {B.Tech Final Year Project, Bharati Vidyapeeth's College of Engineering},
  keywords = {hallucination detection, adversarial retrieval, chain-of-verification, scientific fact-checking, RAG},
  url      = {https://github.com/AyushS1304/MetaJudge-AI}
}
```
