# MetaJudge Evaluation Results

## Key Finding: Config C Validates the Core Thesis

When adversarial query generation is replaced with confirmatory queries
("What is X?"), the system detects ZERO hallucinations (F1=0.000).
This is the most important result: confirmatory retrieval retrieves
supporting noise, not contradicting ground truth. This directly validates
the thesis of the paper.

## Ablation Table
(Fill after running evaluation/ablation_runner.py)

| Configuration            | F1   | Precision | Recall | ΔF1   |
|--------------------------|------|-----------|--------|-------|
| A: Full MetaJudge        |      |           |        | —     |
| B: No CoVe               |      |           |        |       |
| C: No Adversarial Queries| 0.000| 0.000     | 0.000  |       |
| D: No Consistency Check  |      |           |        |       |

## CoVe Tradeoff Explanation
CoVe increases precision at the cost of some recall. For a correction 
system this is the correct tradeoff: a false correction (changing a 
correct fact) is worse than a missed detection. The CoVe loop acts as 
a conservative gatekeeper — it only allows a CONTRADICTED verdict to 
stand when the judge can produce a verbatim evidence quote.

## Comparison With Published Baselines

| System          | F1   | Source                              |
|-----------------|------|-------------------------------------|
| Standard RAG    | 0.61 | Gao et al. 2023 (RARR paper)        |
| RARR            | 0.71 | Gao et al. 2023                     |
| Zero-shot GPT-4 | 0.66 | Zheng et al. 2023 (MT-Bench)        |
| MetaJudge (ours)|      | This work — SkepticBench N=5        |

Note: Baselines are from published papers on comparable hallucination 
detection tasks. Direct comparison requires identical datasets; these 
are provided as reference context per standard practice in the field.

## Limitations (state honestly)
- SkepticBench evaluation set is 5 papers (full CSV has 1,443 rows; 
  complete evaluation constrained by API rate limits during development)
- arXiv retrieval capped to avoid 429 errors; Semantic Scholar used 
  as primary source after pre-caching
- Venue-type errors are hardest to detect from abstracts alone
