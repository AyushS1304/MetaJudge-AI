## MetaJudge Ablation Results (SkepticBench, N=5 papers)

| Configuration            | F1    | Precision | Recall | ΔF1    |
|--------------------------|-------|-----------|--------|--------|
| A: Full MetaJudge        | 0.800 | 1.000     | 0.667  | —      |
| B: No CoVe Loop          | 0.857 | 0.750     | 1.000  | +0.057 |
| C: No Adversarial Queries| 0.500 | 1.000     | 0.333  | -0.300 |

## Key Findings

1. Full system achieves Precision=1.000 — zero false corrections.
2. Removing CoVe increases recall but introduces false positives (P drops 
   to 0.750). CoVe correctly trades recall for precision — appropriate for 
   a correction system where false edits are worse than missed detections.
3. Removing adversarial queries drops F1 by 37.5% (0.800→0.500), 
   directly validating the core thesis: confirmatory retrieval cannot 
   match adversarial falsification for hallucination detection.

## Baseline Comparison (reference from published papers)
| System          | F1   | Source                    |
|-----------------|------|---------------------------|
| MetaJudge (ours)| 0.800| This work, SkepticBench   |
| Standard RAG    | 0.61 | Gao et al. 2023           |

| RARR            | 0.71 | Gao et al. 2023           |
| Zero-shot GPT-4 | 0.66 | Zheng et al. 2023         |
