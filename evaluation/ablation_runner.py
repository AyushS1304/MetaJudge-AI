import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from env_utils import load_env
load_env()

BENCHMARK_FILE = "data/skepticbench_sample.json"
RESULTS_FILE   = "results/ablation.json"

with open(BENCHMARK_FILE) as f:
    samples = json.load(f)

def run_config(name: str, pipeline_kwargs: dict) -> dict:
    print(f"\n{'─'*55}")
    print(f"  Running: {name}")
    print(f"{'─'*55}")

    # CRITICAL: reset arXiv counter so each config starts fresh
    from modules.retriever import reset_arxiv_counter
    reset_arxiv_counter()

    from evaluation.skeptic_score import compute_metrics
    from pipeline import run_pipeline

    predictions, ground_truth = [], []

    for sample in samples:
        try:
            result = run_pipeline(
                sample["summary"],
                mode="full",
                **pipeline_kwargs
            )
            any_detected = any(
                cr.get("verdict") in {"CONTRADICTED", "INTERNAL_CONTRADICTION"}
                for cr in result.get("results", [])
            )
            predictions.append(any_detected)

            any_corrupted = any(
                lbl in {"false", "corrupted"}
                for lbl in sample.get("labels", [])
            )
            ground_truth.append(any_corrupted)
        except Exception as e:
            print(f"  ERROR on sample '{sample.get('paper','?')[:40]}': {e}")

    if len(predictions) == 0:
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                   "false_positive_rate": 0.0, "sample_count": 0}
    else:
        metrics = compute_metrics(predictions, ground_truth)

    print(f"  -> F1={metrics['f1']:.3f}  "
          f"P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}")
    return metrics

configs = [
    ("A: Full MetaJudge",           {}),
    ("B: No CoVe",                  {"skip_cove": True}),
    ("C: No Adversarial Queries",   {"adversarial_queries": False}),
    ("D: No Consistency Check",     {"skip_consistency": True}),
]

all_results = {}
for name, kwargs in configs:
    all_results[name] = run_config(name, kwargs)

baseline_f1 = all_results["A: Full MetaJudge"]["f1"]

print("\n" + "="*65)
print(f"  {'ABLATION STUDY RESULTS':^61}")
print("="*65)
print(f"  {'Configuration':<32} {'F1':>6}  {'Precision':>9}  "
      f"{'Recall':>6}  {'ΔF1':>6}")
print(f"  {'-'*61}")
for name, m in all_results.items():
    delta = m["f1"] - baseline_f1
    sign  = "+" if delta >= 0 else ""
    print(f"  {name:<32} {m['f1']:>6.3f}  "
          f"{m['precision']:>9.3f}  {m['recall']:>6.3f}  "
          f"{sign}{delta:>5.3f}")
print("="*65)

os.makedirs("results", exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved -> {RESULTS_FILE}")
