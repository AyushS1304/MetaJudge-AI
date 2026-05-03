"""
Simple CLI for inter-annotator agreement on SkepticBench labels.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


BENCH_PATH = Path("data/skepticbench_full.json")
OUTPUT_PATH = Path("results/inter_annotator_kappa.json")
SAMPLE_SIZE = 50
SEED = 42


def _flatten_claims(dataset: list[dict]) -> list[dict]:
    flattened = []
    for entry in dataset:
        corrupted_facts = entry.get("corrupted_facts", entry.get("atomic_facts", []))
        labels = entry.get("labels", [])
        for index, claim in enumerate(corrupted_facts):
            flattened.append(
                {
                    "entry_id": entry.get("id"),
                    "paper": entry.get("paper"),
                    "claim_index": index,
                    "claim": claim,
                    "label": "corrupted" if index < len(labels) and str(labels[index]).lower() == "false" else "clean",
                }
            )
    return flattened


def _cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have the same length.")
    if not labels_a:
        return 0.0

    total = len(labels_a)
    observed = sum(1 for left, right in zip(labels_a, labels_b) if left == right) / total
    p_a_yes = sum(labels_a) / total
    p_b_yes = sum(labels_b) / total
    p_a_no = 1.0 - p_a_yes
    p_b_no = 1.0 - p_b_yes
    expected = (p_a_yes * p_b_yes) + (p_a_no * p_b_no)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def run_cli() -> dict:
    if not BENCH_PATH.exists():
        raise SystemExit(f"Missing dataset: {BENCH_PATH}")

    dataset = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    claims = _flatten_claims(dataset)
    random.seed(SEED)
    sample = claims if len(claims) <= SAMPLE_SIZE else random.sample(claims, SAMPLE_SIZE)

    original_labels: list[int] = []
    annotator_labels: list[int] = []

    print(f"Annotating {len(sample)} claims from {BENCH_PATH}")
    print("Enter 'y' for CORRUPTED and 'n' for CLEAN.\n")

    for index, item in enumerate(sample, start=1):
        print(f"[{index:02d}/{len(sample):02d}] {item['paper']}")
        print(item["claim"])
        while True:
            response = input("Corrupted? [y/n]: ").strip().lower()
            if response in {"y", "n"}:
                break
            print("Please enter 'y' or 'n'.")
        original_labels.append(1 if item["label"] == "corrupted" else 0)
        annotator_labels.append(1 if response == "y" else 0)
        print()

    result = {
        "dataset": str(BENCH_PATH),
        "sample_size": len(sample),
        "cohens_kappa": round(_cohens_kappa(original_labels, annotator_labels), 4),
        "seed": SEED,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_cli()
