"""
build_dataset/complete_skepticbench.py
=======================================
Reads skeptic_dataset.csv, generates corruptions for false-labeled facts,
outputs skepticbench_25.json. Uses retry logic + 8B model to avoid rate limits.

Run:
    python build_dataset/complete_skepticbench.py \
        --csv data/skeptic_dataset.csv \
        --out data/skepticbench_25.json
"""

import os, sys, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from dotenv import load_dotenv
from build_dataset.groq_utils import call_groq, BULK_MODEL

load_dotenv()

CORRUPT_SYSTEM = """You are building a hallucination benchmark. Given a correct atomic claim, introduce ONE subtle error.
Change ONLY ONE thing: a number, name, date, or method. Keep structure identical.
Return ONLY the corrupted sentence. Nothing else. No quotes, no prefix."""


def corrupt_fact(fact: str, title: str) -> str:
    try:
        result = call_groq(
            messages=[
                {"role": "system", "content": CORRUPT_SYSTEM},
                {"role": "user", "content": f"Paper: {title}\n\nCorrupt:\n{fact}"},
            ],
            model=BULK_MODEL, temperature=0.7, max_tokens=200,
        )
        result = re.sub(r'^(Corrupted|Wrong|Output|Answer):\s*', '', result.strip('"\''), flags=re.I)
        return result.strip()
    except Exception as e:
        print(f"    [warn] {e}")
        return fact


def parse_list(raw: str) -> list[str]:
    return [f.strip().lstrip("- •*").strip() for f in raw.split("\n")
            if f.strip() and f.strip() not in ("nan", "#NAME?")]


def parse_labels(raw: str) -> list[bool]:
    return [l.strip().lower() == "true" for l in str(raw).strip().split("\n")
            if l.strip() in ("true", "false")]


def run(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    output, total_corrupted = [], 0

    for idx, row in df.iterrows():
        title  = str(row["title"])
        atomic = parse_list(str(row["atomic_facts"]))
        labels = parse_labels(str(row["labels"]))
        existing = parse_list(str(row["corrupted_facts"])) if "#NAME?" not in str(row["corrupted_facts"]) else []

        n = min(len(atomic), len(labels))
        atomic, labels = atomic[:n], labels[:n]
        false_count = labels.count(False)

        print(f"\n[{idx+1:02d}/{len(df)}] {title[:55]}")
        print(f"       {n} facts — {false_count} to corrupt")

        final, injected = [], []

        for j, (fact, is_correct) in enumerate(zip(atomic, labels)):
            if is_correct:
                final.append(fact)
                continue

            has_real = (j < len(existing) and existing[j].strip() != fact.strip() and len(existing[j].strip()) > 10)
            if has_real:
                corrupted = existing[j]
                print(f"  fact {j+1}: reusing ✓")
            else:
                print(f"  fact {j+1}: generating...")
                corrupted = corrupt_fact(fact, title)

            if corrupted.strip() == fact.strip():
                print(f"    identical — skipped")
                final.append(fact)
                continue

            final.append(corrupted)
            injected.append({"type": "corrupted_fact", "correct": fact, "wrong": corrupted, "fact": corrupted, "source": title})
            total_corrupted += 1

        output.append({
            "id": f"sb{idx+1:03d}", "paper": title,
            "authors": str(row["authors"]), "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "summary": " ".join(final[:6]),
            "atomic_facts": atomic, "corrupted_facts": final,
            "labels": [str(l).lower() for l in labels],
            "injected_errors": injected, "clean": len(injected) == 0,
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"  Done — {len(output)} entries, {total_corrupted} corruptions")
    print(f"  Saved → {out_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/skeptic_dataset.csv")
    p.add_argument("--out", default="data/skepticbench_25.json")
    args = p.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        print("⚠  Set GROQ_API_KEY in .env"); sys.exit(1)
    run(args.csv, args.out)