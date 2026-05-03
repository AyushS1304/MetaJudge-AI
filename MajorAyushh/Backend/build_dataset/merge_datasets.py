"""
build_dataset/merge_datasets.py
=================================
Merges multiple SkepticBench JSON files into one.

Run from project root:
    python build_dataset/merge_datasets.py \
        --inputs data/skepticbench_25.json data/skepticbench_new75.json \
        --out data/skepticbench_full.json
"""

import json, argparse, os, sys


def merge(input_paths, out_path):
    all_entries = []
    seen_titles = set()
    skipped = 0

    for path in input_paths:
        abs_path = os.path.abspath(path)
        print(f"\nLoading: {abs_path}")

        if not os.path.exists(abs_path):
            print(f"  ERROR — file not found: {abs_path}")
            print(f"  Run  ls data/  to see what files exist")
            sys.exit(1)

        with open(abs_path, encoding="utf-8") as f:
            entries = json.load(f)

        print(f"  Found {len(entries)} entries")

        for entry in entries:
            title = entry.get("paper", "").strip().lower()
            if title in seen_titles:
                skipped += 1
                continue
            seen_titles.add(title)
            all_entries.append(entry)

    # Re-ID sequentially
    for i, entry in enumerate(all_entries):
        entry["id"] = f"sb{i+1:03d}"

    # Save — create data/ folder if needed
    out_abs = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    total_facts = sum(len(e.get("atomic_facts", []))    for e in all_entries)
    total_errs  = sum(len(e.get("injected_errors", [])) for e in all_entries)
    clean       = sum(1 for e in all_entries if e.get("clean"))

    print(f"\n{'='*55}")
    print(f"  MERGE COMPLETE")
    print(f"{'='*55}")
    print(f"  Total entries      : {len(all_entries)}")
    print(f"  Duplicates removed : {skipped}")
    print(f"  With errors        : {len(all_entries) - clean}")
    print(f"  Clean entries      : {clean}")
    print(f"  Atomic facts       : {total_facts}")
    print(f"  Corruptions        : {total_errs}")
    print(f"  Saved to           : {out_abs}")
    print(f"{'='*55}")
    print(f"\nNext step:")
    print(f"  python build_dataset/verify_dataset.py --json {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", default="data/skepticbench_full.json")
    args = p.parse_args()

    print(f"Merging {len(args.inputs)} files into {args.out}")
    print(f"Input files:")
    for f in args.inputs:
        exists = os.path.exists(f)
        size   = os.path.getsize(f) if exists else 0
        print(f"  {'OK' if exists else 'MISSING':7s} {f}  ({size} bytes)")

    missing = [f for f in args.inputs if not os.path.exists(f)]
    if missing:
        print(f"\nERROR: these files are missing:")
        for f in missing:
            print(f"  {f}")
        print(f"\nRun  ls data/  to see what you have")
        sys.exit(1)

    merge(args.inputs, args.out)