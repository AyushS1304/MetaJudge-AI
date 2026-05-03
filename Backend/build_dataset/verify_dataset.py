"""
build_dataset/verify_dataset.py
================================
Run after complete_skepticbench.py to QA your dataset.
Checks for: valid JSON, real corruptions, label alignment,
duplicate entries, and prints a full summary report.

Run from project root:
    python build_dataset/verify_dataset.py --json data/skepticbench_full.json
"""

import json
import argparse
import sys

def verify(path: str):
    print(f"\nLoading {path}...")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  FAIL — could not load JSON: {e}")
        sys.exit(1)

    print(f"  OK — {len(data)} entries loaded\n")

    issues      = []
    total_facts = 0
    total_err   = 0
    identical   = 0
    empty_summ  = 0

    for entry in data:
        eid   = entry.get("id", "?")
        paper = entry.get("paper", "?")[:55]

        # Check required keys
        for key in ["id", "paper", "summary", "atomic_facts",
                    "corrupted_facts", "labels", "injected_errors", "clean"]:
            if key not in entry:
                issues.append(f"{eid}: missing key '{key}'")

        facts  = entry.get("atomic_facts", [])
        corrupt = entry.get("corrupted_facts", [])
        errors  = entry.get("injected_errors", [])
        labels  = entry.get("labels", [])
        summary = entry.get("summary", "")

        total_facts += len(facts)
        total_err   += len(errors)

        # Summary shouldn't be empty
        if len(summary.strip()) < 20:
            empty_summ += 1
            issues.append(f"{eid}: summary too short ({len(summary)} chars)")

        # Label count should match fact count
        if len(labels) != len(facts):
            issues.append(f"{eid}: label count ({len(labels)}) != fact count ({len(facts)})")

        # Each injected error should have correct != wrong
        for err in errors:
            c = err.get("correct", "").strip()
            w = err.get("wrong",   "").strip()
            if c == w:
                identical += 1
                issues.append(f"{eid}: corruption identical to original: '{c[:60]}'")
            if len(w) < 10:
                issues.append(f"{eid}: corruption too short: '{w}'")

        # injected_errors count should match false-label count
        false_count = labels.count("false")
        if len(errors) != false_count:
            issues.append(
                f"{eid}: {len(errors)} injected errors but {false_count} false labels"
            )

    # Check for duplicate IDs
    ids = [e.get("id") for e in data]
    if len(ids) != len(set(ids)):
        issues.append("Duplicate entry IDs found")

    # ── Print report ──────────────────────────────────────────────────────
    clean_count = sum(1 for e in data if e.get("clean"))

    print("=" * 55)
    print("  SKEPTICBENCH VERIFICATION REPORT")
    print("=" * 55)
    print(f"  Total entries          : {len(data)}")
    print(f"  Clean entries          : {clean_count}")
    print(f"  Entries with errors    : {len(data) - clean_count}")
    print(f"  Total atomic facts     : {total_facts}")
    print(f"  Total corruptions      : {total_err}")
    print(f"  Avg facts per entry    : {total_facts/len(data):.1f}")
    print(f"  Avg errors per entry   : {total_err/len(data):.1f}")
    print(f"  Corruption rate        : {total_err/total_facts*100:.1f}%")
    print("-" * 55)

    if issues:
        print(f"  ISSUES FOUND: {len(issues)}")
        for iss in issues[:20]:
            print(f"    ✗ {iss}")
        if len(issues) > 20:
            print(f"    ... and {len(issues)-20} more")
    else:
        print("  No issues found — dataset looks good!")

    print("=" * 55)

    # ── Show 3 sample corruptions ─────────────────────────────────────────
    print("\n  SAMPLE CORRUPTIONS (spot-check these manually):\n")
    shown = 0
    for entry in data:
        for err in entry.get("injected_errors", [])[:1]:
            c = err.get("correct", "")
            w = err.get("wrong",   "")
            if c != w and shown < 5:
                print(f"  Paper: {entry['paper'][:50]}")
                print(f"    CORRECT:   {c[:100]}")
                print(f"    CORRUPTED: {w[:100]}")
                print()
                shown += 1
        if shown >= 5:
            break

    return len(issues) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="data/skepticbench_full.json")
    args = parser.parse_args()
    ok = verify(args.json)
    sys.exit(0 if ok else 1)