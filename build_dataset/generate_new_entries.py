"""
build_dataset/generate_new_entries.py
======================================
Fetches AI/ML papers from arXiv and auto-generates SkepticBench entries.
Rate-limit safe — uses llama-3.1-8b-instant + exponential backoff.

Run from project root:
    python build_dataset/generate_new_entries.py --count 75 --out data/skepticbench_new75.json
"""

import os, sys, json, re, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arxiv
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

client    = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
MODEL     = "llama-3.1-8b-instant"   # 131k tokens/min — safe for bulk
MIN_DELAY = 3.0                        # seconds between every API call
MAX_RETRY = 5

ARXIV_IDS = [
    "1706.03762","1810.04805","2005.14165","2303.08774","2302.13971",
    "2307.09288","2310.06825","1907.11692","1910.10683","2203.02155",
    "2305.14251","2210.08726","2310.11511","2309.11495","2305.11747",
    "2109.07958","2306.05685","2308.11495","2307.13528","2310.00741",
    "2201.11903","2303.11366","2210.03629","2305.20050","2303.17651",
    "2305.14325","2302.04761","2305.13534","2311.05684","2106.09685",
    "2305.18290","2304.01196","2112.09332","2303.16634","2103.00020",
    "2112.10752","2301.13688","2304.08485","2310.03744","2005.11401",
    "2302.12813","2009.03300","1905.00537","1803.05457","1606.05250",
    "1806.03822","2107.03374","2110.14168","2206.14858","2212.09561",
    "2210.07316","2302.07712","2205.01068","2303.17580","2308.11432",
    "2304.03442","2305.16291","2306.06070","2307.15043","2302.07459",
    "2209.07858","2212.08073","2305.15324","2309.06180","2306.11644",
    "2112.00114","2210.17323","2307.03025","2305.14233","2204.02311",
    "2210.11610","2305.06983","2304.05197","2305.10601","2306.08640",
]

# deduplicate while preserving order
seen, ARXIV_IDS_UNIQUE = set(), []
for x in ARXIV_IDS:
    if x not in seen:
        seen.add(x)
        ARXIV_IDS_UNIQUE.append(x)


PROMPT = """You are building a hallucination-detection benchmark for AI/ML papers.

Given a paper's title, authors, year, and abstract, return ONLY a valid JSON object.
No markdown. No explanation. No code fences. Just the raw JSON.

Format:
{
  "summary": "3-5 sentence factual summary of the paper",
  "atomic_facts": [
    "Self-contained factual claim 1 with explicit subject.",
    "Self-contained factual claim 2 with explicit subject."
  ],
  "labels": ["true", "false", "true"],
  "corrupted_facts": [
    "Self-contained factual claim 1 with explicit subject.",
    "Corrupted version of claim 2 with ONE subtle error.",
    "Self-contained factual claim 3 with explicit subject."
  ]
}

Rules:
- atomic_facts: 8 to 12 items. Each is one claim. Subject must be explicit (no pronouns).
- labels: same length as atomic_facts. Strings "true" or "false". About 30% should be "false". Space them out.
- corrupted_facts: same length as atomic_facts. For "true" items, copy exactly. For "false" items, change ONE thing (a number, a name, a year, or a method name). Keep the rest identical.
- The JSON must be valid. Use only double quotes. No trailing commas."""


def call_groq_safe(messages: list, max_tokens: int = 1500) -> str:
    """Call Groq with retry on rate limit."""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            time.sleep(MIN_DELAY)
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except (RateLimitError, APIStatusError) as e:
            wait = min(2 ** attempt * 10, 120)
            print(f"    [rate limit] attempt {attempt}/{MAX_RETRY} — waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            if attempt == MAX_RETRY:
                raise
            print(f"    [error] {e} — retrying in 5s...")
            time.sleep(5)
    raise RuntimeError("Groq call failed after all retries")


def fetch_paper(arxiv_id: str) -> dict | None:
    try:
        results = list(arxiv.Search(id_list=[arxiv_id], max_results=1).results())
        if not results:
            print(f"    [arXiv] no results for {arxiv_id}")
            return None
        p = results[0]
        return {
            "arxiv_id": arxiv_id,
            "title":    p.title,
            "authors":  ", ".join(a.name for a in p.authors[:4]),
            "year":     p.published.year,
            "abstract": p.summary[:1200],
        }
    except Exception as e:
        print(f"    [arXiv error] {e}")
        return None


def normalise_label(val) -> str:
    """
    FIX for the silent-failure bug:
    JSON true/false parses to Python True/False (booleans).
    Also handle variations like "True", "False", 1, 0.
    Always return the string "true" or "false".
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    s = str(val).strip().lower()
    return "true" if s in ("true", "1", "yes") else "false"


def generate_entry(paper: dict) -> dict | None:
    user_msg = (
        f"Title: {paper['title']}\n"
        f"Authors: {paper['authors']}\n"
        f"Year: {paper['year']}\n\n"
        f"Abstract:\n{paper['abstract']}"
    )
    try:
        raw = call_groq_safe(
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1500,
        )
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$",       "", raw)
        raw = raw.strip()

        parsed = json.loads(raw)

        # Normalise labels to strings (fixes the boolean bug)
        parsed["labels"] = [normalise_label(l) for l in parsed.get("labels", [])]

        return parsed

    except json.JSONDecodeError as e:
        print(f"    [JSON parse error] {e}")
        print(f"    Raw response (first 300 chars): {raw[:300]}")
        return None
    except Exception as e:
        print(f"    [Groq error] {e}")
        return None


def validate(g: dict) -> bool:
    facts     = g.get("atomic_facts", [])
    labels    = g.get("labels", [])
    corrupted = g.get("corrupted_facts", [])

    if len(facts) < 5:
        print(f"    [skip] only {len(facts)} facts — need at least 5")
        return False
    if len(labels) != len(facts):
        print(f"    [skip] label count {len(labels)} != fact count {len(facts)}")
        return False
    if len(corrupted) != len(facts):
        print(f"    [skip] corrupted count {len(corrupted)} != fact count {len(facts)}")
        return False

    # Count real differences (labels are now guaranteed strings)
    false_count = labels.count("false")
    real_diffs  = sum(
        1 for f, l, c in zip(facts, labels, corrupted)
        if l == "false" and c.strip() != f.strip()
    )

    print(f"    labels: {len(labels)} total, {false_count} false, {real_diffs} real diffs")

    if false_count == 0:
        print(f"    [skip] no false labels at all")
        return False
    if real_diffs == 0:
        print(f"    [skip] false labels exist but corruptions are identical to originals")
        return False

    return True


def build_entry(paper: dict, g: dict, entry_id: int) -> dict:
    facts     = g["atomic_facts"]
    labels    = g["labels"]
    corrupted = g["corrupted_facts"]

    injected = [
        {
            "type":    "corrupted_fact",
            "correct": f,
            "wrong":   c,
            "fact":    c,
            "source":  paper["title"],
        }
        for f, l, c in zip(facts, labels, corrupted)
        if l == "false" and c.strip() != f.strip()
    ]

    return {
        "id":              f"sb{entry_id:03d}",
        "paper":           paper["title"],
        "authors":         paper["authors"],
        "year":            paper["year"],
        "arxiv_id":        paper["arxiv_id"],
        "summary":         g.get("summary", " ".join(corrupted[:5])),
        "atomic_facts":    facts,
        "corrupted_facts": corrupted,
        "labels":          labels,
        "injected_errors": injected,
        "clean":           len(injected) == 0,
    }


def run(count: int, out_path: str, start_id: int = 26):
    output, failed = [], 0

    print(f"Generating {count} entries starting from ID sb{start_id:03d}")
    print(f"Model: {MODEL}  |  Delay: {MIN_DELAY}s per call\n")

    for arxiv_id in ARXIV_IDS_UNIQUE:
        if len(output) >= count:
            break

        entry_id = start_id + len(output)
        print(f"[{len(output)+1:03d}/{count}] arXiv:{arxiv_id}")

        # Step 1 — fetch from arXiv
        paper = fetch_paper(arxiv_id)
        if not paper:
            failed += 1
            continue
        print(f"  Title: {paper['title'][:65]}")

        # Step 2 — generate with Groq
        print(f"  Calling Groq...")
        generated = generate_entry(paper)
        if not generated:
            failed += 1
            continue

        # Step 3 — validate
        if not validate(generated):
            failed += 1
            continue

        # Step 4 — build entry
        entry = build_entry(paper, generated, entry_id)
        output.append(entry)
        print(f"  DONE — {len(entry['atomic_facts'])} facts, {len(entry['injected_errors'])} corruptions")

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_facts = sum(len(e["atomic_facts"])    for e in output)
    total_errs  = sum(len(e["injected_errors"]) for e in output)

    print(f"\n{'='*55}")
    print(f"  Completed: {len(output)} entries  ({failed} failed/skipped)")
    print(f"  Atomic facts    : {total_facts}")
    print(f"  Corruptions     : {total_errs}")
    print(f"  Clean entries   : {sum(1 for e in output if e['clean'])}")
    print(f"  Saved to        : {out_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count",    type=int, default=75,
                   help="Number of entries to generate")
    p.add_argument("--out",      default="data/skepticbench_new75.json",
                   help="Output JSON path")
    p.add_argument("--start-id", type=int, default=26,
                   help="Starting ID number (26 if appending to 25 existing)")
    args = p.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    run(args.count, args.out, args.start_id)