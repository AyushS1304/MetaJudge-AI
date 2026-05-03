"""
build_dataset/generate_new_entries.py
======================================
Rate-limit safe SkepticBench generator (FIXED VERSION)
"""

import os, sys, json, re, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arxiv
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
MODEL = "llama-3.1-8b-instant"

MIN_DELAY = 4.0     # increased safety delay
MAX_RETRY = 5

ARXIV_CACHE = {}    # 🔥 cache added

ARXIV_IDS = [
    "1706.03762","1810.04805","2005.14165","2303.08774","2302.13971",
    "2307.09288","2310.06825","1907.11692","1910.10683","2203.02155",
    "2305.14251","2210.08726","2310.11511","2309.11495","2305.11747",
    "2109.07958","2306.05685","2308.11495","2307.13528","2310.00741",
]

seen = set()
ARXIV_IDS_UNIQUE = []
for x in ARXIV_IDS:
    if x not in seen:
        seen.add(x)
        ARXIV_IDS_UNIQUE.append(x)


PROMPT = """You are building a hallucination-detection benchmark for AI/ML papers.

Return ONLY valid JSON:
{
  "summary": "3-5 sentence summary",
  "atomic_facts": ["fact1", "fact2"],
  "labels": ["true", "false"],
  "corrupted_facts": ["f1", "f2"]
}

Rules:
- 8–12 atomic facts
- ~30% false
- corrupted_facts: modify ONLY false facts
"""


# ---------------- GROQ CALL ----------------
def call_groq_safe(messages: list, max_tokens: int = 1500) -> str:
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
            wait = min(2 ** attempt * 5, 120)
            print(f"[Groq rate limit] retry {attempt} wait {wait}s")
            time.sleep(wait)

        except Exception as e:
            print(f"[Groq error] {e}")
            time.sleep(3)

    raise RuntimeError("Groq failed after retries")


# ---------------- ARXIV SAFE FETCH ----------------
def fetch_paper(arxiv_id: str):
    if arxiv_id in ARXIV_CACHE:
        return ARXIV_CACHE[arxiv_id]

    for attempt in range(1, MAX_RETRY + 1):
        try:
            time.sleep(1.5)  # arXiv throttle

            results = list(
                arxiv.Search(id_list=[arxiv_id], max_results=1).results()
            )

            if not results:
                return None

            p = results[0]

            paper = {
                "arxiv_id": arxiv_id,
                "title": p.title,
                "authors": ", ".join(a.name for a in p.authors[:4]),
                "year": p.published.year,
                "abstract": p.summary[:1200],
            }

            ARXIV_CACHE[arxiv_id] = paper
            return paper

        except Exception as e:
            wait = min(2 ** attempt * 5, 60)
            print(f"[arXiv retry {attempt}] wait {wait}s -> {e}")
            time.sleep(wait)

    print(f"[arXiv FAIL] skipping {arxiv_id}")
    return None


# ---------------- NORMALIZE ----------------
def normalise_label(val):
    if isinstance(val, bool):
        return "true" if val else "false"
    return "true" if str(val).lower() in ("true", "1", "yes") else "false"


# ---------------- GENERATE ----------------
def generate_entry(paper):
    try:
        raw = call_groq_safe([
            {"role": "system", "content": PROMPT},
            {"role": "user", "content":
                f"{paper['title']}\n{paper['authors']}\n{paper['abstract']}"
            },
        ])

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        parsed["labels"] = [normalise_label(x) for x in parsed.get("labels", [])]

        return parsed

    except Exception as e:
        print("[generate error]", e)
        return None


# ---------------- VALIDATE ----------------
def validate(g):
    if not g:
        return False

    f = g.get("atomic_facts", [])
    l = g.get("labels", [])
    c = g.get("corrupted_facts", [])

    if len(f) < 5 or len(f) != len(l) or len(f) != len(c):
        return False

    return "false" in l


# ---------------- BUILD ----------------
def build_entry(paper, g, idx):
    return {
        "id": f"sb{idx:03d}",
        "paper": paper["title"],
        "arxiv_id": paper["arxiv_id"],
        "summary": g.get("summary", ""),
        "atomic_facts": g.get("atomic_facts", []),
        "corrupted_facts": g.get("corrupted_facts", []),
        "labels": g.get("labels", []),
    }


# ---------------- MAIN LOOP ----------------
def run(count, out_path, start_id=1):
    output = []
    failed = 0

    for arxiv_id in ARXIV_IDS_UNIQUE:
        if len(output) >= count:
            break

        print(f"\n[{len(output)+1}/{count}] {arxiv_id}")

        paper = fetch_paper(arxiv_id)
        if not paper:
            failed += 1
            continue

        print("  ✔ Paper fetched")

        gen = generate_entry(paper)
        if not validate(gen):
            failed += 1
            continue

        entry = build_entry(paper, gen, start_id + len(output))
        output.append(entry)

        print("  ✔ Generated")

        time.sleep(2)  # 🔥 global throttle

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nDONE")
    print("Saved:", out_path)
    print("Failed:", failed)


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", default="data/output.json")
    args = parser.parse_args()

    run(args.count, args.out)