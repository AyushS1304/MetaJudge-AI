# MetaJudge AI

MetaJudge AI is a research-focused system that verifies claims in AI/ML summaries and papers. It decomposes inputs into atomic claims, retrieves supporting and adversarial evidence, runs targeted verification routines, and produces an explainable judgement along with corrective suggestions.

## Key Features
- Claim decomposition (atomicizer)
- Evidence retrieval and caching
- Multi-stage verification (deep verifier, second opinion)
- Explainable judgements and corrections
- Evaluation against SkepticBench datasets

## Repository overview

Top-level layout (important folders):

- `Backend/` — FastAPI backend, core pipeline, evaluation scripts, and requirements.
- `Frontend/` — Vite + React demo UI that calls the backend verification API.
- `modules/` — Core pipeline modules used by the backend (atomicizer, retriever, verifier, judge, etc.).
- `data/` — Datasets and SkepticBench JSONs used for evaluation and demos.
- `results/` — Stored run outputs, ablations, and summaries.
- `figures/`, `images/` — Assets used in reporting and the project writeup.

Files of note

- `Backend/fastapi_app.py` — API entrypoint (serves `/api/v1/verify`).
- `Backend/pipeline.py` — Orchestrates verification pipelines and CLI entrypoints (benchmarks).
- `modules/atomicizer.py` — Breaks text into atomic claims.
- `modules/pdf_extractor.py` — PDF extraction helpers (if verifying PDFs).
- `modules/retriever.py` — Evidence retrieval against provided corpora or web models.
- `modules/query_generator.py` — Generates targeted verification queries from claims.
- `modules/deep_verifier.py` — Runs deep verification checks (LLM/Cot loops).
- `modules/judge.py` — Aggregates evidence and issues final judgement.
- `modules/second_opinion.py` — Optional secondary checks (different model/provider).

## Explainability (How MetaJudge reasons)

The system is designed to be transparent about its decisions:

1. Decompose: `atomicizer` converts input text into discrete claims.
2. Retrieve: `retriever` gathers documents, snippets, and cached evidence relevant to each claim.
3. Query & Verify: `query_generator` composes focused prompts; `deep_verifier` executes verification routines (may call different LLM providers or adversarial retrieval).
4. Judge: `judge` scores each claim using evidence signals (support, contradiction, confidence) and composes an explainable verdict.
5. Second opinion: `second_opinion` optionally re-checks low-confidence or contradictory items with an alternate model or API.

Each judgement includes the claim, the supporting/contradicting evidence, a confidence score, and a short explanation of the reasoning steps used.

## Quickstart (Backend)

Prerequisites: Python 3.10+, pip, optional: virtualenv. API keys: `GROQ_API_KEY` (required for Groq connector), `GEMINI_API_KEY` (optional).

Windows PowerShell example:

```powershell
cd "c:/Users/Ayush Sharma/OneDrive/Desktop/MetaJudge AI/Backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy sample.env .env
# Edit .env to add GROQ_API_KEY (and GEMINI_API_KEY if available)
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

Then run the frontend (separate terminal):

```powershell
cd "c:/Users/Ayush Sharma/OneDrive/Desktop/MetaJudge AI/Frontend"
npm install
npm run dev
```

The demo UI posts verification requests to `POST http://127.0.0.1:8000/api/v1/verify`.

## Running CLI evaluation / benchmarks

From `Backend/`:

```powershell
cd "c:/Users/Ayush Sharma/OneDrive/Desktop/MetaJudge AI/Backend"
pip install -r requirements.txt
python pipeline.py --bench
```

This runs evaluation harnesses that use the JSON files under `data/` (SkepticBench variants).

## Data & Results

- Example datasets: `data/skepticbench_sample.json`, `data/skepticbench_full.json` (see `build_dataset/` for generation tools).
- Results and analysis outputs are written to `results/` (keep `RESULTS_SUMMARY.md` and `ablation.json` for reports).

## Frontend

The `Frontend/` folder is a small Vite + React TypeScript demo intended to show the verification UI. It uses the `/api/v1/verify` endpoint.

## Development notes

- Keep API keys out of source control — `sample.env` is provided as a template.
- Evidence retrieval results are cached (SQLite or local caches) to speed repeated runs and reduce API usage.
- To extend the pipeline, add or replace modules in `modules/` and wire them into `Backend/pipeline.py`.

## Contributing

If you plan to contribute, please open an issue describing your change and follow common GitHub PR workflow. Tests live in `tests/`.

## Contact

For questions or collaboration, open an issue or contact the maintainer listed in the repository metadata.

