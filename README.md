# MetaJudge AI

MetaJudge AI verifies AI/ML paper summaries by decomposing them into atomic claims, searching for adversarial evidence, judging support or contradiction, and applying targeted corrections when a contradiction is confirmed.

## Clean Repository Layout

```text
MetaJudge-AI/
|-- Backend/            # Working Groq/Gemini demo backend
|-- Frontend/           # React demo frontend for the backend
|-- results/            # Preserved metrics and result summaries
|-- figures/            # Paper/report figures
|-- images/             # Report images
|-- paper.tex
`-- major_project_report_main.tex
```

The old root-level Python pipeline/modules/evaluation copies and the old `MajorAyushh/` wrapper were removed. Run the maintained backend and frontend directly from the root folders below.

## Demo Stack

Use this for presentations.

```bash
cd Backend
cp sample.env .env
# Add GROQ_API_KEY. GEMINI_API_KEY is optional.
./run_demo.sh
```

For the React demo:

```bash
cd Backend
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

```bash
cd Frontend
npm install
npm run dev
```

React calls `POST http://127.0.0.1:8000/api/v1/verify`.

## CLI And Metrics

```bash
cd Backend
pip install -r requirements.txt
python3 pipeline.py --bench
```

The backend expects `GROQ_API_KEY`. `GEMINI_API_KEY` is optional and enables deeper PDF/second-opinion verification.

## Preserved Outputs

Keep these files for the report/demo narrative:

- `results/RESULTS_SUMMARY.md`
- `results/ablation.json`

SQLite evidence caches, virtual environments, build outputs, `node_modules`, and local `.env` files are generated locally and ignored.

## API

- Backend: `Backend`, Groq/Gemini, `POST /api/v1/verify`.
- Frontend: `Frontend`, Vite/React, defaults to `http://127.0.0.1:8000`.
