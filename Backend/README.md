# MetaJudge Demo Backend

Working Groq/Gemini backend for the live demo.

## What It Contains

- `pipeline.py`: six-step Groq/Gemini verification and correction pipeline.
- `fastapi_app.py`: API used by the React frontend.
- `streamlit_app.py`: self-contained Streamlit demo.
- `modules/`: demo pipeline components.
- `evaluation/skeptic_score.py`: metric helper used by `pipeline.py --bench`.
- `data/skepticbench_sample.json`: small benchmark sample.

## Environment

Create `.env` from the sample:

```bash
cp sample.env .env
```

Required:

```text
GROQ_API_KEY=...
```

Optional:

```text
GEMINI_API_KEY=...
```

## Streamlit Demo

```bash
./run_demo.sh
```

## FastAPI Demo

```bash
pip install -r requirements.txt
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints:

- `GET /`
- `GET /health`
- `POST /api/v1/verify`

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{"summary":"BERT achieved 80.5% F1 on the SQuAD 2.0 benchmark.","verbose":false}'
```

## CLI

```bash
python3 pipeline.py --text "BERT achieved 80.5% F1 on the SQuAD 2.0 benchmark."
python3 pipeline.py --bench
```
