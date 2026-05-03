# MetaJudge AI Frontend

React/Vite demo UI for `../Backend`.

The UI calls the Groq demo backend at `/api/v1/verify` and focuses on the input/results flow. Backend health details stay available through `/health` instead of a separate diagnostics card in the app.

## Run

Start the Groq FastAPI backend first:

```bash
cd ../Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

Then run the frontend:

```bash
cd ../Frontend
npm install
npm run dev
```

Set a custom API URL with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```
