"""
FastAPI backend for FactGuard.

Run:
    uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env_utils import load_env


load_env()


def _parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:3000", "http://localhost:5173"]


def _refresh_groq_clients(groq_api_key: str) -> None:
    """
    Refresh module-level Groq clients.
    These modules initialize clients at import time, so we rebind them when key changes.
    """
    if not groq_api_key:
        return

    try:
        from groq import Groq

        client = Groq(api_key=groq_api_key)

        import modules.atomicizer as atomicizer
        import modules.query_generator as query_generator
        import modules.judge as judge
        import modules.cove_loop as cove_loop
        import modules.editor as editor

        atomicizer.client = client
        query_generator.client = client
        judge.client = client
        cove_loop.client = client
        editor.client = client
    except Exception:
        # If refresh fails, pipeline execution will still raise clear provider errors.
        pass


class VerifyRequest(BaseModel):
    summary: str = Field(..., min_length=1, description="Summary text to fact-check")
    verbose: bool = Field(False, description="Include verbose pipeline prints on server logs")


class VerifyResponse(BaseModel):
    success: bool
    result: dict[str, Any]
    meta: dict[str, Any]


app = FastAPI(
    title="FactGuard API",
    version="1.0.0",
    description="Backend API for FactGuard Skeptical CoVe-RAG",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "FactGuard API is running"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "groq_key_configured": bool(os.getenv("GROQ_API_KEY")),
        "gemini_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/verify", response_model=VerifyResponse)
async def verify_summary(payload: VerifyRequest) -> VerifyResponse:
    summary = payload.summary.strip()
    if not summary:
        raise HTTPException(status_code=400, detail="`summary` cannot be empty.")

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is missing. Add it in .env before starting the API.",
        )

    # Ensure module-level Groq clients use the .env key.
    _refresh_groq_clients(os.environ["GROQ_API_KEY"])

    started = time.perf_counter()
    try:
        # Import lazily so app can still boot and expose clear health/errors.
        from pipeline import run_pipeline

        result = await run_in_threadpool(run_pipeline, summary, payload.verbose)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return VerifyResponse(
        success=True,
        result=result,
        meta={
            "elapsed_ms": elapsed_ms,
            "groq_key_configured": bool(os.getenv("GROQ_API_KEY")),
            "gemini_key_configured": bool(os.getenv("GEMINI_API_KEY")),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
