"""
FastAPI backend for MetaJudge AI v2.0

Run:
    uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from env_utils import load_env
from modules.nvidia_client import health_check, MODELS
from modules.cache_layer import get_stats, clear_cache

load_env()

app = FastAPI(
    title="MetaJudge AI",
    version="2.0",
    description="Adversarial Hallucination Detection & Correction API — powered by NVIDIA NIM",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    from modules.cache_layer import _get_conn
    _get_conn()  # initialise DB and table
    key_set = bool(os.environ.get("NVIDIA_API_KEY"))
    print(f"NVIDIA_API_KEY: {'set ✓' if key_set else 'MISSING ✗'}")


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    loop = asyncio.get_event_loop()
    checks = await loop.run_in_executor(None, health_check)
    return {
        "status": "ok",
        "models": checks,
        "nvidia_key_set": bool(os.environ.get("NVIDIA_API_KEY")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ── Analyze (sync) ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Summary text to fact-check")
    mode: str = Field("fast", description="'fast' or 'full'")


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not os.getenv("NVIDIA_API_KEY"):
        raise HTTPException(status_code=400, detail="NVIDIA_API_KEY is missing.")
    t0 = time.time()
    loop = asyncio.get_event_loop()
    from pipeline import run_pipeline
    result = await loop.run_in_executor(
        None, lambda: run_pipeline(req.text, verbose=False, mode=req.mode)
    )
    result["processing_time_ms"] = round((time.time() - t0) * 1000)
    return result


# ── Analyze (SSE stream) ──────────────────────────────────────────────────

@app.get("/analyze/stream")
async def analyze_stream(text: str, mode: str = "fast"):
    if not os.getenv("NVIDIA_API_KEY"):
        raise HTTPException(status_code=400, detail="NVIDIA_API_KEY is missing.")

    async def event_generator():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def stage_callback(stage_name: str, result: Any, duration_ms: float):
            # Sanitize result for JSON serialization
            safe_result = {}
            if isinstance(result, dict):
                for k, v in result.items():
                    try:
                        json.dumps(v)
                        safe_result[k] = v
                    except (TypeError, ValueError):
                        safe_result[k] = str(v)
            else:
                safe_result = {"value": str(result)}

            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"stage": stage_name, "status": "complete",
                 "result": safe_result, "duration_ms": round(duration_ms, 1)}
            )

        def run():
            from pipeline import run_pipeline
            r = run_pipeline(text, verbose=False, mode=mode, stage_callback=stage_callback)
            loop.call_soon_threadsafe(queue.put_nowait, {"stage": "done", "final_output": r})

        loop.run_in_executor(None, run)

        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, default=str)}\n\n"
            if event.get("stage") == "done":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Benchmark ──────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    sample_count: int = Field(5, ge=1, le=25)
    mode: str = Field("fast", description="'fast' or 'full'")


@app.post("/benchmark/run")
async def benchmark_run(req: BenchmarkRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "result": None}
    background_tasks.add_task(_run_benchmark_job, job_id, req.sample_count, req.mode)
    return {"job_id": job_id, "status": "running"}


def _run_benchmark_job(job_id: str, sample_count: int, mode: str):
    try:
        with open("data/skepticbench_sample.json") as f:
            samples = json.load(f)[:sample_count]

        from pipeline import run_pipeline
        from evaluation.skeptic_score import ClaimResult, BenchmarkReport

        report = BenchmarkReport()
        for item in samples:
            output = run_pipeline(
                item["summary"],
                verbose=False,
                mode=mode,
                facts_override=item.get("corrupted_facts") or item.get("atomic_facts"),
            )
            injected_by_fact = {e["fact"]: e for e in item.get("injected_errors", [])}
            for result in output["results"]:
                injected = injected_by_fact.get(result["fact"])
                report.add(ClaimResult(
                    fact=result["fact"],
                    ground_truth="hallucinated" if injected else "correct",
                    verdict=result["verdict"],
                    cove_applied=result.get("cove_applied", False),
                    cove_meta_verdict=result.get("cove_meta_verdict"),
                    correction=next(
                        (c["correction"] for c in output["corrections"] if c["fact"] == result["fact"]),
                        "",
                    ),
                    source_url=result.get("evidence_source", ""),
                    ground_truth_correction=(injected or {}).get("ground_truth_correction", ""),
                    detection_confidence=float(result.get("detection_confidence", 0.65)),
                ))

        _jobs[job_id] = {
            "status": "complete",
            "result": {
                "precision": report.precision(),
                "recall": report.recall(),
                "f1": report.f1(),
                "false_positive_rate": report.false_positive_rate(),
                "skeptic_score": report.skeptic_score(),
                "total_claims": report.total_claims,
                "tp": report.true_positive,
                "fp": report.false_positive,
                "tn": report.true_negative,
                "fn": report.false_negative,
            },
        }
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "result": {"error": str(exc)}}


@app.get("/benchmark/status/{job_id}")
async def benchmark_status(job_id: str):
    return _jobs.get(job_id, {"status": "not_found"})


# ── Cache endpoints ────────────────────────────────────────────────────────

@app.get("/cache/stats")
async def cache_stats():
    return get_stats()


@app.delete("/cache")
async def cache_clear():
    clear_cache()
    return {"status": "cleared"}


# ── Root ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "MetaJudge AI API v2.0 is running", "models": MODELS}
