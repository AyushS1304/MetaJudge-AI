"""
config.py — Central configuration for MetaJudge AI.

All LLM calls use NVIDIA NIM exclusively.
Set NVIDIA_API_KEY in a .env file or as an environment variable.
"""

import os

from env_utils import load_env

load_env()

# ── NVIDIA NIM (sole LLM provider) ─────────────────────────────────────
NVIDIA_API_KEY   = os.environ.get("NVIDIA_API_KEY", "[NVIDIA_API_KEY_HERE]")
NVIDIA_BASE_URL  = "https://integrate.api.nvidia.com/v1"

MODEL_FAST       = "meta/llama-3.1-8b-instruct"
MODEL_JUDGE      = "nvidia/llama-3.1-nemotron-70b-instruct"
MODEL_EDITOR     = "meta/llama-3.3-70b-instruct"
MODEL_ESCALATION = "meta/llama-3.1-405b-instruct"

# ── Retrieval limits (arXiv rate limit protection) ─────────────────────
ARXIV_DELAY_SECONDS      = 4.0   # sleep between sequential arXiv calls
MAX_ARXIV_CALLS_PER_RUN  = 4     # hard cap per pipeline run
RESULTS_PER_QUERY        = 2     # results fetched per search query
RETRIEVAL_TIMEOUT_SECONDS = 25

# ── Pipeline behaviour ─────────────────────────────────────────────────
MAX_FACTS                  = 20
MIN_EVIDENCE_CHARS         = 30
ESCALATION_CONFIDENCE_GATE = 0.50   # only call 405B if confidence < this
CACHE_TTL_DAYS             = 7
CACHE_DB_PATH              = "data/evidence_cache.db"
