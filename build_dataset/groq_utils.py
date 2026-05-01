"""
build_dataset/groq_utils.py
============================
Retry wrapper for NVIDIA NIM API calls with exponential backoff.
Import this in any build_dataset script instead of calling
the API directly.

Replaces the original Groq-based utility.
"""

import time

from modules.nvidia_client import nvidia_chat

# Model alias for bulk generation
BULK_MODEL   = "fast"       # maps to meta/llama-3.1-8b-instruct
STRONG_MODEL = "judge"      # maps to nvidia/llama-3.1-nemotron-70b-instruct

MIN_DELAY    = 2.5   # seconds between every call
MAX_RETRIES  = 5


def call_groq(
    messages: list[dict],
    model: str = BULK_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """
    Call NVIDIA NIM with automatic retry on rate limit errors.
    Returns the response text or raises after MAX_RETRIES attempts.

    The function name is kept as call_groq for backwards compatibility
    with existing build_dataset scripts.
    """
    # Map old model strings to nvidia_client roles
    role = model if model in ("fast", "judge", "editor", "escalation") else "fast"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(MIN_DELAY)
            return nvidia_chat(
                messages,
                role=role,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RuntimeError as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = min(2 ** attempt * 10, 120)
                print(f"    [rate limit] attempt {attempt}/{MAX_RETRIES} — waiting {wait}s...")
                time.sleep(wait)
            elif attempt == MAX_RETRIES:
                raise
            else:
                print(f"    [error] {e} — retrying in 5s...")
                time.sleep(5)

    raise RuntimeError(f"API call failed after {MAX_RETRIES} attempts")