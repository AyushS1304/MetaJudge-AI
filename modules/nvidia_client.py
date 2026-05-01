"""
modules/nvidia_client.py — Single LLM Gateway (NVIDIA NIM)
------------------------------------------------------------
Every LLM call in this project routes through this module.
NVIDIA NIM exposes an OpenAI-compatible API so we use the openai library.

Usage:
    from modules.nvidia_client import nvidia_chat, nvidia_chat_json, MODELS
    response = nvidia_chat(messages, role="judge")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from openai import OpenAI
from env_utils import load_env

# Ensure .env is loaded before we read the key
load_env()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "[NVIDIA_API_KEY_HERE]")
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Model roster — all tasks use NVIDIA NIM exclusively
MODELS = {
    "fast":       "meta/llama-3.1-8b-instruct",            # atomicizer, query-gen, consistency
    "judge":      "nvidia/llama-3.1-nemotron-70b-instruct", # judge, cove loop
    "editor":     "meta/llama-3.3-70b-instruct",            # surgical correction
    "escalation": "meta/llama-3.1-405b-instruct",           # deep verification (low-confidence only)
}

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a shared OpenAI client pointed at NVIDIA NIM."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=NVIDIA_API_KEY, base_url=BASE_URL)
    return _client


def nvidia_chat(
    messages: list[dict],
    role: str = "fast",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    retries: int = 4,
) -> str:
    """
    Single call interface for all LLM calls in this project.

    Args:
        messages:    OpenAI-format message list.
        role:        Key into MODELS dict — selects the model tier.
        temperature: Sampling temperature.
        max_tokens:  Maximum response tokens.
        retries:     Number of retry attempts with exponential backoff.

    Returns:
        Response text string.

    Raises:
        RuntimeError after all retries are exhausted.
    """
    client = get_client()
    model = MODELS[role]
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many" in err or "rate" in err.lower():
                wait = (2 ** attempt) * 2
                logging.warning(
                    f"NVIDIA rate limit hit. Waiting {wait}s "
                    f"(attempt {attempt + 1}/{retries})"
                )
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"nvidia_chat failed after {retries} attempts: {e}"
                )
    raise RuntimeError("nvidia_chat: all retries exhausted")


def nvidia_chat_json(
    messages: list[dict],
    role: str = "fast",
    retries: int = 4,
) -> dict:
    """
    Wrapper that calls nvidia_chat and parses JSON response.
    Strips markdown fences before parsing.

    Returns:
        Parsed dict.

    Raises:
        ValueError if JSON parsing fails.
    """
    raw = nvidia_chat(messages, role=role, retries=retries)
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed. Raw output was:\n{raw}\nError: {e}")


def health_check() -> dict:
    """Test all model tiers. Returns {role: bool} availability map."""
    results = {}
    for role, model in MODELS.items():
        try:
            nvidia_chat(
                [{"role": "user", "content": "Reply OK"}],
                role=role,
                max_tokens=5,
            )
            results[role] = True
        except Exception:
            results[role] = False
    return results
