"""
build_dataset/groq_utils.py
============================
Retry wrapper for Groq API calls with exponential backoff.
Import this in any build_dataset script instead of calling
client.chat.completions.create() directly.

Groq free tier limits (per minute):
  llama-3.3-70b-versatile : 30 req/min, 6,000 tokens/min   ← too slow for bulk
  llama-3.1-70b-versatile : 30 req/min, 6,000 tokens/min
  llama-3.1-8b-instant    : 30 req/min, 131,072 tokens/min  ← use this for bulk
  mixtral-8x7b-32768      : 30 req/min, 5,000 tokens/min

Strategy:
  - Use llama-3.1-8b-instant for dataset generation (bulk, high token limit)
  - Use llama-3.3-70b-versatile only for pipeline judge + CoVe (accuracy matters)
  - Retry up to 5 times with exponential backoff on rate limit errors
  - Hard sleep of MIN_DELAY seconds between every call
"""

import time
import os
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

# Use the 8B model for bulk generation — 131k tokens/min vs 6k for 70B
BULK_MODEL   = "llama-3.1-8b-instant"
STRONG_MODEL = "llama-3.3-70b-versatile"

MIN_DELAY    = 2.5   # seconds between every call (stays under 30 req/min)
MAX_RETRIES  = 5


def call_groq(
    messages: list[dict],
    model: str = BULK_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """
    Call Groq with automatic retry on rate limit errors.
    Returns the response text or raises after MAX_RETRIES attempts.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(MIN_DELAY)   # always wait before calling
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            wait = min(2 ** attempt * 10, 120)   # 20s, 40s, 80s, 120s, 120s
            print(f"    [rate limit] attempt {attempt}/{MAX_RETRIES} — waiting {wait}s...")
            time.sleep(wait)

        except APIStatusError as e:
            if e.status_code == 429:
                wait = min(2 ** attempt * 10, 120)
                print(f"    [429 error] attempt {attempt}/{MAX_RETRIES} — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"    [error] {e} — retrying in 5s...")
            time.sleep(5)

    raise RuntimeError(f"Groq call failed after {MAX_RETRIES} attempts")