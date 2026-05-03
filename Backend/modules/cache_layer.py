"""
modules/cache_layer.py — SQLite-backed Evidence Cache
------------------------------------------------------
Caches arXiv and web search results to avoid rate limiting.
Each entry is keyed by a SHA-256 hash of (source:query).
Entries expire after CACHE_TTL_DAYS.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta

from config import CACHE_DB_PATH, CACHE_TTL_DAYS


def _get_conn() -> sqlite3.Connection:
    """Get (and lazily create) the cache database."""
    os.makedirs(os.path.dirname(CACHE_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evidence_cache (
            query_hash  TEXT PRIMARY KEY,
            query_text  TEXT,
            source      TEXT,
            results_json TEXT,
            created_at  TEXT
        )
    """)
    conn.commit()
    return conn


def get_cached(query: str, source: str) -> list[dict] | None:
    """
    Return cached results for a (query, source) pair, or None if miss/expired.
    """
    h = hashlib.sha256(f"{source}:{query}".encode()).hexdigest()
    conn = _get_conn()
    row = conn.execute(
        "SELECT results_json, created_at FROM evidence_cache WHERE query_hash=?", (h,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    age = datetime.now() - datetime.fromisoformat(row[1])
    if age > timedelta(days=CACHE_TTL_DAYS):
        return None  # expired
    return json.loads(row[0])


def set_cached(query: str, source: str, results: list[dict]) -> None:
    """
    Store results for a (query, source) pair.
    """
    h = hashlib.sha256(f"{source}:{query}".encode()).hexdigest()
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO evidence_cache VALUES (?,?,?,?,?)",
        (h, query, source, json.dumps(results), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_stats() -> dict:
    """Return cache statistics."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM evidence_cache").fetchone()[0]
    conn.close()
    size_mb = os.path.getsize(CACHE_DB_PATH) / 1e6 if os.path.exists(CACHE_DB_PATH) else 0
    return {"total_cached": total, "cache_size_mb": round(size_mb, 3)}


def clear_cache() -> None:
    """Delete all cached entries."""
    conn = _get_conn()
    conn.execute("DELETE FROM evidence_cache")
    conn.commit()
    conn.close()
