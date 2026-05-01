"""
Shared runtime helpers for entrypoints and API-key-driven client refreshes.

With the move to NVIDIA NIM, there is no per-module Groq client to refresh.
This module is kept for backwards compatibility.
"""

from __future__ import annotations


def refresh_groq_clients(api_key: str = "", module_names=None) -> None:
    """No-op — retained for API compatibility with older callers."""
    pass
