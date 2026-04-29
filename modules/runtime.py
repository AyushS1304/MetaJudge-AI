"""
Shared runtime helpers for entrypoints and API-key-driven client refreshes.
"""

from __future__ import annotations

import importlib
from typing import Iterable

from groq import Groq


GROQ_CLIENT_MODULES = (
    "modules.atomicizer",
    "modules.query_generator",
    "modules.judge",
    "modules.cove_loop",
    "modules.editor",
)


def refresh_groq_clients(api_key: str, module_names: Iterable[str] = GROQ_CLIENT_MODULES) -> None:
    """
    Refresh module-level Groq clients after the API key changes.

    Several modules initialize a Groq client at import time. Streamlit and FastAPI
    allow keys to be provided at runtime, so we rebind those shared clients.
    """
    if not api_key:
        return

    client = Groq(api_key=api_key)
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "client"):
            module.client = client
