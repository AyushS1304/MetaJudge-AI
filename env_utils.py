"""
Small environment loader with graceful fallback when python-dotenv is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    """
    Load environment variables from .env.
    Prefers python-dotenv when installed; otherwise uses a simple parser.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        return
    except Exception:
        pass

    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        if key and key not in os.environ:
            os.environ[key] = value
