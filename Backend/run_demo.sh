#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="python3"
PIP_FLAGS=()

if [ -f ".venv/bin/activate" ] && .venv/bin/python -m pip --version >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  PYTHON_BIN="python"
elif python3 -m venv .venv 2>/tmp/metajudge_venv_error.log && .venv/bin/python -m pip --version >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  PYTHON_BIN="python"
else
  echo "Python venv creation is unavailable; using the current Python environment."
  echo "Install python3-venv/ensurepip later if you want an isolated environment."
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    PIP_FLAGS=(--user --break-system-packages)
  else
    echo "pip is not available for python3. Install dependencies manually from requirements.txt."
  fi
fi

if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install "${PIP_FLAGS[@]}" -r requirements.txt
fi

if [ -z "${GROQ_API_KEY:-}" ] && [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "GROQ_API_KEY is not set in the shell."
  echo "You can still enter it in the Streamlit sidebar."
fi

echo "Starting MetaJudge AI Streamlit demo..."
"$PYTHON_BIN" -m streamlit run streamlit_app.py
