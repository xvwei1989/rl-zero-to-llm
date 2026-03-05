#!/usr/bin/env bash
set -euo pipefail

# One-command local setup.
# Usage:
#   bash scripts/setup.sh
# Then:
#   source .venv/bin/activate

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python3}

if command -v uv >/dev/null 2>&1; then
  echo "[setup] uv detected → using uv (fast)"
  uv venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  echo "[setup] uv not found → using venv + pip"
  "$PYTHON" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
fi

echo "[setup] done. Activate with: source .venv/bin/activate" 
