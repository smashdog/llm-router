#!/usr/bin/env bash
# One OpenAI-compatible front door for llama.cpp router, KVMem and NInfer.
# Starts with no engine and no model loaded; the requested model chooses the engine.
# On the first run a minimal config.json is created; configure engines in the web UI.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

PY="${ROUTER_PYTHON:-python3}"
if [[ -z "${ROUTER_PYTHON:-}" && -x "$root/.venv/bin/python" ]]; then
    PY="$root/.venv/bin/python"
fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "llm-router: python3 not found; install Python 3.10+ or set ROUTER_PYTHON" >&2
    exit 1
fi
if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "llm-router: Python 3.10+ is required (found $("$PY" -V 2>&1))" >&2
    exit 1
fi
if ! "$PY" -c "import requests" >/dev/null 2>&1; then
    echo "llm-router: missing dependency 'requests'" >&2
    echo "  Debian/Ubuntu (PEP 668) recommended:" >&2
    echo "    python3 -m venv '$root/.venv' && '$root/.venv/bin/pip' install -r '$root/requirements.txt'" >&2
    echo "  other environments: $PY -m pip install -r '$root/requirements.txt'" >&2
    exit 1
fi

exec "$PY" "$root/router.py" "$@"
