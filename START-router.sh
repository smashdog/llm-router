#!/usr/bin/env bash
# One OpenAI-compatible front door for llama.cpp router and KVMem.
# Starts with no engine and no model loaded; the requested model chooses the engine.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

PY="${ROUTER_PYTHON:-python3}"
exec "$PY" "$root/router.py" "$@"
