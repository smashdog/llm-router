#!/usr/bin/env bash
# One OpenAI-compatible front door for llama.cpp router, KVMem and NInfer.
# Starts with no engine and no model loaded; the requested model chooses the engine.
# On the first run a minimal config.json is created; configure engines in the web UI.
#
# The first start also creates .venv and installs requirements.txt plus the
# optional hf / modelscope download tools.  A failed optional tool install is
# reported but does not stop the gateway; set ROUTER_PYTHON to use an existing
# interpreter and skip the managed venv bootstrap.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

VENV="$root/.venv"
base_py="${ROUTER_PYTHON:-}"
managed_venv=0

check_python() {
    local py="$1"
    if ! command -v "$py" >/dev/null 2>&1; then
        echo "llm-router: python3 not found; install Python 3.10+ or set ROUTER_PYTHON" >&2
        return 1
    fi
    if ! "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
        echo "llm-router: Python 3.10+ is required (found $("$py" -V 2>&1))" >&2
        return 1
    fi
}

if [[ -n "$base_py" ]]; then
    check_python "$base_py" || exit 1
    PY="$base_py"
else
    managed_venv=1
    if [[ -x "$VENV/bin/python" ]]; then
        PY="$VENV/bin/python"
        check_python "$PY" || exit 1
    else
        base_py="$(command -v python3 || true)"
        if [[ -z "$base_py" ]]; then
            echo "llm-router: python3 not found; install Python 3.10+ or set ROUTER_PYTHON" >&2
            exit 1
        fi
        check_python "$base_py" || exit 1
        echo "llm-router: creating virtual environment in $VENV"
        if ! "$base_py" -m venv "$VENV"; then
            echo "llm-router: failed to create $VENV" >&2
            echo "  Debian/Ubuntu: install the python3-venv package, then retry" >&2
            echo "  or set ROUTER_PYTHON to an existing Python 3.10+ interpreter" >&2
            exit 1
        fi
        PY="$VENV/bin/python"
        check_python "$PY" || exit 1
    fi
fi

if [[ "$managed_venv" == "1" ]]; then
    stamp="$VENV/.router-requirements"
    want="$("$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$root/requirements.txt")"
    need_install=0
    if [[ ! -f "$stamp" || "$(cat "$stamp" 2>/dev/null || true)" != "$want" ]]; then
        need_install=1
    fi
    if ! "$PY" -c "import requests, socks, socksio" >/dev/null 2>&1; then
        need_install=1
    fi
    if [[ "$need_install" == "1" ]]; then
        echo "llm-router: installing dependencies from requirements.txt"
        if ! "$PY" -m pip install --disable-pip-version-check -r "$root/requirements.txt"; then
            echo "llm-router: failed to install dependencies; fix the error above and retry" >&2
            exit 1
        fi
        printf '%s\n' "$want" > "$stamp"
    fi

    # Optional tools used by the model search / download page.  Their absence is
    # not fatal: the gateway and inference features keep working without them.
    if ! "$PY" -c "import huggingface_hub" >/dev/null 2>&1 || [[ ! -x "$VENV/bin/hf" ]]; then
        echo "llm-router: installing Hugging Face CLI (hf)"
        if ! "$PY" -m pip install --disable-pip-version-check "huggingface_hub>=0.34"; then
            echo "llm-router: ERROR: failed to install huggingface_hub; Hugging Face / HF-Mirror search and download will be unavailable" >&2
        elif [[ ! -x "$VENV/bin/hf" ]]; then
            echo "llm-router: ERROR: hf command was not created in $VENV/bin; Hugging Face / HF-Mirror search and download will be unavailable" >&2
        fi
    fi
    if ! "$PY" -c "import modelscope" >/dev/null 2>&1 || [[ ! -x "$VENV/bin/modelscope" ]]; then
        echo "llm-router: installing ModelScope CLI"
        if ! "$PY" -m pip install --disable-pip-version-check modelscope; then
            echo "llm-router: ERROR: failed to install modelscope; ModelScope search and download will be unavailable" >&2
        elif [[ ! -x "$VENV/bin/modelscope" ]]; then
            echo "llm-router: ERROR: modelscope command was not created in $VENV/bin; ModelScope search and download will be unavailable" >&2
        fi
    fi
else
    # Explicit ROUTER_PYTHON: never install into an environment we do not manage.
    if ! "$PY" -c "import requests" >/dev/null 2>&1; then
        echo "llm-router: missing dependency 'requests' for $PY" >&2
        echo "  install it with: $PY -m pip install -r '$root/requirements.txt'" >&2
        exit 1
    fi
    "$PY" -c "import huggingface_hub" >/dev/null 2>&1 || \
        echo "llm-router: warning: huggingface_hub is not installed; Hugging Face / HF-Mirror download will be unavailable" >&2
    "$PY" -c "import socksio" >/dev/null 2>&1 || \
        echo "llm-router: warning: socksio is not installed; SOCKS5 proxy for the hf command will be unavailable" >&2
    "$PY" -c "import modelscope" >/dev/null 2>&1 || \
        echo "llm-router: warning: modelscope is not installed; ModelScope download will be unavailable" >&2
fi

exec "$PY" "$root/router.py" "$@"
