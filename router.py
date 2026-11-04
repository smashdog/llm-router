#!/usr/bin/env python3
"""llm-router: one OpenAI-compatible front door for llama.cpp, KVMem and NInfer.

The gateway owns the public port.  It starts with no engine and no model loaded; the
requested model selects the engine.  A cross-engine switch stops the current engine and
starts the other one; a same-engine switch uses the engine's own model-switch API.

Only the Python standard library and requests are used.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hmac
import json
import logging
import mimetypes
import os
import queue
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

import requests


VERSION = "0.4.0"

STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ico": "image/x-icon",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm",
    ".webmanifest": "application/manifest+json; charset=utf-8",
    ".woff2": "font/woff2",
}
UI_API_PREFIXES = (
    "/v1/",
    "/models",
    "/props",
    "/slots",
    "/tools",
    "/health",
    "/router",
    "/version",
)

OPENAI_JSON_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
    "/v1/messages",
}
MODEL_REQUIRED_PATHS = OPENAI_JSON_PATHS

# Model search / download sources.  HF and HF-Mirror share the ``hf`` CLI; the
# mirror is selected with HF_ENDPOINT.  ModelScope uses its own CLI.
DOWNLOAD_SOURCES: dict[str, dict[str, str]] = {
    "huggingface": {"label": "Hugging Face", "tool": "hf", "kind": "hf"},
    "hf-mirror": {"label": "HF-Mirror", "tool": "hf", "kind": "hf"},
    "modelscope": {"label": "ModelScope", "tool": "modelscope", "kind": "modelscope"},
}
DOWNLOAD_SOURCE_ORDER = ("huggingface", "hf-mirror", "modelscope")
DOWNLOAD_LIMIT_MAX = 50
PROXY_TYPES = ("http", "socks5")
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
RESPONSE_SKIP = HOP_BY_HOP | {
    "content-encoding",
    "server",
    "date",
    "access-control-allow-origin",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-expose-headers",
    "access-control-max-age",
}


# ---------------------------------------------------------------------------
# errors


class GatewayError(Exception):
    status = 500
    kind = "server_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelNotFound(GatewayError):
    status = 404
    kind = "invalid_request_error"


class ModelUnavailable(GatewayError):
    status = 503
    kind = "server_error"


class ModelBusy(GatewayError):
    status = 503
    kind = "server_error"


class EngineFailure(GatewayError):
    status = 503
    kind = "server_error"


class MethodNotAllowed(GatewayError):
    status = 405
    kind = "invalid_request_error"


class BadRequest(GatewayError):
    status = 400
    kind = "invalid_request_error"


class RequestTooLarge(GatewayError):
    status = 413
    kind = "invalid_request_error"


class NotFound(GatewayError):
    status = 404
    kind = "invalid_request_error"


class Forbidden(GatewayError):
    status = 403
    kind = "invalid_request_error"


class Unauthorized(GatewayError):
    status = 401
    kind = "authentication_error"


class Unsupported(GatewayError):
    status = 501
    kind = "invalid_request_error"


# ---------------------------------------------------------------------------
# small helpers


def now() -> float:
    return time.time()


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def tail_file(path: Path, lines: int = 80) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            data = fh.read().decode("utf-8", "replace").splitlines()
            return "\n".join(data[-lines:])
    except OSError:
        return ""


def parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_key(value: Any) -> str | None:
    """llama.cpp/KVMem accept 'key', ['key', ...] or 'key1,key2' for api_key."""
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str):
        value = value.split(",")[0].strip()
        return value or None
    return None


def resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


# ---------------------------------------------------------------------------
# proxy / model download helpers

def proxy_url(proxy_cfg: dict[str, Any] | None) -> str | None:
    """Build a requests/CLI proxy URL without ever logging the credentials."""
    if not isinstance(proxy_cfg, dict):
        return None
    kind = str(proxy_cfg.get("type") or "http").strip().lower()
    scheme = "socks5h" if kind == "socks5" else "http"
    host = str(proxy_cfg.get("host") or "").strip()
    port = parse_int(proxy_cfg.get("port"))
    if not host or not port:
        return None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    auth = ""
    username = str(proxy_cfg.get("username") or "")
    password = str(proxy_cfg.get("password") or "")
    if username:
        auth = quote(username, safe="")
        if password:
            auth += ":" + quote(password, safe="")
        auth += "@"
    return f"{scheme}://{auth}{host}:{port}"


def apply_proxy_env(
    env: dict[str, str], proxy_cfg: dict[str, Any] | None, use_proxy: bool
) -> None:
    """Replace inherited proxy settings so "no proxy" really means direct."""
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    if not use_proxy:
        return
    url = proxy_url(proxy_cfg)
    if not url:
        return
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        env[key] = url


def find_tool(name: str) -> Path | None:
    """Prefer the executable next to the running interpreter (the managed .venv)."""
    candidate = Path(sys.executable).parent / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def tool_python(tool: Path) -> str:
    """Interpreter that owns a CLI tool, used for small library calls."""
    for name in ("python", "python3"):
        candidate = tool.parent / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def normalize_repo_id(value: Any) -> str:
    text = str(value or "").strip()
    parts = text.split("/")
    if len(parts) != 2:
        raise BadRequest("模型仓库名必须是 owner/name 格式")
    for part in parts:
        if not part or part in (".", "..") or not re.fullmatch(r"[\w.-]+", part):
            raise BadRequest(f"模型仓库名不合法：{text!r}")
    return text


def safe_repo_file(value: Any) -> str | None:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not text or text.startswith("/"):
        return None
    if any(ord(ch) < 32 for ch in text):
        return None
    parts = text.split("/")
    if any(not part or part in (".", "..") for part in parts):
        return None
    return text


def parse_modelscope_table(text: str) -> tuple[list[dict[str, Any]], int | None]:
    """Parse the human table printed by ``modelscope list``.

    The CLI has no JSON mode for listings; the banner goes to stderr, so stdout
    contains the header, a dashed separator and one row per repository.
    """
    lines = text.splitlines()
    separator = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip()
            and re.fullmatch(r"[- ]+", line)
            and index > 0
            and "repo_id" in lines[index - 1]
        ),
        None,
    )
    if separator is None:
        return [], None
    headers = [cell.strip() for cell in re.split(r"\s{2,}", lines[separator - 1].strip())]
    items: list[dict[str, Any]] = []
    total: int | None = None
    for line in lines[separator + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("page "):
            match = re.search(r"total\s+(\d+)", stripped)
            if match:
                total = int(match.group(1))
            break
        if stripped.startswith("(no repositories"):
            break
        cells = [cell.strip() for cell in re.split(r"\s{2,}", stripped)]
        if len(cells) < 2:
            continue
        row = dict(zip(headers, cells))
        repo_id = row.get("repo_id")
        if not repo_id:
            continue
        items.append(
            {
                "repo_id": repo_id,
                "downloads": parse_int(row.get("downloads")),
                "likes": parse_int(row.get("likes")),
                "license": row.get("license") or None,
                "visibility": row.get("visibility") or None,
            }
        )
    return items, total


def hf_search_item(raw: dict[str, Any], source: str) -> dict[str, Any]:
    tags = [str(tag) for tag in (raw.get("tags") or []) if isinstance(tag, str)]
    license_name = next(
        (tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), None
    )
    params = None
    safetensors = raw.get("safetensors")
    if isinstance(safetensors, dict):
        params = parse_int(safetensors.get("total"))
    return {
        "source": source,
        "source_label": DOWNLOAD_SOURCES[source]["label"],
        "repo_id": str(raw.get("id") or ""),
        "pipeline_tag": raw.get("pipeline_tag"),
        "library": raw.get("library_name"),
        "downloads": parse_int(raw.get("downloads")),
        "likes": parse_int(raw.get("likes")),
        "license": license_name,
        "params": params,
        "updated_at": raw.get("last_modified") or raw.get("created_at"),
        "created_at": raw.get("created_at"),
        "tags": tags[:20],
    }


# ---------------------------------------------------------------------------
# per-model sampling / generation parameters

SAMPLING_GENERATION_PATHS = {"/v1/chat/completions", "/v1/completions"}

# Sampling defaults for a model added from the console: the values the two
# reference models run with (llama.cpp: LFM3-Omni-2B, KVMem: Ornith 1.5).
DEFAULT_SAMPLING = {
    "llama": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    "kvmem": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    "ninfer": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
}
# A llama.cpp preset that encodes the reference defaults (the [*] block plus
# the LFM3-2B-omni section); KVMem's reference config is Ornith 1.5.
LLAMA_REFERENCE_MODEL = "LFM3-2B-omni"
KVMEM_REFERENCE_MODEL = "ornith-1.5-35b-a3b-vl-thinking"
SMOKE_PROMPT = "Reply with the single word: OK"
SMOKE_IMAGE_PROMPT = "What shape is in this image? Answer in one word."


def _param_spec(
    key: str,
    label: str,
    group: str,
    kind: str = "number",
    *,
    low: float | int | None = None,
    high: float | int | None = None,
    step: float | int | None = None,
    low_exclusive: bool = False,
    choices: list[dict[str, Any]] | None = None,
    note: str | None = None,
    disabled: bool = False,
) -> dict[str, Any]:
    """One editable model parameter.  ``group`` drives the basic/more tabs."""
    spec: dict[str, Any] = {"key": key, "label": label, "group": group, "type": kind}
    if low is not None:
        spec["min"] = low
    if high is not None:
        spec["max"] = high
    if step is not None:
        spec["step"] = step
    if low_exclusive:
        spec["exclusive_min"] = True
    if choices:
        spec["choices"] = choices
    if note:
        spec["note"] = note
    if disabled:
        spec["disabled"] = True
    return spec


def _basic_specs(engine: str) -> list[dict[str, Any]]:
    kv = engine == "kvmem"
    ninfer = engine == "ninfer"
    return [
        _param_spec("temperature", "温度", "basic", low=0.0, high=2.0, step=0.01),
        _param_spec("top_p", "top_p", "basic", low=0.0, high=1.0, step=0.01, low_exclusive=True),
        _param_spec(
            "top_k", "top_k", "basic", kind="integer",
            low=1 if kv else 0, high=64 if kv else 1000, step=1,
        ),
        _param_spec("min_p", "min_p", "basic", low=0.0, high=1.0, step=0.01),
        _param_spec(
            "repetition_penalty", "重复惩罚", "basic", low=0.5, high=2.0, step=0.01,
            note="NInfer 只接受 1" if ninfer else None, disabled=ninfer,
        ),
        _param_spec(
            "repeat_last_n", "重复窗口", "basic",
            kind="integer", low=1 if kv else -1, high=131072, step=1,
            note="映射为 penalty_last_n" if kv else None,
            disabled=ninfer,
        ),
        _param_spec("presence_penalty", "存在惩罚", "basic", low=-2.0, high=2.0, step=0.01),
        _param_spec("frequency_penalty", "频率惩罚", "basic", low=-2.0, high=2.0, step=0.01),
    ]


def _llama_more_specs() -> list[dict[str, Any]]:
    """llama.cpp server sampling controls, minus per-request plumbing
    (stream / stop / grammar / logit_bias / response_fields / ...)."""
    return [
        _param_spec("seed", "随机种子", "more", kind="integer", low=-1, high=2147483647, step=1,
                    note="-1 表示随机"),
        _param_spec("n_predict", "最大生成 token", "more", kind="integer", low=-1, high=1048576,
                    step=1, note="请求中的 max_tokens 优先"),
        _param_spec("typical_p", "典型采样", "more", low=0.0, high=1.0, step=0.01, low_exclusive=True),
        _param_spec("dynatemp_range", "动态温度范围", "more", low=0.0, high=10.0, step=0.01),
        _param_spec("dynatemp_exponent", "动态温度指数", "more", low=0.0, high=10.0, step=0.01,
                    low_exclusive=True),
        _param_spec("xtc_probability", "XTC 概率", "more", low=0.0, high=1.0, step=0.01),
        _param_spec("xtc_threshold", "XTC 阈值", "more", low=0.0, high=1.0, step=0.01),
        _param_spec("dry_multiplier", "DRY 倍数", "more", low=0.0, high=100.0, step=0.01),
        _param_spec("dry_base", "DRY 基数", "more", low=1.0, high=100.0, step=0.01, low_exclusive=True),
        _param_spec("dry_allowed_length", "DRY 允许长度", "more", kind="integer", low=1, high=131072,
                    step=1),
        _param_spec("dry_penalty_last_n", "DRY 窗口", "more", kind="integer", low=-1, high=131072,
                    step=1, note="-1 表示上下文长度"),
        _param_spec(
            "mirostat", "Mirostat 模式", "more", kind="integer", low=0, high=2, step=1,
            choices=[
                {"value": 0, "label": "0 · 关闭"},
                {"value": 1, "label": "1 · Mirostat"},
                {"value": 2, "label": "2 · Mirostat 2.0"},
            ],
        ),
        _param_spec("mirostat_tau", "Mirostat τ", "more", low=0.0, high=100.0, step=0.01,
                    low_exclusive=True),
        _param_spec("mirostat_eta", "Mirostat η", "more", low=0.0, high=10.0, step=0.001,
                    low_exclusive=True),
        _param_spec("n_keep", "保留 token 数", "more", kind="integer", low=-1, high=1048576, step=1),
        _param_spec("min_keep", "最小候选数", "more", kind="integer", low=0, high=1000, step=1),
        _param_spec("n_probs", "返回候选概率数", "more", kind="integer", low=0, high=1000, step=1),
        _param_spec("ignore_eos", "忽略 EOS", "more", kind="boolean"),
        _param_spec("t_max_predict_ms", "生成时限（毫秒）", "more", kind="integer", low=0,
                    high=86400000, step=1, note="0 表示不限制"),
        _param_spec("samplers", "采样器顺序", "more", kind="list",
                    note="逗号分隔，例如 top_k,top_p,min_p,temperature"),
    ]


def sampling_schema(engine: str) -> list[dict[str, Any]]:
    """Editable parameters for one engine kind, split into basic and more."""
    engine = str(engine or "").strip().lower()
    if engine not in ("llama", "kvmem", "ninfer"):
        engine = "llama"
    if engine == "kvmem":
        more = [
            _param_spec("seed", "随机种子", "more", kind="integer", low=1, high=2147483647,
                        step=1, note="正整数"),
        ]
    elif engine == "ninfer":
        more = [
            _param_spec("seed", "随机种子", "more", kind="integer", low=0, high=2147483647, step=1),
        ]
    else:
        more = _llama_more_specs()
    return _basic_specs(engine) + more


def param_specs_by_key(engine: str) -> dict[str, dict[str, Any]]:
    return {spec["key"]: spec for spec in sampling_schema(engine)}


def sampling_aliases(key: str) -> tuple[str, ...]:
    if key == "repetition_penalty":
        return ("repetition_penalty", "repeat_penalty")
    if key == "repeat_last_n":
        return ("repeat_last_n", "penalty_last_n")
    if key == "n_predict":
        return ("n_predict", "max_tokens")
    return (key,)


def _sampling_number(key: str, raw: Any, low: float, high: float, *, integer: bool = False,
                     low_exclusive: bool = False) -> float | int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BadRequest(f"{key}: expected a number")
    if raw != raw or raw in (float("inf"), float("-inf")):
        raise BadRequest(f"{key}: expected a finite number")
    if integer:
        if raw != int(raw):
            raise BadRequest(f"{key}: expected an integer")
        raw = int(raw)
    ok = (raw > low and raw <= high) if low_exclusive else (low <= raw <= high)
    if not ok:
        if low_exclusive:
            raise BadRequest(f"{key}: expected {low} < value <= {high}")
        raise BadRequest(f"{key}: expected {low}..{high}")
    return float(raw) if not integer else raw


def _coerce_param(spec: dict[str, Any], raw: Any) -> Any:
    key = spec["key"]
    kind = spec["type"]
    if kind == "boolean":
        if isinstance(raw, bool):
            return raw
        if raw in (0, 1):
            return bool(raw)
        raise BadRequest(f"{key}: expected true or false")
    if kind == "list":
        if isinstance(raw, str):
            items = [part.strip() for part in re.split(r"[,，]", raw) if part.strip()]
        elif isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            items = [item.strip() for item in raw if item.strip()]
        else:
            raise BadRequest(f"{key}: expected a list of strings")
        if not items:
            return None
        if len(items) > 64:
            raise BadRequest(f"{key}: too many entries")
        return items
    value = _sampling_number(
        key, raw, spec.get("min", 0), spec.get("max", 0),
        integer=(kind == "integer"),
        low_exclusive=bool(spec.get("exclusive_min")),
    )
    choices = spec.get("choices")
    if choices and value not in [choice["value"] for choice in choices]:
        allowed = ", ".join(str(choice["value"]) for choice in choices)
        raise BadRequest(f"{key}: expected one of {allowed}")
    return value


def normalize_sampling(values: dict[str, Any], engine: str) -> dict[str, Any]:
    """Validate one model's parameter set.  None / '' clears a field; unknown fields are rejected."""
    if not isinstance(values, dict):
        raise BadRequest("params must be an object")
    specs = param_specs_by_key(engine)
    engine = str(engine or "").strip().lower()
    out: dict[str, Any] = {}
    for key, raw in values.items():
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        spec = specs.get(key)
        if spec is None:
            raise BadRequest(f"unknown sampling parameter {key!r}")
        if engine == "ninfer" and key == "repeat_last_n":
            raise BadRequest("NInfer 不支持 repeat_last_n（重复窗口）")
        value = _coerce_param(spec, raw)
        if value is None:
            continue
        if engine == "ninfer" and key == "repetition_penalty" and value != 1.0:
            raise BadRequest("NInfer 只接受中立值 repetition_penalty=1")
        out[key] = value
    return out


class SamplingStore:
    """One sampling set per model, persisted as model-params.json."""

    def __init__(self, path: Path, log: logging.Logger, registry: Registry):
        self.path = path
        self.log = log
        self.registry = registry
        self.lock = threading.RLock()
        self.models: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        data = read_json(self.path, {})
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, dict):
            if self.path.exists():
                self.log.warning("cannot read sampling parameters from %s; starting empty", self.path)
            models = {}
        cleaned: dict[str, dict[str, Any]] = {}
        for model_id, values in models.items():
            if not isinstance(model_id, str) or not isinstance(values, dict):
                continue
            entry = self.registry.resolve(model_id)
            engine = entry.engine if entry is not None else "llama"
            try:
                item = normalize_sampling(values, engine)
            except BadRequest as exc:
                self.log.warning("ignoring invalid sampling parameters for %s: %s", model_id, exc.message)
                continue
            if item:
                cleaned[model_id] = item
        self.models = cleaned

    def get(self, model_id: str) -> dict[str, Any]:
        with self.lock:
            return dict(self.models.get(model_id) or {})

    def all(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            return {key: dict(value) for key, value in self.models.items()}

    def replace(self, model_id: str, values: dict[str, Any], engine: str) -> dict[str, Any]:
        cleaned = normalize_sampling(values, engine)
        with self.lock:
            updated = {key: dict(value) for key, value in self.models.items()}
            if cleaned:
                updated[model_id] = cleaned
            else:
                updated.pop(model_id, None)
            write_json_atomic(self.path, {"version": 1, "models": updated})
            self.models = updated
        return dict(cleaned)

    def drop(self, model_id: str) -> None:
        with self.lock:
            if model_id not in self.models:
                return
            updated = {key: dict(value) for key, value in self.models.items() if key != model_id}
            write_json_atomic(self.path, {"version": 1, "models": updated})
            self.models = updated

    def apply(self, engine: str, model_id: str, payload: dict[str, Any]) -> bool:
        """Fill fields the request omitted; explicit request values win."""
        params = self.get(model_id)
        if not params:
            return False
        changed = False
        for key, value in params.items():
            if value is None:
                continue
            if any(payload.get(alias) is not None for alias in sampling_aliases(key)):
                continue
            if engine == "llama" and key == "repetition_penalty":
                target = "repeat_penalty"
            elif engine == "kvmem" and key == "repeat_last_n":
                target = "penalty_last_n"
            elif engine == "ninfer" and key in ("repetition_penalty", "repeat_last_n"):
                # NInfer has no repeat-window sampler and only accepts the
                # neutral repetition penalty; do not forward either field.
                continue
            else:
                target = key
            payload[target] = value
            changed = True
        return changed


def port_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def wait_port_free(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_free(host, port):
            return True
        time.sleep(0.5)
    return port_free(host, port)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cmdline_of(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def kill_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cleanup_stale_state(state_path: Path, log: logging.Logger) -> None:
    """A gateway that crashed can leave an engine behind; stop only a process that
    is recognisably one of ours."""
    data = read_json(state_path)
    if not isinstance(data, dict):
        return
    engine = data.get("engine")
    pid = parse_int(data.get("pid"))
    pgid = parse_int(data.get("pgid")) or pid
    if not pid or not pgid or not pid_alive(pid):
        try:
            state_path.unlink()
        except OSError:
            pass
        return
    cmd = cmdline_of(pid)
    expected = ("llama-server" in cmd) if engine == "llama" else ("serve.server" in cmd and "strata" in cmd)
    if not expected:
        log.warning("stale state names pid %s (%s), but its command line does not match; leaving it", pid, engine)
        try:
            state_path.unlink()
        except OSError:
            pass
        return
    log.warning("stopping stale %s engine pid=%s left by a previous gateway", engine, pid)
    kill_group(pgid, signal.SIGINT)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and pid_alive(pid):
        time.sleep(0.5)
    if pid_alive(pid):
        kill_group(pgid, signal.SIGTERM)
        time.sleep(3)
    if pid_alive(pid):
        kill_group(pgid, signal.SIGKILL)
        time.sleep(1)
    try:
        state_path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# models


@dataclass
class Model:
    id: str
    engine: str
    engine_model: str
    config_path: str | None = None
    display_name: str = ""
    aliases: list[str] = field(default_factory=list)
    context_length: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    max_output_tokens: int | None = None
    multimodal: bool = False
    video: bool = False
    audio: bool = False
    reasoning: bool = False
    notes: str | None = None
    quantization: str | None = None
    source: str = ""
    available: bool = True
    problem: str | None = None
    created: int = field(default_factory=lambda: int(now()))

    def openai(self, status: str, loaded: bool, params: dict[str, Any] | None = None) -> dict[str, Any]:
        value = status if status in ("loaded", "loading", "unloaded", "failed") else "unloaded"
        input_modalities = ["text"]
        if self.multimodal:
            input_modalities.append("image")
        if self.audio:
            input_modalities.append("audio")
        if self.video:
            input_modalities.append("video")
        capabilities: list[str] = []
        if self.multimodal:
            capabilities.append("vision")
        if self.reasoning:
            capabilities.append("reasoning")
        item: dict[str, Any] = {
            "id": self.id,
            "object": "model",
            "created": self.created,
            "owned_by": {
                "llama": "llama.cpp",
                "kvmem": "KVMem",
                "ninfer": "NInfer",
            }.get(self.engine, self.engine),
            # llama.cpp router UI reads model/name/status/architecture/meta/tags.
            "model": self.id,
            "name": self.display_name or self.id,
            "status": {"value": value, "args": []},
            "state": value,
            "architecture": {
                "input_modalities": input_modalities,
                "output_modalities": ["text"],
            },
            "meta": {"n_ctx_train": self.context_length} if self.context_length else {},
            "can_remove": False,
            "tags": [],
            "capabilities": capabilities,
            # gateway-specific additions (harmless to OpenAI clients)
            "engine": self.engine,
            "engine_model": self.engine_model,
            "display_name": self.display_name or self.id,
            "aliases": list(self.aliases),
            "context_length": self.context_length,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "multimodal": self.multimodal,
            "video": self.video,
            "audio": self.audio,
            "reasoning": self.reasoning,
            "notes": self.notes,
            "quantization": self.quantization,
            "source": self.source,
            "config_path": self.config_path,
            "deletable": self.source in ("preset", "strata", "ninfer"),
            "loaded": loaded,
            "available": self.available,
            "problem": self.problem,
            "params": dict(params or {}),
        }
        # OpenAI-compatible clients (e.g. dsh's "fetch models") read the output
        # capability from either of these top-level fields.
        if self.max_output_tokens:
            item["max_tokens"] = self.max_output_tokens
            item["max_output_tokens"] = self.max_output_tokens
        return item


def parse_llama_presets(path: Path) -> dict[str, dict[str, str]]:
    """models.ini has top-level keys before the first section; configparser rejects
    that, so read the small INI dialect directly."""
    out: dict[str, dict[str, str]] = {}
    current: str | None = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            out.setdefault(current, {})
            continue
        if current is not None and "=" in line:
            key, value = line.split("=", 1)
            out[current][key.strip()] = value.strip()
    return out


_GGUF_SCALAR_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


def _gguf_skip_value(fh: Any, value_type: int) -> None:
    if value_type == 8:  # string
        length = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(length, os.SEEK_CUR)
        return
    if value_type == 9:  # array
        elem_type = struct.unpack("<I", fh.read(4))[0]
        count = struct.unpack("<Q", fh.read(8))[0]
        if elem_type in (8, 9):
            for _ in range(count):
                _gguf_skip_value(fh, elem_type)
        else:
            size = _GGUF_SCALAR_SIZES.get(elem_type)
            if size is None:
                raise ValueError(f"unknown GGUF value type {elem_type}")
            fh.seek(size * count, os.SEEK_CUR)
        return
    size = _GGUF_SCALAR_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unknown GGUF value type {value_type}")
    fh.seek(size, os.SEEK_CUR)


def gguf_bool_metadata(path: Path, wanted: set[str]) -> dict[str, bool] | None:
    """Read top-level GGUF boolean metadata without loading tensors.

    llama.cpp's offline model scan uses clip.has_vision_encoder and
    clip.has_audio_encoder from the mmproj to report modalities; reading the
    same flags here keeps the router's pre-load capability list accurate.
    Returns None when the file is missing or cannot be parsed.
    """
    found: dict[str, bool] = {}
    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            struct.unpack("<I", fh.read(4))[0]  # version
            fh.read(8)  # tensor count
            n_kv = struct.unpack("<Q", fh.read(8))[0]
            for _ in range(n_kv):
                key_len = struct.unpack("<Q", fh.read(8))[0]
                key = fh.read(key_len).decode("utf-8", "replace")
                value_type = struct.unpack("<I", fh.read(4))[0]
                if key in wanted and value_type == 7:
                    found[key] = fh.read(1) != b"\x00"
                else:
                    _gguf_skip_value(fh, value_type)
    except (OSError, struct.error, ValueError):
        return None
    return found


def _gguf_read_value(fh: Any, value_type: int) -> Any:
    if value_type == 8:  # string
        length = struct.unpack("<Q", fh.read(8))[0]
        return fh.read(length).decode("utf-8", "replace")
    if value_type == 4:
        return struct.unpack("<I", fh.read(4))[0]
    if value_type == 5:
        return struct.unpack("<i", fh.read(4))[0]
    if value_type == 10:
        return struct.unpack("<Q", fh.read(8))[0]
    if value_type == 11:
        return struct.unpack("<q", fh.read(8))[0]
    _gguf_skip_value(fh, value_type)
    return None


def gguf_model_info(path: Path) -> dict[str, Any] | None:
    """Architecture, model name and maximum context length from a GGUF header."""
    info: dict[str, Any] = {}
    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            struct.unpack("<I", fh.read(4))[0]  # version
            fh.read(8)  # tensor count
            n_kv = struct.unpack("<Q", fh.read(8))[0]
            for _ in range(n_kv):
                key_len = struct.unpack("<Q", fh.read(8))[0]
                key = fh.read(key_len).decode("utf-8", "replace")
                value_type = struct.unpack("<I", fh.read(4))[0]
                lower = key.lower()
                if lower in ("general.architecture", "general.name"):
                    value = _gguf_read_value(fh, value_type)
                    if isinstance(value, str) and value:
                        info["architecture" if lower.endswith("architecture") else "name"] = value
                elif lower == "context_length" or lower.endswith(".context_length"):
                    value = _gguf_read_value(fh, value_type)
                    if isinstance(value, int) and value > 0:
                        info["context_length"] = value
                else:
                    _gguf_skip_value(fh, value_type)
    except (OSError, struct.error, ValueError):
        return None
    return info or None


def slugify_model_id(stem: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(stem)).strip("-._")
    return value or "model"


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def llama_preset_defaults(engine_cfg: dict[str, Any]) -> dict[str, str]:
    """The [*] block plus the reference model's non-structural keys."""
    preset = Path(str((engine_cfg.get("models") or {}).get("preset") or ""))
    presets = parse_llama_presets(preset)
    defaults = dict(presets.get("*", {}))
    for key, value in (presets.get(LLAMA_REFERENCE_MODEL) or {}).items():
        if key.strip().lower() in ("model", "mmproj", "c", "ctx-size", "ctx_size", "alias"):
            continue
        defaults[key] = value
    return defaults


def append_llama_preset(path: Path, name: str, values: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "version = 1\n"
    if text and not text.endswith("\n"):
        text += "\n"
    block = [f"\n[{name}]"]
    block.extend(f"{key} = {value}" for key, value in values.items())
    write_text_atomic(path, text + "\n".join(block) + "\n")


_PRESET_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "c": ("c", "ctx-size"),
    "cache-type-k": ("cache-type-k",),
    "cache-type-v": ("cache-type-v",),
}


def update_llama_preset(path: Path, name: str, values: dict[str, str]) -> None:
    """Update keys inside an existing preset section, keeping everything else intact."""
    wanted = {str(key): str(value) for key, value in values.items()}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        lines = []
    start: int | None = None
    end: int | None = None
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if start is not None:
                end = index
                break
            if stripped[1:-1].strip() == name:
                start = index
    if start is None:
        append_llama_preset(path, name, wanted)
        return
    if end is None:
        end = len(lines)
    missing = dict(wanted)
    seen: set[str] = set()
    section: list[str] = []
    for raw in lines[start + 1:end]:
        stripped = raw.strip()
        if stripped and not stripped.startswith((";", "#")) and "=" in stripped:
            current = stripped.split("=", 1)[0].strip().lower().replace("_", "-")
            matched = None
            for key in wanted:
                if current in _PRESET_KEY_ALIASES.get(key, (key,)):
                    matched = key
                    break
            if matched is not None:
                if matched in seen:
                    continue
                seen.add(matched)
                if matched in missing:
                    indent = raw[: len(raw) - len(raw.lstrip())]
                    section.append(f"{indent}{stripped.split('=', 1)[0].strip()} = {missing.pop(matched)}\n")
                continue
        section.append(raw)
    trailing: list[str] = []
    while section and not section[-1].strip():
        trailing.insert(0, section.pop())
    for key, value in missing.items():
        section.append(f"{key} = {value}\n")
    section.extend(trailing)
    write_text_atomic(path, "".join(lines[: start + 1] + section + lines[end:]))


def remove_llama_preset(path: Path, name: str) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        return False
    out: list[str] = []
    inside = False
    removed = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped[1:-1].strip() == name
            if inside:
                removed = True
                while out and not out[-1].strip():
                    out.pop()
                continue
        if inside:
            continue
        out.append(line)
    if removed:
        write_text_atomic(path, "".join(out))
    return removed


def _config_arg_replace(args: list[str], flag: str, value: Any) -> list[str]:
    out = [str(item) for item in args]
    if flag in out[:-1]:
        out[out.index(flag) + 1] = str(value)
    else:
        out.extend([flag, str(value)])
    return out


def _config_arg_remove(args: list[str], flag: str) -> list[str]:
    out: list[str] = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        text = str(item)
        if text == flag:
            skip = True
            continue
        if text.startswith(flag + "="):
            continue
        out.append(text)
    return out


def kvmem_reference_config(engine_cfg: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    """The Ornith 1.5 strata config used as the template for added models."""
    config_dir = Path(str((engine_cfg.get("models") or {}).get("dir") or ""))
    paths = sorted(config_dir.glob("strata-*.json"))
    ordered: list[Path] = []
    preferred = config_dir / "strata-ornith-1.5.json"
    if preferred in paths:
        ordered.append(preferred)
    for path in paths:
        if path not in ordered:
            ordered.append(path)
    fallback: tuple[Path, dict[str, Any]] | None = None
    for path in ordered:
        cfg = read_json(path)
        if not isinstance(cfg, dict) or not cfg.get("exe") or not isinstance(cfg.get("args"), list):
            continue
        if fallback is None:
            fallback = (path, cfg)
        if str(cfg.get("model_name") or "").lower() == KVMEM_REFERENCE_MODEL:
            return path, cfg
        if "ornith" in str(cfg.get("model_name") or path.name).lower():
            return path, cfg
    return fallback if fallback else (None, None)


def ninfer_reference_config(engine_cfg: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    """The per-model JSON whose args seed a newly added NInfer model."""
    models = engine_cfg.get("models") if isinstance(engine_cfg.get("models"), dict) else {}
    raw = str(models.get("template") or "").strip()
    candidates: list[Path] = []
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = Path(str(engine_cfg.get("cwd") or "")) / path
        candidates.append(path)
    else:
        config_dir = Path(str(models.get("dir") or ""))
        try:
            candidates = sorted(config_dir.glob("ninfer-*.json"))
        except OSError:
            candidates = []
    for path in candidates:
        cfg = read_json(path)
        if not isinstance(cfg, dict):
            continue
        if (cfg.get("model") or cfg.get("artifact")) and isinstance(cfg.get("args"), list):
            return path, cfg
    return None, None


def kvmem_tokenizer_tool(engine_cfg: dict[str, Any]) -> Path | None:
    """Locate tools/polystrata/assets.py, which exports a GGUF's own tokenizer."""
    cwd = Path(str(engine_cfg.get("cwd") or ""))
    for candidate in (
        cwd / "tools" / "polystrata" / "assets.py",
        cwd.parent / "tools" / "polystrata" / "assets.py",
        cwd.parent.parent / "tools" / "polystrata" / "assets.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def binary_help_text(binary: Path, cwd: Path | None = None, timeout: float = 20.0) -> str:
    try:
        proc = subprocess.run(
            [str(binary), "--help"],
            cwd=str(cwd) if cwd is not None and cwd.is_dir() else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def engine_help_text(engine_cfg: dict[str, Any], timeout: float = 20.0) -> str:
    command = engine_cfg.get("command")
    if not isinstance(command, list) or not command:
        return ""
    cwd = Path(str(engine_cfg.get("cwd") or "."))
    return binary_help_text(Path(str(command[0])), cwd, timeout)


# Used when the engine help cannot be read; mirrors llama.cpp's choice list.
LLAMA_CACHE_TYPE_FALLBACK = (
    "q8_0", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "f32",
)


def parse_cache_type_choices(help_text: str) -> list[str]:
    """Extract `--cache-type-k/v` allowed values from llama-server --help."""
    lines = str(help_text or "").splitlines()
    for flag in ("--cache-type-k", "--cache-type-v"):
        for index, line in enumerate(lines):
            if flag not in line or (flag + "-draft") in line:
                continue
            for follow in lines[index + 1:index + 4]:
                marker = follow.find("allowed values:")
                if marker < 0:
                    continue
                choices: list[str] = []
                for item in follow[marker + len("allowed values:"):].split(","):
                    name = item.strip().split()[0] if item.strip() else ""
                    if name and name not in choices:
                        choices.append(name)
                if choices:
                    if "q8_0" in choices:
                        choices = ["q8_0"] + [item for item in choices if item != "q8_0"]
                    return choices
    return list(LLAMA_CACHE_TYPE_FALLBACK)


def llama_cache_type_choices(engine_cfg: dict[str, Any]) -> list[str]:
    return parse_cache_type_choices(engine_help_text(engine_cfg))


def llama_reasoning_enabled(settings: dict[str, str]) -> bool:
    """A llama preset opts into thinking only when it configures reasoning
    parsing/budget or passes a thinking switch through the chat template.
    Merely shipping an enable_thinking template in the GGUF is not enough."""
    for key, raw in settings.items():
        name = key.strip().lower().replace("_", "-")
        value = str(raw).strip().lower()
        if name == "reasoning-format":
            if value not in ("", "none", "off", "false", "0"):
                return True
        elif name in ("reasoning-budget", "reasoning-budget-tokens", "reasoning-effort"):
            if value not in ("", "off", "false", "no"):
                return True
        elif "thinking" in name or "reasoning" in name:
            if value not in ("", "off", "false", "no", "0", "none"):
                return True
        elif name == "chat-template-kwargs":
            compact = value.replace("'", '"').replace(" ", "")
            if '"enable_thinking":true' in compact or '"reasoning_effort"' in compact:
                return True
    return False


def discover_llama_models(engine_cfg: dict[str, Any]) -> list[Model]:
    models_dir = Path(engine_cfg["models"]["dir"])
    preset_path = Path(engine_cfg["models"]["preset"])
    presets = parse_llama_presets(preset_path)
    base = presets.get("*", {})
    names: list[str] = []
    for section in presets:
        if section != "*" and section not in names:
            names.append(section)
    dirs: set[str] = set()
    try:
        for child in models_dir.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            if any(child.glob("*.gguf")):
                dirs.add(child.name)
                if child.name not in names:
                    names.append(child.name)
    except OSError:
        pass
    models: list[Model] = []
    for name in sorted(names):
        section = presets.get(name, {})
        merged = dict(base)
        merged.update(section)
        ctx = parse_int(merged.get("c") or merged.get("ctx-size") or merged.get("ctx_size"))
        cache_k = str(merged.get("cache-type-k") or merged.get("cache_type_k") or "").strip() or None
        cache_v = str(merged.get("cache-type-v") or merged.get("cache_type_v") or "").strip() or None
        max_out = parse_int(merged.get("n-predict") or merged.get("n_predict") or merged.get("max_tokens"))
        if max_out is not None and max_out <= 0:
            max_out = None
        alias_raw = merged.get("alias") or ""
        aliases = [x.strip() for x in alias_raw.split(",") if x.strip()]
        model_dir = models_dir / name
        has_dir = model_dir.is_dir()
        model_raw = (merged.get("model") or "").strip()
        model_file: Path | None = None
        if model_raw:
            model_file = Path(model_raw)
            if not model_file.is_absolute():
                model_file = models_dir / model_file
        has_model_file = bool(model_file is not None and model_file.is_file())
        has_gguf = has_model_file or (has_dir and any(model_dir.glob("*.gguf")))
        mmproj_files = sorted(model_dir.glob("mmproj*.gguf")) if has_dir else []
        mmproj_raw = (merged.get("mmproj") or "").strip()
        mmproj_path: Path | None = None
        if mmproj_raw:
            candidate = Path(mmproj_raw)
            if not candidate.is_absolute():
                for root in (model_dir, models_dir):
                    if (root / candidate).is_file():
                        candidate = root / candidate
                        break
            mmproj_path = candidate
        if mmproj_path is None and mmproj_files:
            mmproj_path = mmproj_files[0]
        clip_flags = (
            gguf_bool_metadata(
                mmproj_path,
                {"clip.has_vision_encoder", "clip.has_audio_encoder"},
            )
            if mmproj_path is not None
            else None
        )
        if clip_flags is not None:
            vision = bool(clip_flags.get("clip.has_vision_encoder"))
            audio = bool(clip_flags.get("clip.has_audio_encoder"))
        else:
            # Older or unreadable mmproj: assume the vision encoder (the common case).
            vision = bool(mmproj_files)
            audio = False
        problem = None
        available = has_gguf
        if not has_dir and model_file is None:
            problem = f"model directory is missing ({model_dir})"
        elif model_file is not None and not has_model_file:
            problem = f"model file is missing ({model_file})"
        elif not has_gguf:
            problem = f"no .gguf file in {model_dir}"
        models.append(
            Model(
                id=name,
                engine="llama",
                engine_model=name,
                config_path=str(preset_path) if section else None,
                display_name=name,
                aliases=aliases,
                context_length=ctx,
                cache_type_k=cache_k,
                cache_type_v=cache_v,
                max_output_tokens=max_out,
                multimodal=vision,
                # video comes from mtmd_helper_cap: vision AND MTMD_VIDEO, which is
                # compiled in here; offline preset discovery does not report it.
                video=vision,
                audio=audio,
                reasoning=llama_reasoning_enabled(merged),
                source="preset" if section else "models_dir",
                available=available,
                problem=problem,
            )
        )
    return models


def discover_kvmem_models(engine_cfg: dict[str, Any]) -> list[Model]:
    config_dir = Path(engine_cfg["models"]["dir"])
    meta_path = Path(engine_cfg["models"].get("meta") or config_dir / "models.json")
    extra = read_json(meta_path, {})
    if not isinstance(extra, dict):
        extra = {}
    candidates = []
    try:
        for path in config_dir.glob("strata-*.json"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
    except OSError:
        candidates = []
    candidates.sort(key=lambda item: item[0], reverse=True)
    models: list[Model] = []
    seen: set[str] = set()
    for _, path in candidates:
        cfg = read_json(path)
        if not isinstance(cfg, dict) or not cfg.get("exe") or not isinstance(cfg.get("args"), list):
            continue
        model_id = str(cfg.get("model_name") or path.stem.removeprefix("strata-"))
        if model_id in seen:
            continue
        seen.add(model_id)
        args = [str(x) for x in cfg["args"]]
        ctx = None
        if "--max-context" in args[:-1]:
            ctx = parse_int(args[args.index("--max-context") + 1])
        elif "--ctx" in args[:-1]:
            ctx = parse_int(args[args.index("--ctx") + 1])
        max_out = parse_int(cfg.get("kvmem_default_max_tokens"))
        if max_out is not None and max_out <= 0:
            max_out = None
        aliases_raw = cfg.get("aliases")
        if isinstance(aliases_raw, str):
            aliases = [x.strip() for x in aliases_raw.split(",") if x.strip()]
        elif isinstance(aliases_raw, list):
            aliases = [str(x).strip() for x in aliases_raw if str(x).strip()]
        else:
            aliases = []
        cwd = Path(cfg.get("cwd") or engine_cfg["cwd"])
        exe = Path(str(cfg.get("exe")))
        if not exe.is_absolute():
            exe = cwd / exe
        model_path = ""
        if "--model" in args[:-1]:
            model_path = args[args.index("--model") + 1]
        model_file = Path(model_path)
        if model_path and not model_file.is_absolute():
            model_file = cwd / model_file
        tokenizer = Path(str(cfg.get("tokenizer") or ""))
        problem = None
        if not exe.is_file() or not os.access(exe, os.X_OK):
            problem = f"engine is missing or not executable ({exe})"
        elif model_path and not model_file.is_file():
            problem = f"model file is missing ({model_file})"
        elif not tokenizer or not (tokenizer / "vocab.json").is_file():
            problem = f"tokenizer is missing ({tokenizer})"
        display_name = cfg.get("display_name") or model_id
        notes = None
        quantization = None
        more = extra.get(model_id)
        if isinstance(more, dict):
            display_name = more.get("display_name") or display_name
            notes = more.get("notes")
            quantization = more.get("quantization")
        reasoning = bool(
            cfg.get("preserve_thinking")
            or cfg.get("client_reasoning_history")
            or cfg.get("reasoning_budget_tokens")
        )
        vision_cfg = cfg.get("vision")
        video = bool(vision_cfg.get("video")) if isinstance(vision_cfg, dict) else False
        models.append(
            Model(
                id=model_id,
                engine="kvmem",
                engine_model=model_id,
                config_path=str(path),
                display_name=str(display_name),
                aliases=aliases,
                context_length=ctx,
                max_output_tokens=max_out,
                multimodal=bool(cfg.get("vision")),
                video=video,
                reasoning=reasoning,
                notes=notes,
                quantization=quantization,
                source="strata",
                available=problem is None,
                problem=problem,
            )
        )
    return models


def discover_ninfer_models(engine_cfg: dict[str, Any]) -> list[Model]:
    models_cfg = engine_cfg.get("models") if isinstance(engine_cfg.get("models"), dict) else {}
    config_dir = Path(str(models_cfg.get("dir") or ""))
    engine_cwd = Path(str(engine_cfg.get("cwd") or "."))
    engine_command = engine_cfg.get("command")
    engine_binary = str(engine_command[0]) if isinstance(engine_command, list) and engine_command else ""
    template_raw = str(models_cfg.get("template") or "").strip()
    template_path = None
    if template_raw:
        template = Path(template_raw)
        if not template.is_absolute():
            template = engine_cwd / template
        template_path = os.path.abspath(str(template))
    candidates = []
    try:
        for path in config_dir.glob("ninfer-*.json"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
    except OSError:
        candidates = []
    candidates.sort(key=lambda item: item[0], reverse=True)
    models: list[Model] = []
    seen: set[str] = set()
    for _, path in candidates:
        if template_path and os.path.abspath(str(path)) == template_path:
            continue
        cfg = read_json(path)
        if not isinstance(cfg, dict):
            continue
        args = cfg.get("args")
        if not isinstance(args, list):
            continue
        raw_model = str(cfg.get("model") or cfg.get("artifact") or "").strip()
        if not raw_model:
            continue
        model_id = str(cfg.get("model_name") or path.stem.removeprefix("ninfer-"))
        if model_id in seen:
            continue
        seen.add(model_id)
        args = [str(item) for item in args]
        ctx = parse_int(cfg.get("context_length"))
        if ctx is None and "--max-context" in args[:-1]:
            ctx = parse_int(args[args.index("--max-context") + 1])
        max_out = parse_int(cfg.get("default_max_tokens"))
        if max_out is None and "--default-max-tokens" in args[:-1]:
            max_out = parse_int(args[args.index("--default-max-tokens") + 1])
        if max_out is not None and max_out <= 0:
            max_out = None
        aliases_raw = cfg.get("aliases")
        if isinstance(aliases_raw, str):
            aliases = [x.strip() for x in aliases_raw.split(",") if x.strip()]
        elif isinstance(aliases_raw, list):
            aliases = [str(x).strip() for x in aliases_raw if str(x).strip()]
        else:
            aliases = []
        cwd_raw = Path(str(cfg.get("cwd") or engine_cwd))
        if not cwd_raw.is_absolute():
            cwd_raw = engine_cwd / cwd_raw
        exe_raw = str(cfg.get("exe") or engine_binary)
        exe = Path(exe_raw)
        if not exe.is_absolute():
            exe = cwd_raw / exe
        model_file = Path(raw_model)
        if not model_file.is_absolute():
            model_file = cwd_raw / model_file
        problem = None
        if not exe.is_file() or not os.access(exe, os.X_OK):
            problem = f"引擎可执行文件不存在或不可执行（{exe}）"
        elif not model_file.is_file():
            problem = f"模型文件不存在（{model_file}）"
        engine_model = str(cfg.get("model_id") or model_id)
        display_name = str(cfg.get("display_name") or model_id)
        models.append(
            Model(
                id=model_id,
                engine="ninfer",
                engine_model=engine_model,
                config_path=str(path),
                display_name=display_name,
                aliases=aliases,
                context_length=ctx,
                max_output_tokens=max_out,
                multimodal=bool(cfg.get("vision")),
                reasoning=bool(cfg.get("reasoning")),
                notes=cfg.get("notes"),
                quantization=cfg.get("quantization"),
                source="ninfer",
                available=problem is None,
                problem=problem,
            )
        )
    return models


class Registry:
    def __init__(self, config: dict[str, Any], base_dir: Path, log: logging.Logger):
        self.config = config
        self.base_dir = base_dir
        self.log = log
        self.lock = threading.RLock()
        self.models: list[Model] = []
        self.by_id: dict[str, Model] = {}
        self.by_alias: dict[str, Model | None] = {}
        self.loaded_at = 0.0
        self.reload()

    def reload(self) -> list[Model]:
        all_models: list[Model] = []
        for engine_id, engine_cfg in self.config.get("engines", {}).items():
            kind = engine_cfg.get("models", {}).get("kind")
            try:
                if kind == "llama_dir":
                    all_models.extend(discover_llama_models(engine_cfg))
                elif kind == "kvmem_dir":
                    all_models.extend(discover_kvmem_models(engine_cfg))
                elif kind == "ninfer_dir":
                    all_models.extend(discover_ninfer_models(engine_cfg))
            except Exception:
                self.log.exception("cannot scan models for engine %s", engine_id)
        counts: dict[str, int] = {}
        for model in all_models:
            counts[model.id] = counts.get(model.id, 0) + 1
        for model in all_models:
            if counts[model.id] > 1:
                # Keep both usable under deterministic names.  The original name
                # stays as an alias for the first one encountered.
                if model.id not in model.aliases:
                    model.aliases.append(model.id)
                model.id = f"{model.engine}/{model.id}"
        by_id = {m.id: m for m in all_models}
        by_alias: dict[str, Model | None] = {}
        for model in all_models:
            for name in [model.id, model.display_name, *model.aliases]:
                if not name:
                    continue
                key = name.lower()
                if key in by_alias and by_alias[key] is not model:
                    by_alias[key] = None
                else:
                    by_alias[key] = model
        with self.lock:
            self.models = all_models
            self.by_id = by_id
            self.by_alias = by_alias
            self.loaded_at = now()
        self.log.info(
            "model registry: %d models (%s)",
            len(all_models),
            ", ".join(f"{k}={v}" for k, v in sorted(self._counts().items())),
        )
        for model in all_models:
            if not model.available:
                self.log.warning("model %s is unavailable: %s", model.id, model.problem)
        return all_models

    def _counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for model in self.models:
            out[model.engine] = out.get(model.engine, 0) + 1
        return out

    def list(self) -> list[Model]:
        with self.lock:
            return list(self.models)

    def resolve(self, name: str) -> Model | None:
        if not name:
            return None
        with self.lock:
            if name in self.by_id:
                return self.by_id[name]
            return self.by_alias.get(name.lower())

    def counts(self) -> dict[str, int]:
        with self.lock:
            return self._counts()


# ---------------------------------------------------------------------------
# SSE status hub
#
# The llama.cpp UI expects the router's /models/sse stream: frames such as
# {"model": "...", "event": "model_status", "data": {"status": "loaded"}}.
# We poll the gateway state and fan the changes out to every subscriber.


class StatusHub:
    def __init__(
        self,
        model_source: Any,
        reload_stamp: Any,
        log: logging.Logger,
        interval: float = 0.5,
    ):
        self.model_source = model_source
        self.reload_stamp = reload_stamp
        self.log = log
        self.interval = interval
        self.lock = threading.Lock()
        self.subscribers: set[queue.Queue] = set()
        self._last: dict[str, str] = {}
        self._last_reload = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="status-hub", daemon=True)
        self._thread.start()

    @staticmethod
    def frame(model_id: str, status: str) -> dict[str, Any]:
        return {"model": model_id, "event": "model_status", "data": {"status": status}}

    def _offer(self, target: queue.Queue, payload: Any) -> None:
        try:
            target.put_nowait(payload)
        except queue.Full:
            # A slow reader must not stall the hub or other clients.
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(payload)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=512)
        try:
            snapshot = self.model_source()
        except Exception:
            snapshot = {"data": []}
        with self.lock:
            self.subscribers.add(q)
            self._offer(q, {"event": "models_reload", "data": {}})
            for item in snapshot.get("data") or []:
                model_id = str(item.get("id"))
                status = ((item.get("status") or {}).get("value")) or "unloaded"
                # If the poller already saw a newer state than the snapshot above,
                # prefer it; otherwise the next poll will correct the snapshot.
                self._offer(q, self.frame(model_id, str(self._last.get(model_id, status))))
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.subscribers.discard(q)
        self._offer(q, None)

    def broadcast(self, payload: Any) -> None:
        with self.lock:
            targets = list(self.subscribers)
        for target in targets:
            self._offer(target, payload)

    def publish(self, model_id: str, status: str) -> None:
        with self.lock:
            if self._last.get(model_id) == status:
                return
            self._last[model_id] = status
        self.broadcast(self.frame(model_id, status))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                snapshot = self.model_source()
            except Exception:
                continue
            try:
                stamp = float(self.reload_stamp())
            except Exception:
                stamp = 0.0
            if stamp != self._last_reload:
                if self._last_reload:
                    self.broadcast({"event": "models_reload", "data": {}})
                self._last_reload = stamp
            seen: set[str] = set()
            for item in snapshot.get("data") or []:
                model_id = str(item.get("id") or "")
                if not model_id:
                    continue
                seen.add(model_id)
                status = str(((item.get("status") or {}).get("value")) or "unloaded")
                if self._last.get(model_id) != status:
                    self._last[model_id] = status
                    self.broadcast(self.frame(model_id, status))
            for gone in set(self._last) - seen:
                self._last.pop(gone, None)

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        with self.lock:
            targets = list(self.subscribers)
            self.subscribers.clear()
        for target in targets:
            self._offer(target, None)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# fixed engine slots and path configuration

ENGINE_SLOTS: dict[str, dict[str, Any]] = {
    "llama": {
        "label": "llama.cpp",
        "kind": "llama_router",
        "models_kind": "llama_dir",
        "fields": ("binary", "cwd", "models_dir", "preset"),
        "model_key": "preset",
        "host": "127.0.0.1",
        "port": 18080,
        "start_timeout_s": 120,
        "probe_timeout_s": 15,
        "log_path": "logs/llama-engine.log",
    },
    "kvmem": {
        "label": "KVMem（polystrata）",
        "kind": "kvmem",
        "models_kind": "kvmem_dir",
        "fields": ("binary", "cwd", "models_dir", "meta"),
        "model_key": "meta",
        "host": "127.0.0.1",
        "port": 18081,
        "start_timeout_s": 900,
        "probe_timeout_s": 30,
        "log_path": "logs/kvmem-engine.log",
    },
    "ninfer": {
        "label": "NInfer（KVMem）",
        "kind": "ninfer",
        "models_kind": "ninfer_dir",
        "fields": ("binary", "cwd", "models_dir", "template"),
        "model_key": "template",
        "host": "127.0.0.1",
        "port": 18082,
        "start_timeout_s": 900,
        "probe_timeout_s": 30,
        "log_path": "logs/ninfer-engine.log",
    },
}
ENGINE_SLOT_ORDER = ("llama", "kvmem", "ninfer")
KV_PROBE_CODE = (
    "import serve.server\n"
    "print('LLM_ROUTER_KVPROBE_' + str(6 * 7))\n"
)
KV_PROBE_MARKER = "LLM_ROUTER_KVPROBE_42"


def engine_slot_defaults(base_dir: Path, engine_id: str) -> dict[str, str]:
    """Conventional sibling-directory layout, used to prefill a new engine."""
    root = base_dir.parent
    if engine_id == "llama":
        cwd = root / "llama.cpp"
        return {
            "binary": str(cwd / "build" / "bin" / "llama-server"),
            "cwd": str(cwd),
            "models_dir": str(base_dir / "models"),
            "preset": str(base_dir / "models.ini"),
        }
    if engine_id == "kvmem":
        home = root / "kvmem"
        models_dir = home / "local_config"
        return {
            "binary": str(home / ".venv" / "bin" / "python"),
            "cwd": str(home / "backends" / "polystrata"),
            "models_dir": str(models_dir),
            "meta": str(models_dir / "models.json"),
        }
    if engine_id == "ninfer":
        home = root / "kvmem"
        models_dir = home / "local_config"
        return {
            "binary": str(root / "build-ninfer" / "apps" / "ninfer-serve"),
            "cwd": str(home),
            "models_dir": str(models_dir),
            "template": str(models_dir / "ninfer.template.json"),
        }
    return {}


def engine_path_fields(engine_id: str, engine_cfg: dict[str, Any]) -> dict[str, str]:
    spec = ENGINE_SLOTS.get(engine_id) or {}
    models = engine_cfg.get("models")
    models = models if isinstance(models, dict) else {}
    command = engine_cfg.get("command")
    command = command if isinstance(command, list) else []
    fields = {
        "binary": str(command[0]) if command else "",
        "cwd": str(engine_cfg.get("cwd") or ""),
        "models_dir": str(models.get("dir") or ""),
    }
    model_key = str(spec.get("model_key") or "")
    if model_key:
        fields[model_key] = str(models.get(model_key) or "")
    return fields


def _slot_command(engine_id: str, fields: dict[str, str]) -> list[str]:
    if engine_id == "llama":
        return [
            fields["binary"],
            "--models-preset", fields["preset"],
            "--models-dir", fields["models_dir"],
            "--host", "{host}",
            "--port", "{port}",
        ]
    if engine_id == "ninfer":
        # The model artifact and its serve options come from the per-model
        # ninfer-*.json; the manager inserts them after the executable.
        return [
            fields["binary"],
            "--host", "{host}",
            "--port", "{port}",
        ]
    return [
        fields["binary"],
        "-m", "serve.server",
        "--engine", "strata",
        "--config", "{config}",
        "--host", "{host}",
        "--port", "{port}",
        "--switch-on-request",
    ]


def _rewrite_slot_command(
    command: list[str], fields: dict[str, str], old: dict[str, str]
) -> list[str]:
    """Keep custom flags but move every known path to the new value."""
    replacements = []
    for key in ("binary", "models_dir", "preset", "meta", "template", "cwd"):
        before = str(old.get(key) or "")
        after = str(fields.get(key) or "")
        if before and after and before != after:
            replacements.append((before, after))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    out = []
    for token in command:
        value = str(token)
        for before, after in replacements:
            value = value.replace(before, after)
        out.append(value)
    binary = fields.get("binary") or (out[0] if out else "")
    if out and binary and out[0] != binary:
        out[0] = binary
    return out


def build_engine_cfg(
    engine_id: str, fields: dict[str, str], existing: dict[str, Any] | None
) -> dict[str, Any]:
    spec = ENGINE_SLOTS[engine_id]
    cfg = copy.deepcopy(existing) if isinstance(existing, dict) else {}
    old = engine_path_fields(engine_id, cfg) if cfg else {}
    cfg["kind"] = str(cfg.get("kind") or spec["kind"])
    cfg["host"] = str(cfg.get("host") or spec["host"])
    cfg["port"] = int(cfg.get("port") or spec["port"])
    cfg["cwd"] = fields["cwd"]
    command = cfg.get("command")
    if isinstance(command, list) and command:
        cfg["command"] = _rewrite_slot_command(command, fields, old)
    else:
        cfg["command"] = _slot_command(engine_id, fields)
    cfg.setdefault("start_timeout_s", spec["start_timeout_s"])
    cfg.setdefault("log_path", spec["log_path"])
    if existing is None:
        cfg.setdefault("api_key", None)
    if engine_id == "ninfer" and not cfg.get("env"):
        # NInfer builds keep their shared libraries beside the executable.
        build_root = Path(fields["binary"]).parent.parent
        lib_dir = build_root / "lib"
        if lib_dir.is_dir():
            cfg["env"] = {"LD_LIBRARY_PATH": f"{build_root}:{lib_dir}"}
    models = cfg.get("models")
    models = models if isinstance(models, dict) else {}
    models["kind"] = str(models.get("kind") or spec["models_kind"])
    models["dir"] = fields["models_dir"]
    model_key = str(spec["model_key"])
    models[model_key] = fields.get(model_key, "")
    cfg["models"] = models
    return cfg


def resolve_engine_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    # Keep symlinks intact: a venv's bin/python must stay inside the venv.
    return Path(os.path.abspath(str(path)))


def resolve_binary_path(base_dir: Path, value: str) -> str:
    """Allow either a path or a bare executable name resolved from PATH."""
    text = value.strip()
    has_sep = os.sep in text or (os.altsep is not None and os.altsep in text)
    if has_sep or text.startswith("~"):
        return str(resolve_engine_path(base_dir, text))
    found = shutil.which(text)
    if not found:
        raise BadRequest(f"binary：在 PATH 中找不到 {text}")
    return str(Path(os.path.abspath(found)))


def validate_engine_paths(
    engine_id: str, paths: dict[str, Any], base_dir: Path, create: bool = False
) -> tuple[dict[str, str], list[tuple[str, Path]]]:
    spec = ENGINE_SLOTS[engine_id]
    if not isinstance(paths, dict):
        raise BadRequest("paths 必须是对象")
    allowed = set(spec["fields"])
    unknown = sorted(str(key) for key in paths if key not in allowed)
    if unknown:
        raise BadRequest(f"不支持的路径字段：{', '.join(unknown)}")
    fields: dict[str, str] = {}
    for key in spec["fields"]:
        raw = paths.get(key, "")
        if raw is None:
            raw = ""
        if not isinstance(raw, str):
            raise BadRequest(f"{key} 必须是字符串")
        text = raw.strip()
        if not text:
            if key == "meta":
                fields[key] = ""
                continue
            raise BadRequest(f"{key} 不能为空")
        if key == "binary":
            fields[key] = resolve_binary_path(base_dir, text)
        else:
            fields[key] = str(resolve_engine_path(base_dir, text))
    for key, message in (
        ("binary", "文件不存在"),
        ("cwd", "目录不存在"),
    ):
        path = Path(fields[key])
        ok = path.is_file() if key == "binary" else path.is_dir()
        if not ok:
            raise BadRequest(f"{key}：{message}（{path}）")
    binary = Path(fields["binary"])
    if not os.access(binary, os.X_OK):
        raise BadRequest(f"binary：文件不可执行（{binary}）")

    # Data paths may legitimately be absent when creating a fresh slot. They
    # are not created here: the caller probes the binary first, then calls
    # materialize_engine_paths(), so a failed probe leaves no stray files.
    pending: list[tuple[str, Path]] = []
    models_dir = Path(fields["models_dir"])
    if models_dir.is_dir():
        pass
    elif models_dir.exists():
        raise BadRequest(f"models_dir：不是目录（{models_dir}）")
    elif create:
        pending.append(("models_dir", models_dir))
    else:
        raise BadRequest(f"models_dir：目录不存在（{models_dir}）")

    model_key = str(spec["model_key"])
    preset_value = fields.get(model_key) or ""
    if preset_value:
        preset = Path(preset_value)
        if preset.is_file():
            pass
        elif preset.exists():
            raise BadRequest(f"{model_key}：不是文件（{preset}）")
        elif create and engine_id == "llama":
            pending.append(("llama_preset", preset))
        elif create and engine_id == "kvmem":
            pending.append(("kvmem_meta", preset))
        else:
            raise BadRequest(f"{model_key}：文件不存在（{preset}）")
    return fields, pending


def materialize_engine_paths(pending: list[tuple[str, Path]]) -> list[str]:
    """Create the empty data paths offered for a brand-new engine slot."""
    notes: list[str] = []
    for kind, path in pending:
        try:
            if kind == "models_dir":
                if path.is_dir():
                    continue
                path.mkdir(parents=True, exist_ok=True)
                notes.append(f"已创建模型目录 {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                handle = path.open("x", encoding="utf-8")
            except FileExistsError:
                continue
            with handle:
                handle.write("version = 1\n" if kind == "llama_preset" else "{}\n")
            notes.append(
                f"已创建空预设 {path}" if kind == "llama_preset" else f"已创建空元数据 {path}"
            )
        except OSError as exc:
            raise BadRequest(f"无法创建 {path}：{exc}") from exc
    return notes


def _probe_tail(text: str, limit: int = 600) -> str:
    compact = " ".join(text.split())
    return compact[-limit:]


def probe_engine_binary(
    engine_id: str,
    fields: dict[str, str],
    engine_cfg: dict[str, Any] | None = None,
) -> None:
    """Cheaply verify the chosen executable is the right program.

    llama.cpp is asked for its version; KVMem's interpreter must import
    serve.server from the working directory; NInfer must identify itself as
    ninfer-serve. None of the probes starts a server.
    """
    spec = ENGINE_SLOTS[engine_id]
    binary = fields["binary"]
    if engine_id == "kvmem":
        command = [binary, "-c", KV_PROBE_CODE]
    else:
        command = [binary, "--version"]
    env = os.environ.copy()
    extra = (engine_cfg or {}).get("env")
    if isinstance(extra, dict):
        env.update({str(key): str(value) for key, value in extra.items()})
    env.setdefault("PYTHONUNBUFFERED", "1")
    timeout = float(spec.get("probe_timeout_s") or 15)
    try:
        proc = subprocess.run(
            command,
            cwd=fields["cwd"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BadRequest(f"binary：探测超时（{int(timeout)} 秒，{binary}）") from exc
    except OSError as exc:
        raise BadRequest(f"binary：无法执行探测（{exc}）") from exc
    stdout = proc.stdout.decode("utf-8", "replace") if proc.stdout else ""
    stderr = proc.stderr.decode("utf-8", "replace") if proc.stderr else ""
    output = (stdout + "\n" + stderr).strip()
    if engine_id == "kvmem":
        if proc.returncode != 0 or KV_PROBE_MARKER not in stdout:
            detail = _probe_tail(output)
            message = "binary：无法导入 serve.server"
            if proc.returncode:
                message += f"（退出码 {proc.returncode}）"
            if detail:
                message += f"：{detail}"
            raise BadRequest(message)
        return
    if engine_id == "ninfer":
        if proc.returncode != 0 or not re.search(r"ninfer-serve\s+\S+", output, re.I):
            detail = _probe_tail(output)
            message = "binary：不是可用的 ninfer-serve"
            if proc.returncode:
                message += f"（退出码 {proc.returncode}）"
            if detail:
                message += f"：{detail}"
            raise BadRequest(message)
        return
    if proc.returncode != 0 or not re.search(r"version\s*:\s*\S+", output, re.I):
        detail = _probe_tail(output)
        message = "binary：不是可用的 llama-server"
        if proc.returncode:
            message += f"（退出码 {proc.returncode}）"
        if detail:
            message += f"：{detail}"
        raise BadRequest(message)


def scan_engine_models(engine_id: str, cfg: dict[str, Any]) -> list[Model]:
    try:
        if engine_id == "llama":
            return discover_llama_models(cfg)
        if engine_id == "ninfer":
            return discover_ninfer_models(cfg)
        return discover_kvmem_models(cfg)
    except Exception as exc:
        raise BadRequest(f"无法扫描模型目录：{exc}") from exc


# ---------------------------------------------------------------------------
# engine supervisor


class EngineManager:
    def __init__(
        self,
        engine_id: str,
        cfg: dict[str, Any],
        base_dir: Path,
        timeouts: dict[str, Any],
        state_path: Path,
        log: logging.Logger,
    ):
        self.engine_id = engine_id
        self.cfg = cfg
        self.base_dir = base_dir
        self.timeouts = timeouts
        self.state_path = state_path
        self.log = log
        self.host = str(cfg.get("host") or "127.0.0.1")
        self.port = int(cfg.get("port") or 0)
        self.kind = str(cfg.get("kind") or "")
        self.proc: subprocess.Popen | None = None
        self.pgid: int | None = None
        self.log_path = self._resolve_path(str(cfg.get("log_path") or f"logs/{engine_id}-engine.log"))
        self.log_fh = None
        self.current_model: str | None = None
        self.current_entry: Model | None = None
        self.started_at: float | None = None
        self.started_paths: dict[str, str] | None = None
        self.last_pid: int | None = None

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.base_dir / path

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def _state(self, model: str | None) -> dict[str, Any]:
        return {
            "engine": self.engine_id,
            "pid": self.proc.pid if self.proc else None,
            "pgid": self.pgid,
            "host": self.host,
            "port": self.port,
            "model": model,
            "log_path": str(self.log_path),
            "started_at": self.started_at,
            "updated_at": iso_now(),
        }

    def _write_state(self) -> None:
        try:
            write_json_atomic(self.state_path, self._state(self.current_model))
        except OSError:
            self.log.exception("cannot write state file %s", self.state_path)

    def _clear_state(self) -> None:
        try:
            data = read_json(self.state_path)
            if not isinstance(data, dict) or data.get("engine") == self.engine_id:
                self.state_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _headers(self, entry: Model | None = None, json_body: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        # Engines are loopback-only; auth for clients is enforced by the gateway.
        # An optional engine-level key is still supported for upstream compatibility.
        key = first_key(self.cfg.get("api_key"))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        entry: Model | None = None,
        json_body: Any = None,
        timeout: float = 10.0,
    ) -> tuple[int, bytes, dict[str, str]]:
        headers = self._headers(entry, json_body is not None)
        session = requests.Session()
        session.trust_env = False
        try:
            resp = session.request(
                method,
                self._url(path),
                headers=headers,
                json=json_body,
                timeout=timeout,
                allow_redirects=False,
            )
            return resp.status_code, resp.content, dict(resp.headers)
        finally:
            session.close()

    def _json_request(
        self,
        method: str,
        path: str,
        *,
        entry: Model | None = None,
        json_body: Any = None,
        timeout: float = 10.0,
    ) -> tuple[int, Any]:
        status, body, _ = self._request(method, path, entry=entry, json_body=json_body, timeout=timeout)
        try:
            return status, json.loads(body) if body else None
        except ValueError:
            return status, body.decode("utf-8", "replace")

    def _error_from_body(self, status: int, body: Any) -> str:
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)
            if err:
                return str(err)
            return str(body)
        return str(body)[:1000] or f"HTTP {status}"

    def _rotate_log(self) -> None:
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > 20 * 1024 * 1024:
                backup = self.log_path.with_name(self.log_path.name + ".1")
                backup.unlink(missing_ok=True)
                os.replace(self.log_path, backup)
        except OSError:
            pass

    def _ninfer_model_config(self, entry: Model) -> tuple[dict[str, Any], str, list[str]]:
        if not entry.config_path:
            raise EngineFailure(f"{self.engine_id}: {entry.id} has no config path")
        cfg = read_json(Path(entry.config_path))
        if not isinstance(cfg, dict):
            raise EngineFailure(f"{self.engine_id}: cannot read {entry.config_path}")
        artifact = str(cfg.get("model") or cfg.get("artifact") or "").strip()
        if not artifact:
            raise EngineFailure(f"{self.engine_id}: {entry.id} config has no model path")
        args = cfg.get("args")
        args = [str(item) for item in args] if isinstance(args, list) else []
        return cfg, artifact, args

    def _build_command(self, entry: Model) -> list[str]:
        command = self.cfg.get("command")
        if not isinstance(command, list) or not command:
            raise EngineFailure(f"{self.engine_id}: no command in config")
        if self.kind == "kvmem" and not entry.config_path:
            raise EngineFailure(f"{self.engine_id}: {entry.id} has no config path")
        if self.kind == "ninfer":
            model_cfg, artifact, args = self._ninfer_model_config(entry)
            binary = str(model_cfg.get("exe") or command[0])
            # ninfer-serve takes the artifact path first, then its serve
            # options; host/port stay from the engine slot command.
            command = [binary, artifact, *args, *[str(token) for token in command[1:]]]
        out = []
        for token in command:
            value = str(token)
            value = value.replace("{config}", entry.config_path or "")
            value = value.replace("{host}", self.host)
            value = value.replace("{port}", str(self.port))
            value = value.replace("{cwd}", str(self.cfg.get("cwd") or self.base_dir))
            out.append(value)
        return out

    def _environment(self, entry: Model) -> dict[str, str]:
        env = os.environ.copy()
        extra = self.cfg.get("env")
        if isinstance(extra, dict):
            env.update({str(k): str(v) for k, v in extra.items()})
        if self.kind in ("kvmem", "ninfer") and entry.config_path:
            cfg = read_json(Path(entry.config_path))
            if isinstance(cfg, dict) and isinstance(cfg.get("env"), dict):
                env.update({str(k): str(v) for k, v in cfg["env"].items()})
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def start(self, entry: Model) -> None:
        if self.alive():
            raise EngineFailure(f"{self.engine_id}: engine is already running")
        if not port_free(self.host, self.port):
            raise EngineFailure(
                f"{self.engine_id}: internal port {self.host}:{self.port} is already in use; "
                "stop the old service or choose another port"
            )
        cmd = self._build_command(entry)
        started_paths = engine_path_fields(self.engine_id, self.cfg)
        cwd = Path(str(self.cfg.get("cwd") or self.base_dir))
        if not cwd.is_dir():
            raise EngineFailure(f"{self.engine_id}: working directory is missing ({cwd})")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_log()
        self.log_fh = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.log_fh.write(f"\n===== {iso_now()} starting {self.engine_id} for {entry.id} =====\n")
        self.log_fh.write("command: " + " ".join(cmd) + "\n")
        self.log_fh.flush()
        self.log.info("starting engine %s for %s on %s:%s", self.engine_id, entry.id, self.host, self.port)
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=self._environment(entry),
                stdout=self.log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            self._close_log()
            raise EngineFailure(f"{self.engine_id}: cannot start engine: {exc}") from exc
        try:
            self.pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            self.pgid = self.proc.pid
        self.last_pid = self.proc.pid
        self.started_at = now()
        self.current_model = entry.engine_model if self.kind in ("kvmem", "ninfer") else None
        self._write_state()
        try:
            if self.kind == "llama_router":
                self._wait_llama_ready()
            elif self.kind == "ninfer":
                self._wait_ninfer_ready(entry)
            else:
                self._wait_kvmem_ready(entry)
        except Exception:
            self.log.error("engine %s failed to start; log tail:\n%s", self.engine_id, tail_file(self.log_path, 60))
            try:
                self.stop()
            except Exception:
                self.log.exception("cannot clean up failed engine %s", self.engine_id)
            raise
        self.current_model = entry.engine_model
        self.current_entry = entry
        self.started_paths = started_paths
        self._write_state()
        self.log.info("engine %s is ready (%s)", self.engine_id, entry.id)

    def _wait_llama_ready(self) -> None:
        deadline = time.monotonic() + float(self.cfg.get("start_timeout_s") or 120)
        last_note = 0.0
        while time.monotonic() < deadline:
            if not self.alive():
                raise EngineFailure(f"llama engine exited during startup; log: {tail_file(self.log_path, 40)}")
            try:
                status, body, _ = self._request("GET", "/health", timeout=3)
                if status == 200:
                    status2, _, _ = self._request("GET", "/models", timeout=3)
                    if status2 == 200:
                        return
            except requests.RequestException:
                pass
            if time.monotonic() - last_note > 10:
                last_note = time.monotonic()
                self.log.info("waiting for llama router on %s:%s", self.host, self.port)
            time.sleep(0.5)
        raise EngineFailure(f"llama engine did not become ready within {self.cfg.get('start_timeout_s')}s")

    def _wait_kvmem_ready(self, entry: Model) -> None:
        timeout = float(self.cfg.get("start_timeout_s") or self.timeouts.get("load_timeout_s") or 900)
        deadline = time.monotonic() + timeout
        last_note = 0.0
        last_error = ""
        while time.monotonic() < deadline:
            if not self.alive():
                raise EngineFailure(f"KVMem engine exited during startup; log: {tail_file(self.log_path, 40)}")
            try:
                status, body = self._json_request("GET", "/api/models", entry=entry, timeout=4)
                if status == 200 and isinstance(body, dict):
                    models = body.get("models") or []
                    current = body.get("current")
                    target = next((m for m in models if m.get("id") == entry.engine_model), None)
                    if target is not None and target.get("status") == "unavailable":
                        raise EngineFailure(
                            f"KVMem reports {entry.id} unavailable: {target.get('problem') or 'unknown reason'}"
                        )
                    if current == entry.engine_model and target is not None and target.get("status") == "loaded":
                        return
                elif status in (401, 403):
                    raise EngineFailure(
                        f"KVMem rejected the configured API key (HTTP {status}); check engines.kvmem.api_key"
                    )
            except requests.RequestException as exc:
                last_error = str(exc)
            if time.monotonic() - last_note > 15:
                last_note = time.monotonic()
                self.log.info(
                    "waiting for KVMem model %s on %s:%s%s",
                    entry.id,
                    self.host,
                    self.port,
                    f" ({last_error})" if last_error else "",
                )
            time.sleep(1.0)
        raise EngineFailure(
            f"KVMem model {entry.id} did not become ready within {int(timeout)}s; "
            f"log: {tail_file(self.log_path, 40)}"
        )

    def _wait_ninfer_ready(self, entry: Model) -> None:
        timeout = float(self.cfg.get("start_timeout_s") or self.timeouts.get("load_timeout_s") or 900)
        deadline = time.monotonic() + timeout
        last_note = 0.0
        last_error = ""
        while time.monotonic() < deadline:
            if not self.alive():
                raise EngineFailure(f"NInfer engine exited during startup; log: {tail_file(self.log_path, 40)}")
            try:
                status, body = self._json_request("GET", "/health", entry=entry, timeout=4)
                if status == 200:
                    status2, body2 = self._json_request("GET", "/v1/models", entry=entry, timeout=5)
                    if status2 == 200 and isinstance(body2, dict):
                        data = body2.get("data")
                        data = data if isinstance(data, list) else []
                        target = next((m for m in data if m.get("id") == entry.engine_model), None)
                        if target is not None and self._llama_status(target) in ("loaded", "sleeping"):
                            return
                        if data:
                            last_error = "models=" + ", ".join(str(m.get("id")) for m in data[:5])
                elif status in (401, 403):
                    raise EngineFailure(
                        f"NInfer rejected the configured API key (HTTP {status}); check engines.ninfer.api_key"
                    )
                elif isinstance(body, dict):
                    err = body.get("error")
                    if isinstance(err, dict):
                        last_error = str(err.get("message") or err)
            except requests.RequestException as exc:
                last_error = str(exc)
            if time.monotonic() - last_note > 15:
                last_note = time.monotonic()
                self.log.info(
                    "waiting for NInfer model %s on %s:%s%s",
                    entry.id,
                    self.host,
                    self.port,
                    f" ({last_error})" if last_error else "",
                )
            time.sleep(1.0)
        raise EngineFailure(
            f"NInfer model {entry.id} did not become ready within {int(timeout)}s; "
            f"log: {tail_file(self.log_path, 40)}"
        )

    def _llama_models(self) -> list[dict[str, Any]]:
        status, body = self._json_request("GET", "/models", timeout=5)
        if status != 200 or not isinstance(body, dict):
            raise EngineFailure(f"llama /models returned HTTP {status}: {self._error_from_body(status, body)}")
        data = body.get("data")
        if not isinstance(data, list):
            raise EngineFailure("llama /models returned no data list")
        return data

    @staticmethod
    def _llama_status(entry: dict[str, Any]) -> str:
        status = entry.get("status")
        if isinstance(status, dict):
            return str(status.get("value") or "unknown")
        return str(status or "unknown")

    def _llama_load_failure(self, entry: Model, status: dict[str, Any]) -> EngineFailure:
        code = status.get("exit_code")
        tail = tail_file(self.log_path, 60)
        low = tail.lower()
        if any(marker in low for marker in (
            "out of memory",
            "cudamalloc failed",
            "failed to allocate buffer for kv cache",
        )):
            return EngineFailure(
                f"llama 加载 {entry.id} 失败：显存不足，模型或 KV cache 无法装入 GPU"
                f"（exit code {code}）；日志：{self.log_path}"
            )
        return EngineFailure(f"llama failed to load {entry.id} (exit code {code}); log: {tail}")

    def _load_llama(self, entry: Model) -> None:
        timeout = float(self.timeouts.get("load_timeout_s") or 900)
        deadline = time.monotonic() + timeout
        target_id = entry.engine_model
        last_note = 0.0
        while True:
            if not self.alive():
                raise EngineFailure(f"llama engine exited while loading {entry.id}")
            data = self._llama_models()
            target = next((m for m in data if m.get("id") == target_id), None)
            if target is not None:
                state = self._llama_status(target)
                if state in ("loaded", "sleeping"):
                    self.current_model = target_id
                    return
                if state == "unloaded" and isinstance(target.get("status"), dict) and target["status"].get("failed"):
                    raise self._llama_load_failure(entry, target["status"])
            if time.monotonic() >= deadline:
                raise EngineFailure(f"llama did not load {entry.id} within {int(timeout)}s; log: {tail_file(self.log_path, 60)}")
            if time.monotonic() - last_note > 15:
                last_note = time.monotonic()
                self.log.info("loading llama model %s", entry.id)
            time.sleep(1.0)

    def load(self, entry: Model) -> None:
        if not self.alive():
            raise EngineFailure(f"{self.engine_id}: engine is not running")
        if self.kind == "llama_router":
            # An engine that just started reports itself healthy before its models
            # are loaded; always confirm the router's per-model state.
            self._load_llama_model(entry)
        elif self.kind == "ninfer":
            # NInfer runs one artifact per process; switching models means a
            # restart, which the gateway already does when it switches engines.
            if self.current_model == entry.engine_model:
                self.current_entry = entry
                return
            self.log.info("restarting NInfer engine for %s (was %s)", entry.id, self.current_model or "—")
            self.stop()
            self.start(entry)
        elif self.current_model == entry.engine_model:
            self.current_entry = entry
            return
        else:
            self._load_kvmem_model(entry)
        self.current_model = entry.engine_model
        self.current_entry = entry
        self._write_state()

    def _load_llama_model(self, entry: Model) -> None:
        timeout = float(self.timeouts.get("load_timeout_s") or 900)
        target_id = entry.engine_model
        data = self._llama_models()
        target = next((m for m in data if m.get("id") == target_id), None)
        if target is None:
            raise EngineFailure(
                f"llama router does not list model {target_id}; available: "
                + ", ".join(str(m.get("id")) for m in data[:30])
            )
        state = self._llama_status(target)
        if state in ("loaded", "sleeping"):
            return
        if state == "unloaded" and isinstance(target.get("status"), dict) and target["status"].get("failed"):
            raise self._llama_load_failure(entry, target["status"])
        # The gateway keeps one engine/model at a time.  Unload anything else first so
        # the 8 GB card is not asked to hold two models.
        others = [
            m for m in data
            if m.get("id") != target_id and self._llama_status(m) in ("loaded", "loading", "sleeping")
        ]
        for other in others:
            other_id = str(other.get("id"))
            self.log.info("unloading llama model %s before loading %s", other_id, entry.id)
            status, body = self._json_request("POST", "/models/unload", json_body={"model": other_id}, timeout=15)
            if status >= 400:
                self.log.warning("unload %s returned HTTP %s: %s", other_id, status, self._error_from_body(status, body))
        unload_deadline = time.monotonic() + min(180.0, timeout)
        while others and time.monotonic() < unload_deadline:
            data = self._llama_models()
            busy = [
                m for m in data
                if m.get("id") != target_id and self._llama_status(m) in ("loaded", "loading", "sleeping")
            ]
            if not busy:
                break
            time.sleep(1.0)
        status, body = self._json_request("POST", "/models/load", json_body={"model": target_id}, timeout=30)
        if status >= 400:
            message = self._error_from_body(status, body)
            if status == 400 and "already running" in message.lower():
                pass
            else:
                raise EngineFailure(f"llama refused to load {entry.id} (HTTP {status}): {message}")
        self.log.info("llama load requested for %s; waiting", entry.id)
        self._load_llama(entry)

    def _kvmem_models(self, entry: Model) -> tuple[list[dict[str, Any]], str | None]:
        status, body = self._json_request("GET", "/api/models", entry=entry, timeout=8)
        if status != 200 or not isinstance(body, dict):
            raise EngineFailure(f"KVMem /api/models returned HTTP {status}: {self._error_from_body(status, body)}")
        models = body.get("models")
        if not isinstance(models, list):
            raise EngineFailure("KVMem /api/models returned no models list")
        return models, body.get("current")

    def _load_kvmem_model(self, entry: Model) -> None:
        timeout = float(self.timeouts.get("load_timeout_s") or 900)
        models, current = self._kvmem_models(entry)
        target = next((m for m in models if m.get("id") == entry.engine_model), None)
        if target is None:
            raise EngineFailure(
                f"KVMem does not list model {entry.engine_model}; available: "
                + ", ".join(str(m.get("id")) for m in models[:30])
            )
        if target.get("status") == "unavailable":
            raise EngineFailure(f"KVMem reports {entry.id} unavailable: {target.get('problem') or 'unknown reason'}")
        if current == entry.engine_model and target.get("status") == "loaded":
            return
        deadline = time.monotonic() + timeout
        last_message = ""
        while True:
            status, body = self._json_request(
                "POST",
                "/api/models/load",
                entry=entry,
                json_body={"model": entry.engine_model},
                timeout=min(timeout + 60, 960),
            )
            if status == 200:
                break
            message = self._error_from_body(status, body)
            last_message = message
            if status in (409, 503) and time.monotonic() < deadline:
                self.log.info("KVMem is busy switching to %s; retrying: %s", entry.id, message)
                time.sleep(2.0)
                continue
            raise EngineFailure(f"KVMem refused to load {entry.id} (HTTP {status}): {message}")
        models, current = self._kvmem_models(entry)
        target = next((m for m in models if m.get("id") == entry.engine_model), None)
        if current != entry.engine_model or target is None or target.get("status") != "loaded":
            raise EngineFailure(
                f"KVMem reported success but {entry.id} is not loaded (current={current}, "
                f"status={target.get('status') if target else 'missing'}); last message: {last_message}"
            )

    def unload(self) -> None:
        if not self.alive():
            self.current_model = None
            self.current_entry = None
            self._clear_state()
            return
        if self.kind == "ninfer":
            # NInfer has no unload endpoint in single-model mode; stopping the
            # engine is the unload, and the next request starts it again.
            self.stop()
            return
        if self.kind == "llama_router":
            data = self._llama_models()
            for model in data:
                if self._llama_status(model) in ("loaded", "loading", "sleeping"):
                    status, body = self._json_request(
                        "POST", "/models/unload", json_body={"model": model.get("id")}, timeout=15
                    )
                    if status >= 400:
                        self.log.warning("llama unload %s: HTTP %s %s", model.get("id"), status, self._error_from_body(status, body))
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if not any(self._llama_status(m) in ("loaded", "loading", "sleeping") for m in self._llama_models()):
                    break
                time.sleep(1.0)
        else:
            entry = self.current_entry
            deadline = time.monotonic() + 120
            while True:
                status, body = self._json_request(
                    "POST", "/api/models/unload", entry=entry, json_body={}, timeout=30
                )
                if status == 200:
                    break
                message = self._error_from_body(status, body)
                if status in (409, 503) and time.monotonic() < deadline:
                    self.log.info("KVMem is busy; retrying unload: %s", message)
                    time.sleep(2.0)
                    continue
                if status in (401, 403):
                    raise EngineFailure(
                        f"KVMem rejected the API key while unloading (HTTP {status}); "
                        "check engines.kvmem.api_key"
                    )
                raise EngineFailure(f"KVMem refused to unload (HTTP {status}): {message}")
        self.current_model = None
        self.current_entry = None
        self._write_state()

    def stop(self) -> None:
        proc = self.proc
        if proc is None:
            self._clear_state()
            return
        pid = proc.pid
        pgid = self.pgid or pid
        self.log.info("stopping engine %s (pid=%s, pgid=%s)", self.engine_id, pid, pgid)
        if proc.poll() is None:
            if self.kind == "ninfer":
                # The first SIGINT only asks NInfer to confirm ("press Ctrl+C
                # again"), so a plain SIGINT would burn the full stop timeout.
                # SIGTERM is its immediate graceful shutdown path.
                kill_group(pgid, signal.SIGTERM)
                timeout = min(float(self.timeouts.get("stop_timeout_s") or 90), 15.0)
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.log.warning("engine %s did not stop after SIGTERM; sending SIGKILL", self.engine_id)
                    kill_group(pgid, signal.SIGKILL)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
            else:
                kill_group(pgid, signal.SIGINT)
                try:
                    proc.wait(timeout=float(self.timeouts.get("stop_timeout_s") or 90))
                except subprocess.TimeoutExpired:
                    self.log.warning("engine %s did not stop after SIGINT; sending SIGTERM", self.engine_id)
                    kill_group(pgid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.log.warning("engine %s did not stop after SIGTERM; sending SIGKILL", self.engine_id)
                        kill_group(pgid, signal.SIGKILL)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
        # The parent may be gone while a child still holds the port/GPU.
        if group_alive(pgid):
            kill_group(pgid, signal.SIGKILL)
        self._close_log()
        self.proc = None
        self.pgid = None
        self.current_model = None
        self.current_entry = None
        self._clear_state()
        free = wait_port_free(self.host, self.port, float(self.timeouts.get("port_free_timeout_s") or 30))
        if not free:
            self.log.error("port %s:%s is still in use after stopping %s", self.host, self.port, self.engine_id)
        else:
            self.log.info("engine %s stopped", self.engine_id)

    def _close_log(self) -> None:
        if self.log_fh is not None:
            try:
                self.log_fh.flush()
                self.log_fh.close()
            except OSError:
                pass
            self.log_fh = None

    def status(self) -> dict[str, Any]:
        return {
            "engine": self.engine_id,
            "kind": self.kind,
            "host": self.host,
            "port": self.port,
            "pid": self.proc.pid if self.proc else None,
            "pgid": self.pgid,
            "alive": self.alive(),
            "model": self.current_model,
            "log_path": str(self.log_path),
            "started_at": self.started_at,
        }


# ---------------------------------------------------------------------------
# gateway state and switching


class Gateway:
    def __init__(
        self,
        config: dict[str, Any],
        registry: Registry,
        engines: dict[str, EngineManager],
        state_path: Path,
        base_dir: Path,
        log: logging.Logger,
        config_path: Path | None = None,
    ):
        self.config = config
        self.registry = registry
        self.engines = engines
        self.state_path = state_path
        self.base_dir = base_dir
        self.log = log
        self.config_path = config_path or (base_dir / "config.json")
        self._config_lock = threading.Lock()
        self._auth_lock = threading.Lock()
        self.timeouts = config.get("timeouts", {})
        self.cond = threading.Condition(threading.RLock())
        self.switching = False
        self.in_flight = 0
        self.queue = 0
        self.active: Model | None = None
        self.target: Model | None = None
        self.phase = "idle"
        self.last_error: str | None = None
        self.last_switch: dict[str, Any] | None = None
        self.engine: EngineManager | None = None
        self.started_at = now()
        self.pending_load: Model | None = None
        self.pending_unload: Model | None = None
        self.async_operation: dict[str, Any] | None = None
        self._shutting_down = False
        self._shutdown_lock = threading.Lock()
        self.auth_key = self._load_auth_key()
        params_path = resolve_path(
            base_dir, config.get("model_params_file") or "model-params.json"
        )
        self.params = SamplingStore(params_path, log, registry)
        self.status_hub = StatusHub(self._hub_model_list, self._hub_stamp, log)
        self.model_jobs_lock = threading.Lock()
        self.model_jobs: dict[str, dict[str, Any]] = {}
        self.download_jobs_lock = threading.Lock()
        self.download_jobs: dict[str, dict[str, Any]] = {}
        self.download_procs: dict[str, subprocess.Popen[Any]] = {}
        self.download_logs: dict[str, Path] = {}
        self.smoke_image_path = resolve_path(
            base_dir, config.get("smoke_image") or "smoke.png"
        )

    def _load_auth_key(self) -> str:
        """One client-facing key for every model.  The value may live in a separate file."""
        auth = self.config.get("auth")
        auth = auth if isinstance(auth, dict) else {}
        key = str(auth.get("api_key") or "").strip()
        file_value = auth.get("api_key_file")
        if not key and file_value:
            path = resolve_path(self.base_dir, file_value)
            try:
                lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                key = lines[0] if lines else ""
            except OSError as exc:
                self.log.warning(
                    "cannot read auth.api_key_file %s: %s; inference API authentication is disabled",
                    path,
                    exc,
                )
        if key:
            self.log.info("inference API authentication is enabled")
        else:
            self.log.warning(
                "inference API authentication is disabled (set auth.api_key or auth.api_key_file)"
            )
        return key

    def _auth_config(self, create: bool = False) -> dict[str, Any]:
        auth = self.config.get("auth")
        if isinstance(auth, dict):
            return auth
        if create:
            auth = {}
            self.config["auth"] = auth
            return auth
        return {}

    def _auth_key_path(self) -> Path:
        return resolve_path(self.base_dir, self._auth_config().get("api_key_file") or "api-key")

    def _auth_key_label(self) -> str:
        path = self._auth_key_path()
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)

    @staticmethod
    def _key_hint(key: str) -> str:
        return "****" + key[-4:] if len(key) >= 8 else "****"

    def auth_view(self) -> dict[str, Any]:
        """Describe the server-side key without ever returning the secret."""
        key = self.auth_key
        auth = self._auth_config()
        source = None
        if key:
            source = "config" if str(auth.get("api_key") or "").strip() else "file"
        return {
            "configured": bool(key),
            "source": source,
            "hint": self._key_hint(key) if key else None,
            "key_file": self._auth_key_label(),
            "protected": ["/v1/*", "/props"],
        }

    def _write_auth_file(self, key: str) -> None:
        path = self._auth_key_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write((key + "\n") if key else "")
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def set_auth_key(self, key: str | None) -> dict[str, Any]:
        """Set (or clear) the server key at runtime and persist it for restarts.

        The key lives in a private file referenced by auth.api_key_file; an inline
        auth.api_key is removed so the file remains the single source of truth.
        """
        value = (key or "").strip()
        if "\n" in value or "\r" in value or "\x00" in value:
            raise BadRequest("API key must be a single line")
        if len(value) > 512:
            raise BadRequest("API key must be at most 512 characters")
        with self._auth_lock:
            auth = self._auth_config(create=True)
            auth.pop("api_key", None)
            auth["api_key_file"] = self._auth_key_label()
            try:
                self._write_auth_file(value)
            except OSError as exc:
                self.log.exception("cannot write API key file %s", self._auth_key_label())
                raise GatewayError(f"cannot write API key file: {exc}")
            self.auth_key = value
            try:
                with self._config_lock:
                    write_json_atomic(self.config_path, self.config)
            except OSError as exc:
                self.log.exception("cannot persist auth configuration to %s", self.config_path)
                raise GatewayError(
                    f"API key file was updated but {self.config_path.name} could not be saved: {exc}"
                )
            if value:
                self.log.info("inference API authentication enabled/updated from console")
            else:
                self.log.info("inference API authentication disabled from console")
        return self.auth_view()

    # -- state helpers -----------------------------------------------------

    def _set_phase(self, phase: str, target: Model | None = None, error: str | None = None) -> None:
        with self.cond:
            self.phase = phase
            if target is not None:
                self.target = target
            if error is not None:
                self.last_error = error

    def _clear_active_engine(self) -> None:
        with self.cond:
            self.active = None
            self.engine = None

    def _start_engine_locked(self, entry: Model) -> None:
        manager = self.engines.get(entry.engine)
        if manager is None:
            raise EngineFailure(f"no engine configured for {entry.engine}")
        self._set_phase("starting", entry)
        manager.start(entry)
        with self.cond:
            self.engine = manager
            self.active = None

    def _stop_engine_locked(self, target: Model | None = None) -> None:
        with self.cond:
            self.active = None
            self.phase = "stopping"
            if target is not None:
                self.target = target
            manager = self.engine
            self.engine = None
        if manager is not None:
            manager.stop()

    def _do_switch(self, entry: Model) -> None:
        started = time.monotonic()
        old = None
        with self.cond:
            old = self.active
        self.log.info("switching to %s (%s)", entry.id, entry.engine)
        try:
            if self.engine is not None and not self.engine.alive():
                self.log.warning("engine %s is not alive; restarting it", self.engine.engine_id)
                self._stop_engine_locked(entry)
            if self.engine is None:
                self._start_engine_locked(entry)
            elif self.engine.engine_id != entry.engine:
                self._stop_engine_locked(entry)
                self._start_engine_locked(entry)
            else:
                self._set_phase("loading", entry)
            if self.engine is None:
                raise EngineFailure("engine was not started")
            self._set_phase("loading", entry)
            self.engine.load(entry)
            elapsed = time.monotonic() - started
            with self.cond:
                self.active = entry
                self.phase = "ready"
                self.target = None
                self.last_error = None
                self.last_switch = {
                    "at": iso_now(),
                    "from": f"{old.engine}/{old.id}" if old else None,
                    "to": f"{entry.engine}/{entry.id}",
                    "seconds": round(elapsed, 2),
                }
            self.log.info("switch to %s done in %.1fs", entry.id, elapsed)
        except Exception as exc:
            self.log.exception("switch to %s failed", entry.id)
            with self.cond:
                self.active = None
                self.phase = "error"
                self.last_error = str(exc)
                self.target = entry
            try:
                self._stop_engine_locked(entry)
            except Exception:
                self.log.exception("cannot stop engine after failed switch")
            raise

    # -- async control (llama.cpp router UI compatible) ---------------------

    def _hub_model_list(self) -> dict[str, Any]:
        return self.model_list()

    def _hub_stamp(self) -> float:
        return self.registry.loaded_at

    def shutting_down(self) -> bool:
        with self.cond:
            return self._shutting_down

    def ensure_no_async(self) -> None:
        with self.cond:
            if self.async_operation is not None:
                raise ModelBusy("another load/unload operation is in progress")
            if self.switching:
                raise ModelBusy("gateway is switching models; retry shortly")

    def schedule_load(self, name: str) -> tuple[Model, bool]:
        entry = self.registry.resolve(name)
        if entry is None:
            raise ModelNotFound(f"model {name!r} is not in the registry")
        if not entry.available:
            raise ModelUnavailable(f"model {entry.id} is unavailable: {entry.problem}")
        with self.cond:
            if self._shutting_down:
                raise ModelBusy("gateway is shutting down")
            active = self.active
            if (
                active is not None
                and active.id == entry.id
                and self.engine is not None
                and self.engine.alive()
            ):
                started = False
            elif self.async_operation is not None:
                raise ModelBusy("another load/unload operation is in progress")
            elif self.switching:
                raise ModelBusy("gateway is switching models; retry shortly")
            else:
                self.async_operation = {
                    "kind": "load",
                    "model": entry.id,
                    "engine": entry.engine,
                    "started_at": iso_now(),
                }
                self.pending_load = entry
                self.target = entry
                self.phase = "queued"
                self.last_error = None
                self.cond.notify_all()
                started = True
        self.status_hub.publish(entry.id, "loading" if started else "loaded")
        if not started:
            return entry, False
        thread = threading.Thread(
            target=self._async_load_worker,
            args=(entry,),
            name=f"load-{entry.id}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            with self.cond:
                self.pending_load = None
                self.async_operation = None
                self.phase = "error"
                self.last_error = f"cannot start loader thread: {exc}"
                self.cond.notify_all()
            self.status_hub.publish(entry.id, "failed")
            raise EngineFailure(f"cannot start loader thread: {exc}") from exc
        return entry, True

    def _async_load_worker(self, entry: Model) -> None:
        try:
            self.preload(entry.id)
            self.status_hub.publish(entry.id, "loaded")
        except Exception as exc:
            self.log.exception("asynchronous load of %s failed", entry.id)
            with self.cond:
                self.phase = "error"
                self.target = entry
                self.last_error = str(exc)
            self.status_hub.publish(entry.id, "failed")
        finally:
            with self.cond:
                self.pending_load = None
                self.async_operation = None
                self.cond.notify_all()

    def schedule_unload(self, name: str | None = None) -> tuple[Model | None, bool]:
        entry: Model | None = None
        if name:
            entry = self.registry.resolve(name)
            if entry is None:
                raise ModelNotFound(f"model {name!r} is not in the registry")
        with self.cond:
            if self._shutting_down:
                raise ModelBusy("gateway is shutting down")
            if self.async_operation is not None:
                raise ModelBusy("another load/unload operation is in progress")
            if self.switching:
                raise ModelBusy("gateway is switching models; retry shortly")
            active = self.active
            if active is None or (entry is not None and active.id != entry.id):
                if entry is not None:
                    self.status_hub.publish(entry.id, "unloaded")
                return entry, False
            entry = active
            self.async_operation = {
                "kind": "unload",
                "model": entry.id,
                "engine": entry.engine,
                "started_at": iso_now(),
            }
            self.pending_unload = entry
            self.target = entry
            self.phase = "unloading"
            self.last_error = None
            self.cond.notify_all()
        self.status_hub.publish(entry.id, "loading")
        thread = threading.Thread(
            target=self._async_unload_worker,
            args=(entry,),
            name=f"unload-{entry.id}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            with self.cond:
                self.pending_unload = None
                self.async_operation = None
                self.phase = "error"
                self.last_error = f"cannot start unloader thread: {exc}"
                self.cond.notify_all()
            self.status_hub.publish(entry.id, "failed")
            raise EngineFailure(f"cannot start unloader thread: {exc}") from exc
        return entry, True

    def _async_unload_worker(self, entry: Model) -> None:
        try:
            self.unload_model()
            self.status_hub.publish(entry.id, "unloaded")
        except Exception as exc:
            self.log.exception("asynchronous unload of %s failed", entry.id)
            with self.cond:
                self.phase = "error"
                self.target = entry
                self.last_error = str(exc)
            self.status_hub.publish(entry.id, "failed")
        finally:
            with self.cond:
                self.pending_unload = None
                self.async_operation = None
                self.cond.notify_all()

    # -- gate --------------------------------------------------------------

    def _wait_for_switch_locked(self, deadline: float) -> None:
        while self.switching:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelBusy("gateway is still switching models; retry shortly")
            self.queue += 1
            try:
                self.cond.wait(min(remaining, 1.0))
            finally:
                self.queue -= 1
        if self._shutting_down:
            raise ModelBusy("gateway is shutting down")

    def _drain_locked(self) -> None:
        deadline = time.monotonic() + float(self.timeouts.get("drain_timeout_s") or 300)
        while self.in_flight > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelBusy("the current model is still generating; switch was not started")
            self.cond.wait(min(remaining, 1.0))

    def acquire(self, name: str) -> Model:
        entry = self.registry.resolve(name)
        if entry is None:
            raise ModelNotFound(f"model {name!r} is not in the registry")
        if not entry.available:
            raise ModelUnavailable(f"model {entry.id} is unavailable: {entry.problem}")
        deadline = time.monotonic() + float(self.timeouts.get("queue_timeout_s") or 1200)
        with self.cond:
            while True:
                self._wait_for_switch_locked(deadline)
                current = self.active
                if (
                    current is not None
                    and current.engine == entry.engine
                    and current.id == entry.id
                    and self.engine is not None
                    and self.engine.alive()
                ):
                    self.in_flight += 1
                    return entry
                if current is not None and (self.engine is None or not self.engine.alive()):
                    self.active = None
                self.switching = True
                try:
                    self._drain_locked()
                except Exception:
                    self.switching = False
                    self.cond.notify_all()
                    raise
                break
        try:
            self._do_switch(entry)
        except Exception:
            with self.cond:
                self.switching = False
                self.cond.notify_all()
            raise
        with self.cond:
            self.switching = False
            self.in_flight += 1
            self.cond.notify_all()
            return entry

    def release(self, entry: Model | None) -> None:
        with self.cond:
            if self.in_flight > 0:
                self.in_flight -= 1
            self.cond.notify_all()

    def preload(self, name: str) -> Model:
        entry = self.registry.resolve(name)
        if entry is None:
            raise ModelNotFound(f"model {name!r} is not in the registry")
        if not entry.available:
            raise ModelUnavailable(f"model {entry.id} is unavailable: {entry.problem}")
        deadline = time.monotonic() + float(self.timeouts.get("queue_timeout_s") or 1200)
        with self.cond:
            while True:
                self._wait_for_switch_locked(deadline)
                current = self.active
                if (
                    current is not None
                    and current.engine == entry.engine
                    and current.id == entry.id
                    and self.engine is not None
                    and self.engine.alive()
                ):
                    return entry
                if current is not None and (self.engine is None or not self.engine.alive()):
                    self.active = None
                self.switching = True
                try:
                    self._drain_locked()
                except Exception:
                    self.switching = False
                    self.cond.notify_all()
                    raise
                break
        try:
            self._do_switch(entry)
        except Exception:
            with self.cond:
                self.switching = False
                self.cond.notify_all()
            raise
        with self.cond:
            self.switching = False
            self.cond.notify_all()
            return entry

    def unload_model(self) -> None:
        deadline = time.monotonic() + float(self.timeouts.get("queue_timeout_s") or 1200)
        with self.cond:
            self._wait_for_switch_locked(deadline)
            self.switching = True
            try:
                self._drain_locked()
            except Exception:
                self.switching = False
                self.cond.notify_all()
                raise
            manager = self.engine
        try:
            if manager is not None:
                manager.unload()
            with self.cond:
                self.active = None
                self.phase = "idle"
                self.target = None
                self.last_error = None
        except Exception as exc:
            with self.cond:
                self.last_error = str(exc)
                self.phase = "error"
                self.target = self.active
            raise
        finally:
            with self.cond:
                self.switching = False
                self.cond.notify_all()

    def stop_engine(self) -> None:
        deadline = time.monotonic() + float(self.timeouts.get("queue_timeout_s") or 1200)
        with self.cond:
            self._wait_for_switch_locked(deadline)
            self.switching = True
            try:
                self._drain_locked()
            except Exception:
                self.switching = False
                self.cond.notify_all()
                raise
            manager = self.engine
            self.active = None
            self.engine = None
        try:
            if manager is not None:
                manager.stop()
            with self.cond:
                self.phase = "idle"
                self.target = None
                self.last_error = None
        finally:
            with self.cond:
                self.switching = False
                self.cond.notify_all()

    def shutdown(self) -> None:
        with self._shutdown_lock:
            with self.cond:
                self._shutting_down = True
                self.cond.notify_all()
                manager = self.engine
                self.engine = None
                self.active = None
            if manager is not None:
                try:
                    manager.stop()
                except Exception:
                    self.log.exception("error while stopping engine during shutdown")
            self.cancel_all_downloads()
            self.status_hub.close()

    # -- model management (console add/delete) ------------------------------

    def _engine_kind(self, engine_id: str) -> str | None:
        engine_cfg = self.config.get("engines", {}).get(engine_id)
        if not isinstance(engine_cfg, dict):
            return None
        models = engine_cfg.get("models")
        return str(models.get("kind")) if isinstance(models, dict) else None

    @staticmethod
    def _short_kind(kind: str | None) -> str:
        return {"llama_dir": "llama", "kvmem_dir": "kvmem", "ninfer_dir": "ninfer"}.get(str(kind), "")

    def _engine_config_label(self) -> str:
        try:
            return str(self.config_path.relative_to(self.base_dir))
        except ValueError:
            return str(self.config_path)

    def _backup_config(self) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = self.base_dir / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"config-{stamp}.json"
        index = 1
        while target.exists():
            target = folder / f"config-{stamp}-{index}.json"
            index += 1
        target.write_bytes(self.config_path.read_bytes())
        return target

    def engine_paths_view(self) -> dict[str, Any]:
        engines_cfg = self.config.get("engines")
        engines_cfg = engines_cfg if isinstance(engines_cfg, dict) else {}
        counts = self.registry.counts()
        items = []
        for engine_id in ENGINE_SLOT_ORDER:
            spec = ENGINE_SLOTS[engine_id]
            engine_cfg = engines_cfg.get(engine_id)
            configured = isinstance(engine_cfg, dict)
            manager = self.engines.get(engine_id)
            running = bool(manager is not None and manager.alive())
            current = engine_path_fields(engine_id, engine_cfg) if configured else {}
            defaults = engine_slot_defaults(self.base_dir, engine_id)
            # Unconfigured slots have no real values: keep the inputs empty and
            # expose the conventional layout only as ``defaults`` suggestions.
            # Filling them in made an untouched slot look configured (and the
            # guessed paths often do not exist on a fresh deployment).
            fields = (
                {key: str(current.get(key) or "") for key in spec["fields"]}
                if configured
                else {key: "" for key in spec["fields"]}
            )
            needs_restart = False
            if running and manager is not None:
                needs_restart = (
                    manager.started_paths is None
                    or manager.started_paths != engine_path_fields(engine_id, manager.cfg)
                )
            model_count = int(counts.get(engine_id, 0))
            warning = None
            if configured and model_count == 0:
                warning = "目录中未发现模型；可继续在控制台新增模型，或检查扫描路径。"
            items.append(
                {
                    "engine": engine_id,
                    "label": spec["label"],
                    "kind": engine_id,
                    "configured": configured,
                    "running": running,
                    "needs_restart": needs_restart,
                    "fields": fields,
                    "defaults": defaults,
                    "models": model_count,
                    "warning": warning,
                    "address": f"{spec['host']}:{spec['port']}",
                }
            )
        return {
            "object": "list",
            "config_file": self._engine_config_label(),
            "engines": items,
        }

    def save_engine_paths(
        self, engine_id: str, paths: Any, create: bool = False
    ) -> dict[str, Any]:
        if engine_id not in ENGINE_SLOTS:
            raise BadRequest("engine 必须是 llama、kvmem 或 ninfer")
        engines_cfg = self.config.get("engines")
        if engines_cfg is None:
            engines_cfg = {}
            self.config["engines"] = engines_cfg
        if not isinstance(engines_cfg, dict):
            raise GatewayError("config.engines 必须是对象")
        existing = engines_cfg.get(engine_id)
        if existing is not None and not isinstance(existing, dict):
            raise BadRequest(f"engines.{engine_id} 必须是对象")
        if existing is None and not create:
            raise BadRequest("引擎尚未配置；请确认创建引擎")
        if existing is not None and create:
            raise BadRequest("引擎已配置，不能重复创建")
        if existing is not None:
            models_cfg = existing.get("models")
            kind = models_cfg.get("kind") if isinstance(models_cfg, dict) else None
            if kind and str(kind) != ENGINE_SLOTS[engine_id]["models_kind"]:
                raise BadRequest(
                    f"engines.{engine_id}.models.kind={kind!r} 与固定槽位不匹配"
                )
        fields, pending_paths = validate_engine_paths(
            engine_id, paths, self.base_dir, create=existing is None
        )
        candidate = build_engine_cfg(engine_id, fields, existing)
        probe_engine_binary(engine_id, fields, candidate)
        created_notes = materialize_engine_paths(pending_paths)
        scanned = scan_engine_models(engine_id, candidate)
        with self._config_lock:
            backup_path = None
            if self.config_path.is_file():
                try:
                    backup_path = self._backup_config()
                except OSError as exc:
                    self.log.exception("cannot back up %s", self.config_path)
                    raise GatewayError(f"无法备份 {self.config_path.name}：{exc}") from exc
            new_config = copy.deepcopy(self.config)
            new_engines = new_config.get("engines")
            if not isinstance(new_engines, dict):
                new_engines = {}
                new_config["engines"] = new_engines
            new_engines[engine_id] = copy.deepcopy(candidate)
            try:
                write_json_atomic(self.config_path, new_config)
            except OSError as exc:
                self.log.exception("cannot write %s", self.config_path)
                raise GatewayError(f"无法写入 {self.config_path.name}：{exc}") from exc
            live_engines = self.config.get("engines")
            if not isinstance(live_engines, dict):
                live_engines = {}
                self.config["engines"] = live_engines
            current = live_engines.get(engine_id)
            if isinstance(current, dict):
                current.clear()
                current.update(copy.deepcopy(candidate))
                cfg = current
            else:
                live_engines[engine_id] = copy.deepcopy(candidate)
                cfg = live_engines[engine_id]
            manager = self.engines.get(engine_id)
            if manager is None:
                self.engines[engine_id] = EngineManager(
                    engine_id,
                    cfg,
                    self.base_dir,
                    self.timeouts,
                    self.state_path,
                    self.log,
                )
            else:
                manager.cfg = cfg
        self.registry.reload()
        self.log.info(
            "%s engine paths %s; %d models discovered",
            "created" if existing is None else "updated",
            engine_id,
            len(scanned),
        )
        view = self.engine_paths_view()
        view.update(
            {
                "status": "ok",
                "created": existing is None,
                "backup": str(backup_path) if backup_path else None,
            }
        )
        notes = list(created_notes)
        if not scanned:
            notes.append(
                f"{ENGINE_SLOTS[engine_id]['label']} 引擎已配置，但目录中没有发现模型。"
            )
        if notes:
            view["note"] = " ".join(notes)
        return view

    def restart_engine_paths(self, engine_id: str) -> dict[str, Any]:
        if engine_id not in ENGINE_SLOTS:
            raise BadRequest("engine 必须是 llama、kvmem 或 ninfer")
        engines_cfg = self.config.get("engines")
        if not isinstance(engines_cfg, dict) or not isinstance(
            engines_cfg.get(engine_id), dict
        ):
            raise BadRequest("引擎尚未配置")
        manager = self.engines.get(engine_id)
        if manager is None or not manager.alive():
            return {
                "status": "ok",
                "restarting": False,
                "restart_pending": False,
                "note": "引擎未运行；新路径将在下次加载模型时生效",
                "view": self.engine_paths_view(),
            }
        started, pending, note = self._schedule_engine_restart(engine_id)
        return {
            "status": "ok",
            "restarting": started,
            "restart_pending": pending,
            "note": note,
            "view": self.engine_paths_view(),
        }

    # -- proxy / model search and download ---------------------------------

    def _proxy_config(self, create: bool = False) -> dict[str, Any]:
        proxy = self.config.get("proxy")
        if isinstance(proxy, dict):
            return proxy
        if create:
            proxy = {}
            self.config["proxy"] = proxy
            return proxy
        return {}

    def _downloads_config(self, create: bool = False) -> dict[str, Any]:
        downloads = self.config.get("downloads")
        if isinstance(downloads, dict):
            return downloads
        if create:
            downloads = {}
            self.config["downloads"] = downloads
            return downloads
        return {}

    def _download_sources_config(self) -> dict[str, Any]:
        sources = self._downloads_config().get("sources")
        return sources if isinstance(sources, dict) else {}

    def _normalize_proxy(
        self, payload: dict[str, Any], current: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        current = current if isinstance(current, dict) else {}
        enabled = payload.get("enabled", current.get("enabled", False))
        if not isinstance(enabled, bool):
            raise BadRequest("enabled 必须是布尔值")
        kind = str(payload.get("type", current.get("type") or "http")).strip().lower()
        if kind not in PROXY_TYPES:
            raise BadRequest("代理类型必须是 http 或 socks5")
        host = str(payload.get("host", current.get("host")) or "").strip()
        if len(host) > 255:
            raise BadRequest("代理地址过长")
        if any(ch in host for ch in ("/", "@", "#", "?", "\\")) or any(
            ch.isspace() for ch in host
        ):
            raise BadRequest("代理地址不能包含协议、路径或空格")
        raw_port = payload.get("port", current.get("port"))
        if raw_port is None or raw_port == "":
            port = 0
        else:
            parsed = parse_int(raw_port)
            if parsed is None:
                raise BadRequest("代理端口必须是整数")
            port = parsed
        if port and not (1 <= port <= 65535):
            raise BadRequest("代理端口必须在 1-65535 之间")
        username = str(payload.get("username", current.get("username")) or "").strip()
        if "password" in payload:
            password = str(payload.get("password") or "")
        else:
            password = str(current.get("password") or "")
        if payload.get("clear_password") is True:
            password = ""
        for label, value in (("用户名", username), ("密码", password)):
            if len(value) > (256 if label == "用户名" else 512):
                raise BadRequest(f"代理{label}过长")
            if any(ord(ch) < 32 for ch in value):
                raise BadRequest(f"代理{label}不能包含控制字符")
        if enabled and (not host or not port):
            raise BadRequest("启用代理时必须填写地址和端口")
        if (host or port) and not (host and port):
            raise BadRequest("代理地址和端口需要同时填写")
        return {
            "enabled": enabled,
            "type": kind,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
        }

    def proxy_view(self) -> dict[str, Any]:
        proxy = self._proxy_config()
        kind = str(proxy.get("type") or "http").strip().lower()
        if kind not in PROXY_TYPES:
            kind = "http"
        host = str(proxy.get("host") or "").strip()
        port = parse_int(proxy.get("port")) or 0
        enabled = bool(proxy.get("enabled"))
        configured = bool(
            enabled and host and 1 <= port <= 65535 and kind in PROXY_TYPES
        )
        sources_cfg = self._download_sources_config()
        sources = []
        for source_id in DOWNLOAD_SOURCE_ORDER:
            entry = sources_cfg.get(source_id)
            sources.append(
                {
                    "id": source_id,
                    "label": DOWNLOAD_SOURCES[source_id]["label"],
                    "use_proxy": bool(
                        configured
                        and isinstance(entry, dict)
                        and entry.get("use_proxy")
                    ),
                }
            )
        return {
            "enabled": enabled,
            "type": kind,
            "host": host,
            "port": port or None,
            "username": str(proxy.get("username") or ""),
            "has_password": bool(str(proxy.get("password") or "")),
            "configured": configured,
            "sources": sources,
            "config_file": self._engine_config_label(),
        }

    def _write_config_locked(self, new_config: dict[str, Any]) -> None:
        try:
            write_json_atomic(self.config_path, new_config)
        except OSError as exc:
            self.log.exception("cannot write %s", self.config_path)
            raise GatewayError(f"无法写入 {self.config_path.name}：{exc}") from exc
        self.config.clear()
        self.config.update(new_config)

    def save_proxy(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._normalize_proxy(payload, self._proxy_config())
        proxy_ready = bool(settings["enabled"] and settings["host"] and settings["port"])
        source_updates: dict[str, bool] = {}
        raw_sources = payload.get("sources")
        if raw_sources is not None:
            if not isinstance(raw_sources, dict):
                raise BadRequest("sources 必须是对象")
            for source_id, use_proxy in raw_sources.items():
                if source_id not in DOWNLOAD_SOURCES:
                    raise BadRequest(f"未知下载源：{source_id}")
                if not isinstance(use_proxy, bool):
                    raise BadRequest("use_proxy 必须是布尔值")
                if use_proxy and not proxy_ready:
                    raise BadRequest("代理未启用或地址端口不完整，不能为下载源启用代理")
                source_updates[source_id] = use_proxy
        with self._config_lock:
            new_config = copy.deepcopy(self.config)
            new_config["proxy"] = settings
            downloads = new_config.get("downloads")
            if not isinstance(downloads, dict):
                downloads = {}
                new_config["downloads"] = downloads
            sources = downloads.get("sources")
            if not isinstance(sources, dict):
                sources = {}
                downloads["sources"] = sources
            if not proxy_ready:
                for source_id in DOWNLOAD_SOURCE_ORDER:
                    entry = sources.get(source_id)
                    if isinstance(entry, dict):
                        entry["use_proxy"] = False
            else:
                for source_id, use_proxy in source_updates.items():
                    sources[source_id] = {"use_proxy": use_proxy}
            self._write_config_locked(new_config)
        if settings.get("password"):
            try:
                os.chmod(self.config_path, 0o600)
            except OSError:
                pass
        self.log.info(
            "proxy settings updated (enabled=%s, type=%s, host=%s:%s)",
            settings["enabled"], settings["type"], settings["host"], settings["port"],
        )
        return self.proxy_view()

    def test_proxy(self, payload: dict[str, Any]) -> dict[str, Any]:
        probe = dict(payload or {})
        probe["enabled"] = True
        settings = self._normalize_proxy(probe, self._proxy_config())
        host = settings["host"]
        port = settings["port"]
        url = proxy_url(settings)
        if not url:
            raise BadRequest("代理地址和端口不完整")
        tcp_host = host[1:-1] if host.startswith("[") and host.endswith("]") else host
        reachable = False
        tcp_error = None
        try:
            with socket.create_connection((tcp_host, port), timeout=6.0):
                reachable = True
        except OSError as exc:
            tcp_error = f"{type(exc).__name__}: {exc}"
        tests: list[dict[str, Any]] = []
        if reachable:
            targets = (
                ("HF-Mirror", "https://hf-mirror.com/"),
                ("ModelScope", "https://www.modelscope.cn/"),
            )

            def probe_target(label: str, target_url: str) -> dict[str, Any]:
                started = time.monotonic()
                try:
                    response = requests.get(
                        target_url,
                        proxies={"http": url, "https": url},
                        timeout=(6, 12),
                        headers={"User-Agent": f"llm-router/{VERSION}"},
                        stream=True,
                        allow_redirects=True,
                    )
                    status = response.status_code
                    response.close()
                    return {
                        "target": label,
                        "ok": status < 500 and status != 407,
                        "status": status,
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                    }
                except Exception as exc:
                    return {
                        "target": label,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"[:400],
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                    }

            with ThreadPoolExecutor(
                max_workers=len(targets), thread_name_prefix="proxy-probe"
            ) as pool:
                tests = list(pool.map(lambda item: probe_target(*item), targets))
        ok = bool(reachable and any(test.get("ok") for test in tests))
        error = None
        if not reachable:
            error = f"无法连接代理 {host}:{port}（{tcp_error or '连接失败'}）"
        elif not ok:
            error = "代理端口可连接，但通过代理访问目标站点失败"
        self.log.info(
            "proxy connectivity test %s:%s: reachable=%s ok=%s",
            host, port, reachable, ok,
        )
        return {
            "ok": ok,
            "reachable": reachable,
            "type": settings["type"],
            "address": f"{host}:{port}",
            "tests": tests,
            "error": error,
        }

    def _source_uses_proxy(self, source_id: str) -> bool:
        if not self.proxy_view()["configured"]:
            return False
        entry = self._download_sources_config().get(source_id)
        return bool(isinstance(entry, dict) and entry.get("use_proxy"))

    def _validate_source_id(self, value: Any) -> str:
        source_id = str(value or "").strip()
        if source_id not in DOWNLOAD_SOURCES:
            raise BadRequest("source 必须是 huggingface、hf-mirror 或 modelscope")
        return source_id

    def _download_root(self) -> Path:
        raw = str(self._downloads_config().get("dir") or "").strip()
        if not raw:
            raise BadRequest("请先在下载模型页设置下载目录")
        path = resolve_path(self.base_dir, os.path.expanduser(raw))
        if path.exists() and not path.is_dir():
            raise BadRequest(f"下载目录不是文件夹：{path}")
        return path

    def downloads_view(self) -> dict[str, Any]:
        proxy = self.proxy_view()
        tools: dict[str, dict[str, Any]] = {}
        for command in ("hf", "modelscope"):
            tool = find_tool(command)
            tools[command] = {
                "command": command,
                "path": str(tool) if tool else None,
                "available": tool is not None,
            }
        raw_dir = str(self._downloads_config().get("dir") or "").strip()
        root = resolve_path(self.base_dir, os.path.expanduser(raw_dir)) if raw_dir else None
        sources_cfg = self._download_sources_config()
        sources = []
        for source_id in DOWNLOAD_SOURCE_ORDER:
            spec = DOWNLOAD_SOURCES[source_id]
            entry = sources_cfg.get(source_id)
            use_proxy = bool(
                proxy["configured"]
                and isinstance(entry, dict)
                and entry.get("use_proxy")
            )
            sources.append(
                {
                    "id": source_id,
                    "label": spec["label"],
                    "tool": spec["tool"],
                    "available": tools[spec["tool"]]["available"],
                    "use_proxy": use_proxy,
                }
            )
        with self.download_jobs_lock:
            jobs = [
                {
                    key: job.get(key)
                    for key in (
                        "id", "source", "source_label", "repo_id", "dir",
                        "files_count", "total_bytes", "status",
                        "started_at", "finished_at", "error",
                    )
                }
                for job in list(self.download_jobs.values())[-5:]
            ]
            last_job_id = next(reversed(self.download_jobs), None) if self.download_jobs else None
        return {
            "dir": str(root) if root else "",
            "dir_configured": bool(root),
            "dir_exists": bool(root and root.is_dir()),
            "proxy_configured": proxy["configured"],
            "sources": sources,
            "tools": tools,
            "jobs": jobs,
            "last_job_id": last_job_id,
            "config_file": self._engine_config_label(),
        }

    def save_download_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = payload or {}
        if "dir" not in payload and "sources" not in payload:
            raise BadRequest("没有可保存的下载设置")
        new_dir: str | None = None
        if "dir" in payload:
            raw = str(payload.get("dir") or "").strip()
            if not raw:
                raise BadRequest("下载目录不能为空")
            if "\x00" in raw:
                raise BadRequest("下载目录不合法")
            path = Path(os.path.expanduser(raw))
            if not path.is_absolute():
                path = self.base_dir / path
            path = path.resolve()
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise BadRequest(f"无法创建下载目录 {path}：{exc}") from exc
            if not path.is_dir():
                raise BadRequest(f"下载目录不是文件夹：{path}")
            new_dir = str(path)
        source_updates: dict[str, bool] = {}
        if "sources" in payload:
            raw_sources = payload.get("sources")
            if not isinstance(raw_sources, dict):
                raise BadRequest("sources 必须是对象")
            for source_id, entry in raw_sources.items():
                if source_id not in DOWNLOAD_SOURCES:
                    raise BadRequest(f"未知下载源：{source_id}")
                if not isinstance(entry, dict):
                    raise BadRequest("每个下载源设置必须是对象")
                use_proxy = entry.get("use_proxy")
                if not isinstance(use_proxy, bool):
                    raise BadRequest("use_proxy 必须是布尔值")
                if use_proxy and not self.proxy_view()["configured"]:
                    raise BadRequest("代理尚未启用或配置不完整，不能为下载源启用代理")
                source_updates[source_id] = use_proxy
        with self._config_lock:
            new_config = copy.deepcopy(self.config)
            downloads = new_config.get("downloads")
            if not isinstance(downloads, dict):
                downloads = {}
                new_config["downloads"] = downloads
            if new_dir is not None:
                downloads["dir"] = new_dir
            sources = downloads.get("sources")
            if not isinstance(sources, dict):
                sources = {}
                downloads["sources"] = sources
            for source_id, use_proxy in source_updates.items():
                sources[source_id] = {"use_proxy": use_proxy}
            self._write_config_locked(new_config)
        if new_dir is not None:
            self.log.info("download directory set to %s", new_dir)
        if source_updates:
            self.log.info("download source proxy settings updated: %s", source_updates)
        return self.downloads_view()

    def _source_tool(self, source_id: str) -> Path:
        command = DOWNLOAD_SOURCES[source_id]["tool"]
        tool = find_tool(command)
        if tool is None:
            raise GatewayError(
                f"未找到 {command} 命令；请重新启动 llm-router 以安装模型下载工具"
            )
        return tool

    def _cli_env(self, source_id: str, use_proxy: bool) -> dict[str, str]:
        proxy_cfg = self._proxy_config()
        if use_proxy and not self.proxy_view()["configured"]:
            raise BadRequest("代理尚未启用或配置不完整")
        env = os.environ.copy()
        apply_proxy_env(env, proxy_cfg, use_proxy)
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if source_id == "hf-mirror":
            env["HF_ENDPOINT"] = "https://hf-mirror.com"
        else:
            env.pop("HF_ENDPOINT", None)
        return env

    def _run_cli(
        self, args: list[str], env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                cwd=str(self.base_dir),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise GatewayError(f"未找到命令：{args[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GatewayError(
                f"命令执行超时（{int(timeout)}s）：{Path(str(args[0])).name}"
            ) from exc
        except OSError as exc:
            raise GatewayError(f"命令启动失败：{exc}") from exc

    def _redact_secrets(self, text: str) -> str:
        value = str(text or "")
        password = str(self._proxy_config().get("password") or "")
        if password:
            for variant in {password, quote(password, safe="")}:
                if variant:
                    value = value.replace(variant, "***")
        return re.sub(r"(://[^/\s:@]+):[^/\s@]+@", r"\1:***@", value)

    def _search_source(
        self, source_id: str, query: str, limit: int
    ) -> dict[str, Any]:
        spec = DOWNLOAD_SOURCES[source_id]
        started = time.monotonic()
        use_proxy = self._source_uses_proxy(source_id)
        try:
            tool = self._source_tool(source_id)
            env = self._cli_env(source_id, use_proxy)
            total: int | None = None
            if spec["kind"] == "hf":
                args = [
                    str(tool), "models", "list",
                    "--search", query,
                    "--limit", str(limit),
                    "--json",
                    "--expand",
                    "downloads,likes,lastModified,createdAt,pipeline_tag,tags,"
                    "library_name,safetensors",
                ]
                proc = self._run_cli(args, env, 60.0)
                if proc.returncode != 0 and "expand" in (proc.stderr or "").lower():
                    # Older hf releases do not support --expand; the default
                    # JSON still carries the main listing fields.
                    proc = self._run_cli(
                        [arg for arg in args if arg != "--expand"
                         and not arg.startswith("downloads,likes,")],
                        env,
                        60.0,
                    )
                if proc.returncode != 0:
                    message = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
                    raise GatewayError("Hugging Face 搜索失败：" + message[-600:])
                try:
                    raw = json.loads(proc.stdout or "[]")
                except ValueError as exc:
                    raise GatewayError("Hugging Face 搜索输出不是 JSON") from exc
                if not isinstance(raw, list):
                    raise GatewayError("Hugging Face 搜索输出格式不正确")
                items = [
                    hf_search_item(item, source_id)
                    for item in raw
                    if isinstance(item, dict) and item.get("id")
                ]
            else:
                args = [
                    str(tool), "list",
                    "--repo-type", "model",
                    "--search", query,
                    "--page-size", str(min(limit, 50)),
                ]
                proc = self._run_cli(args, env, 60.0)
                if proc.returncode != 0:
                    message = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
                    raise GatewayError("ModelScope 搜索失败：" + message[-600:])
                rows, total = parse_modelscope_table(proc.stdout or "")
                items = [
                    {
                        "source": source_id,
                        "source_label": spec["label"],
                        "repo_id": row["repo_id"],
                        "pipeline_tag": None,
                        "library": None,
                        "downloads": row.get("downloads"),
                        "likes": row.get("likes"),
                        "license": row.get("license"),
                        "params": None,
                        "updated_at": None,
                        "created_at": None,
                        "tags": [],
                        "visibility": row.get("visibility"),
                    }
                    for row in rows
                ]
            return {
                "id": source_id,
                "label": spec["label"],
                "ok": True,
                "items": items,
                "count": len(items),
                "total": total,
                "error": None,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            }
        except Exception as exc:
            message = self._redact_secrets(str(exc) or type(exc).__name__)
            self.log.warning("download search %s failed: %s", source_id, message)
            return {
                "id": source_id,
                "label": spec["label"],
                "ok": False,
                "items": [],
                "count": 0,
                "total": None,
                "error": message,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            }

    def search_downloads(
        self, query: Any, limit: Any = None, sources: Any = None
    ) -> dict[str, Any]:
        text = str(query or "").strip()
        if not text:
            raise BadRequest("请输入搜索关键词")
        if len(text) > 200:
            raise BadRequest("搜索关键词过长")
        parsed_limit = parse_int(limit)
        limit_value = max(1, min(parsed_limit or 30, DOWNLOAD_LIMIT_MAX))
        if sources is None:
            selected = list(DOWNLOAD_SOURCE_ORDER)
        else:
            raw_sources = str(sources).strip()
            if not raw_sources:
                raise BadRequest("请至少选择一个搜索来源")
            selected = []
            for part in raw_sources.split(","):
                source_id = part.strip()
                if not source_id:
                    continue
                if source_id not in DOWNLOAD_SOURCES:
                    raise BadRequest(f"未知下载源：{source_id}")
                if source_id not in selected:
                    selected.append(source_id)
            if not selected:
                raise BadRequest("请至少选择一个搜索来源")
        selected_set = set(selected)
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=len(selected), thread_name_prefix="download-search"
        ) as pool:
            futures = {
                pool.submit(self._search_source, source_id, text, limit_value): source_id
                for source_id in selected
            }
            for future in as_completed(futures):
                source_id = futures[future]
                try:
                    results[source_id] = future.result()
                except Exception as exc:
                    message = self._redact_secrets(str(exc) or type(exc).__name__)
                    results[source_id] = {
                        "id": source_id,
                        "label": DOWNLOAD_SOURCES[source_id]["label"],
                        "ok": False,
                        "items": [],
                        "count": 0,
                        "total": None,
                        "error": message,
                        "elapsed_ms": 0,
                    }
        ordered = [
            results[source_id]
            for source_id in DOWNLOAD_SOURCE_ORDER
            if source_id in selected_set
        ]
        items: list[dict[str, Any]] = []
        for result in ordered:
            items.extend(result.get("items") or [])
        return {
            "query": text,
            "limit": limit_value,
            "sources": [
                {
                    key: result.get(key)
                    for key in ("id", "label", "ok", "count", "total", "error", "elapsed_ms")
                }
                for result in ordered
            ],
            "items": items,
        }

    def _list_repo_files(
        self, source_id: str, repo_id: str, use_proxy: bool
    ) -> list[dict[str, Any]]:
        spec = DOWNLOAD_SOURCES[source_id]
        tool = self._source_tool(source_id)
        env = self._cli_env(source_id, use_proxy)
        raw: Any
        if spec["kind"] == "hf":
            proc = self._run_cli(
                [str(tool), "models", "list", repo_id, "-R", "--json"],
                env,
                120.0,
            )
            if proc.returncode != 0:
                message = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
                raise GatewayError("读取仓库文件列表失败：" + message[-600:])
            try:
                raw = json.loads(proc.stdout or "[]")
            except ValueError as exc:
                raise GatewayError("仓库文件列表不是 JSON") from exc
            if isinstance(raw, dict):
                raw = raw.get("siblings") or raw.get("files") or []
            entries = raw if isinstance(raw, list) else []
            files: list[dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("type") or "").lower() == "directory":
                    continue
                # ``hf models list -R`` marks directories only with tree_id and
                # omits size; regular files always carry a size.
                if entry.get("size") is None:
                    continue
                safe = safe_repo_file(entry.get("path"))
                if not safe:
                    continue
                files.append({"path": safe, "size": parse_int(entry.get("size")) or 0})
        else:
            python = tool_python(tool)
            code = (
                "import json, sys\n"
                "from modelscope.hub.api import HubApi\n"
                "files = HubApi().get_model_files(sys.argv[1], recursive=True)\n"
                "print(json.dumps(files, ensure_ascii=False))\n"
            )
            proc = self._run_cli(
                [python, "-c", code, repo_id], env, 120.0
            )
            if proc.returncode != 0:
                message = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
                raise GatewayError("读取 ModelScope 仓库文件列表失败：" + message[-600:])
            try:
                raw = json.loads(proc.stdout or "[]")
            except ValueError as exc:
                raise GatewayError("ModelScope 仓库文件列表不是 JSON") from exc
            entries = raw if isinstance(raw, list) else []
            files = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                safe = safe_repo_file(entry.get("Path") or entry.get("path"))
                if not safe:
                    continue
                size = entry.get("Size", entry.get("size"))
                files.append({"path": safe, "size": parse_int(size) or 0})
        deduped: dict[str, dict[str, Any]] = {}
        for entry in files:
            deduped[entry["path"]] = entry
        return list(deduped.values())

    def plan_download(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_id = self._validate_source_id(payload.get("source"))
        repo_id = normalize_repo_id(payload.get("repo_id"))
        root = self._download_root()
        use_proxy = self._source_uses_proxy(source_id)
        files = self._list_repo_files(source_id, repo_id, use_proxy)
        if not files:
            raise BadRequest("仓库中没有可下载的文件")
        target = root / repo_id
        items = []
        existing = []
        for entry in files:
            full_path = target / entry["path"]
            exists = full_path.is_file()
            item = {
                "path": entry["path"],
                "size": entry["size"],
                "exists": exists,
            }
            items.append(item)
            if exists:
                existing.append(item)
        total_bytes = sum(item["size"] for item in items)
        missing_bytes = sum(item["size"] for item in items if not item["exists"])
        return {
            "source": source_id,
            "source_label": DOWNLOAD_SOURCES[source_id]["label"],
            "repo_id": repo_id,
            "dir": str(root),
            "target_dir": str(target),
            "files": items,
            "file_count": len(items),
            "total_bytes": total_bytes,
            "missing_bytes": missing_bytes,
            "existing_count": len(existing),
            "use_proxy": use_proxy,
        }

    def start_download(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_id = self._validate_source_id(payload.get("source"))
        repo_id = normalize_repo_id(payload.get("repo_id"))
        with self.download_jobs_lock:
            if any(
                job.get("status") == "running" for job in self.download_jobs.values()
            ):
                raise ModelBusy("已有下载任务进行中，请等待完成或取消")
        plan = self.plan_download(payload)
        known = {item["path"]: item for item in plan["files"]}
        requested = payload.get("files")
        if requested is None:
            selected = [item for item in plan["files"] if not item["exists"]]
        else:
            if not isinstance(requested, list):
                raise BadRequest("files 必须是文件路径数组")
            selected = []
            seen: set[str] = set()
            for raw in requested:
                safe = safe_repo_file(raw)
                if not safe:
                    raise BadRequest(f"文件路径不合法：{raw!r}")
                item = known.get(safe)
                if item is None:
                    raise BadRequest(f"文件不在仓库中：{safe}")
                if safe not in seen:
                    seen.add(safe)
                    selected.append(item)
        if not selected:
            if plan["existing_count"]:
                raise BadRequest("所选文件均已存在，未开始下载")
            raise BadRequest("没有选择要下载的文件")
        existing = [item for item in selected if item["exists"]]
        if existing:
            preview = "、".join(item["path"] for item in existing[:3])
            suffix = f" 等 {len(existing)} 个" if len(existing) > 3 else ""
            raise BadRequest(f"文件已存在，未开始下载：{preview}{suffix}")
        target = Path(plan["target_dir"])
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BadRequest(f"无法创建下载目录 {target}：{exc}") from exc
        tool = self._source_tool(source_id)
        env = self._cli_env(source_id, plan["use_proxy"])
        command = [
            str(tool), "download", repo_id,
            "--local-dir", str(target),
            "--max-workers", "4",
        ] + [item["path"] for item in selected]
        job_id = uuid.uuid4().hex[:12]
        log_dir = self.base_dir / "logs" / "downloads"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GatewayError(f"无法创建下载日志目录：{exc}") from exc
        log_path = log_dir / f"{job_id}.log"
        job = {
            "id": job_id,
            "kind": "download",
            "source": source_id,
            "source_label": DOWNLOAD_SOURCES[source_id]["label"],
            "repo_id": repo_id,
            "dir": str(Path(plan["dir"])),
            "target_dir": str(target),
            "files": [item["path"] for item in selected],
            "files_count": len(selected),
            "total_bytes": sum(item["size"] for item in selected),
            "status": "running",
            "started_at": iso_now(),
            "finished_at": None,
            "error": None,
            "cancel_requested": False,
            "exit_code": None,
            "log_file": str(log_path.relative_to(self.base_dir)),
        }
        with self.download_jobs_lock:
            if any(
                entry.get("status") == "running"
                for entry in self.download_jobs.values()
            ):
                raise ModelBusy("已有下载任务进行中，请等待完成或取消")
            self.download_jobs[job_id] = job
            self.download_logs[job_id] = log_path
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                proc = subprocess.Popen(
                    command,
                    cwd=str(self.base_dir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            with self.download_jobs_lock:
                job["status"] = "failed"
                job["finished_at"] = iso_now()
                job["error"] = f"无法启动下载命令：{exc}"
            return self.download_job_view(job_id) or job
        with self.download_jobs_lock:
            self.download_procs[job_id] = proc
        worker = threading.Thread(
            target=self._download_worker,
            args=(job_id, proc, target, [item["path"] for item in selected]),
            name=f"download-{job_id}",
            daemon=True,
        )
        worker.start()
        self.log.info(
            "download started %s (%s/%s, %d files, %.1f MB)",
            job_id, source_id, repo_id, len(selected),
            sum(item["size"] for item in selected) / 1_000_000,
        )
        return self.download_job_view(job_id) or job

    def _download_worker(
        self,
        job_id: str,
        proc: subprocess.Popen[Any],
        target: Path,
        files: list[str],
    ) -> None:
        try:
            return_code = proc.wait()
        except Exception:
            return_code = -1
        missing = [name for name in files if not (target / name).is_file()]
        with self.download_jobs_lock:
            job = self.download_jobs.get(job_id)
            if job is None:
                return
            cancelled = bool(job.get("cancel_requested"))
            job["exit_code"] = return_code
            job["finished_at"] = iso_now()
            if cancelled:
                job["status"] = "cancelled"
                job["error"] = "下载已取消"
            elif return_code == 0 and not missing:
                job["status"] = "success"
                job["error"] = None
            else:
                job["status"] = "failed"
                if return_code != 0:
                    job["error"] = f"下载命令退出码 {return_code}"
                else:
                    preview = "、".join(missing[:3])
                    suffix = f" 等 {len(missing)} 个" if len(missing) > 3 else ""
                    job["error"] = f"命令已结束，但缺少文件：{preview}{suffix}"
            self.download_procs.pop(job_id, None)
        self.log.info(
            "download finished %s: status=%s exit=%s missing=%d",
            job_id, job["status"], return_code, len(missing),
        )

    def download_job_view(self, job_id: str) -> dict[str, Any] | None:
        with self.download_jobs_lock:
            job = self.download_jobs.get(job_id)
            if job is None:
                return None
            data = json.loads(json.dumps(job))
        log_path = self.download_logs.get(job_id)
        data["log_tail"] = (
            self._redact_secrets(tail_file(log_path, 120)) if log_path else ""
        )
        return data

    def cancel_download(self, job_id: str) -> dict[str, Any] | None:
        key = str(job_id or "").strip()
        if not key:
            raise BadRequest("缺少下载任务 id")
        with self.download_jobs_lock:
            job = self.download_jobs.get(key)
            if job is None:
                return None
            if job.get("status") != "running":
                proc = None
            else:
                job["cancel_requested"] = True
                proc = self.download_procs.get(key)
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                self.log.warning("cannot terminate download %s: %s", key, exc)
            threading.Thread(
                target=self._force_kill_download,
                args=(proc,),
                name=f"download-kill-{key}",
                daemon=True,
            ).start()
        return self.download_job_view(key)

    @staticmethod
    def _force_kill_download(proc: subprocess.Popen[Any]) -> None:
        time.sleep(3.0)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass

    def cancel_all_downloads(self) -> None:
        with self.download_jobs_lock:
            procs = list(self.download_procs.values())
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass

    def default_sampling(self, engine_id: str) -> dict[str, Any]:
        kind = self._short_kind(self._engine_kind(engine_id))
        reference = LLAMA_REFERENCE_MODEL if kind == "llama" else KVMEM_REFERENCE_MODEL if kind == "kvmem" else ""
        saved = self.params.get(reference) if reference else {}
        return dict(saved or DEFAULT_SAMPLING.get(kind) or {})

    def manage_engines(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for engine_id, engine_cfg in self.config.get("engines", {}).items():
            if not isinstance(engine_cfg, dict):
                continue
            kind = self._engine_kind(engine_id)
            models_cfg = engine_cfg.get("models") if isinstance(engine_cfg.get("models"), dict) else {}
            if kind == "llama_dir":
                out[engine_id] = {
                    "engine": engine_id,
                    "kind": "llama",
                    "label": "llama.cpp",
                    "models_dir": models_cfg.get("dir"),
                    "preset": models_cfg.get("preset"),
                    "defaults": llama_preset_defaults(engine_cfg),
                    "cache_types": llama_cache_type_choices(engine_cfg),
                    "default_params": self.default_sampling(engine_id),
                }
            elif kind == "kvmem_dir":
                template_path, template = kvmem_reference_config(engine_cfg)
                out[engine_id] = {
                    "engine": engine_id,
                    "kind": "kvmem",
                    "label": "KVMem（polystrata）",
                    "models_dir": models_cfg.get("dir"),
                    "template": str(template_path) if template_path else None,
                    "defaults": {
                        "args": list((template or {}).get("args") or []),
                        "vision_gpu": False,
                    },
                    "default_params": self.default_sampling(engine_id),
                }
            elif kind == "ninfer_dir":
                template_path, template = ninfer_reference_config(engine_cfg)
                out[engine_id] = {
                    "engine": engine_id,
                    "kind": "ninfer",
                    "label": "NInfer（KVMem）",
                    "models_dir": models_cfg.get("dir"),
                    "template": str(template_path) if template_path else None,
                    "defaults": {
                        "args": list((template or {}).get("args") or []),
                        "vision_gpu": False,
                    },
                    "default_params": self.default_sampling(engine_id),
                }
        return out

    def model_manage_view(self) -> dict[str, Any]:
        with self.model_jobs_lock:
            jobs = [json.loads(json.dumps(job)) for job in list(self.model_jobs.values())[-10:]]
        return {
            "engines": self.manage_engines(),
            "param_schema": {
                kind: sampling_schema(kind) for kind in ("llama", "kvmem", "ninfer")
            },
            "smoke_image": str(self.smoke_image_path),
            "engine_defaults": {
                "kv": {"method": "q8_0/q8_0 (k8v8)", "fallback": "engine default"},
                "vision_gpu": False,
            },
            "jobs": jobs,
        }

    def get_model_job(self, job_id: str) -> dict[str, Any] | None:
        with self.model_jobs_lock:
            job = self.model_jobs.get(job_id)
            return json.loads(json.dumps(job)) if job is not None else None

    def _job_step(self, job: dict[str, Any], key: str, status: str, detail: str | None = None) -> None:
        with self.model_jobs_lock:
            for step in job.get("steps") or []:
                if step.get("key") == key:
                    step["status"] = status
                    if detail is not None:
                        step["detail"] = str(detail)
                    break

    def _job_finish(self, job: dict[str, Any], status: str, error: str | None = None) -> None:
        with self.model_jobs_lock:
            job["status"] = status
            job["error"] = error
            job["finished_at"] = iso_now()

    def start_model_add(self, payload: dict[str, Any]) -> dict[str, Any]:
        engine_id = str(payload.get("engine") or "").strip()
        if not engine_id or engine_id not in self.config.get("engines", {}):
            raise BadRequest("engine must name a configured engine")
        kind = self._engine_kind(engine_id)
        if kind not in ("llama_dir", "kvmem_dir", "ninfer_dir"):
            raise BadRequest(f"engine {engine_id!r} cannot add models")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise BadRequest('send {"engine": "...", "model": "/path/model.gguf|.ninfer"}')
        vision = payload.get("vision")
        if vision is not None and not isinstance(vision, str):
            raise BadRequest("vision must be a path or null")
        params = payload.get("params")
        if params is not None and not isinstance(params, dict):
            raise BadRequest("params must be an object")
        name = payload.get("name")
        if name is not None and not isinstance(name, str):
            raise BadRequest("name must be a string")
        smoke = payload.get("smoke", True) is not False
        vision_path = (vision or "").strip() or None
        vision_device = payload.get("vision_device", "cpu")
        if vision_device is None:
            vision_device = "cpu"
        if not isinstance(vision_device, str) or vision_device.strip().lower() not in ("cpu", "gpu"):
            raise BadRequest('vision_device must be "cpu" or "gpu"')
        vision_gpu = vision_device.strip().lower() == "gpu"
        context_length = payload.get("context_length")
        if context_length is not None:
            if isinstance(context_length, bool) or not isinstance(context_length, int):
                raise BadRequest("context_length must be an integer")
            if context_length < 256:
                raise BadRequest("context_length must be at least 256")
        cache_type_k = payload.get("cache_type_k")
        cache_type_v = payload.get("cache_type_v")
        for label, value in (("cache_type_k", cache_type_k), ("cache_type_v", cache_type_v)):
            if value is not None and not isinstance(value, str):
                raise BadRequest(f"{label} must be a string")
        if kind == "kvmem_dir" and (
            context_length is not None or cache_type_k is not None or cache_type_v is not None
        ):
            raise BadRequest("KVMem 引擎固定使用模型上下文与 q8 量化，不能设置上下文或 K/V 精度")
        if kind == "ninfer_dir":
            if vision_path:
                raise BadRequest("NInfer 模型暂不支持在控制台配置视觉 mmproj")
            if context_length is not None or cache_type_k is not None or cache_type_v is not None:
                raise BadRequest("NInfer 的上下文与 KV 参数由模板配置决定，不能单独设置")
        steps = [
            {"key": "paths", "label": "检查模型路径", "status": "pending", "detail": ""},
            {"key": "metadata", "label": "读取 GGUF 元数据", "status": "pending", "detail": ""},
            {"key": "kv", "label": "选择 KV 精度", "status": "pending", "detail": ""},
            {"key": "config", "label": "写入引擎配置", "status": "pending", "detail": ""},
            {"key": "start", "label": "启动引擎并加载模型", "status": "pending", "detail": ""},
            {"key": "smoke", "label": "冒烟测试", "status": "pending", "detail": ""},
        ]
        if kind == "ninfer_dir":
            steps[1]["label"] = "检查 .ninfer 模型"
            steps[2]["label"] = "复制模板参数"
        if vision_path:
            steps.append(
                {"key": "vision_smoke", "label": "视觉冒烟测试", "status": "pending", "detail": ""}
            )
        job = {
            "id": uuid.uuid4().hex[:12],
            "kind": "add",
            "engine": engine_id,
            "model": model.strip(),
            "vision": vision_path,
            "vision_device": "gpu" if vision_gpu else "cpu",
            "context_length": context_length,
            "cache_type_k": (cache_type_k or "").strip() or None,
            "cache_type_v": (cache_type_v or "").strip() or None,
            "name": (name or "").strip() or None,
            "params": params,
            "smoke": smoke,
            "status": "running",
            "started_at": iso_now(),
            "finished_at": None,
            "error": None,
            "config_path": None,
            "rollback": False,
            "steps": steps,
        }
        with self.model_jobs_lock:
            if any(item.get("status") == "running" for item in self.model_jobs.values()):
                raise ModelBusy("another model add operation is in progress")
            self.model_jobs[job["id"]] = job
        thread = threading.Thread(
            target=self._model_add_worker,
            args=(job, engine_id, kind, Path(model.strip()).expanduser(),
                  Path(vision_path).expanduser() if vision_path else None, smoke,
                  vision_gpu),
            name=f"model-add-{job['id']}",
            daemon=True,
        )
        thread.start()
        return self.get_model_job(job["id"]) or job

    def _export_kvmem_tokenizer(
        self,
        engine_cfg: dict[str, Any],
        tool: Path,
        model: Path,
        job: dict[str, Any],
    ) -> Path:
        self._job_step(job, "config", "running", "导出 tokenizer…")
        command = engine_cfg.get("command")
        python = sys.executable
        if isinstance(command, list) and command and "python" in Path(str(command[0])).name.lower():
            python = str(command[0])
        source = Path(str(engine_cfg.get("cwd") or tool.parent.parent))
        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from assets import tokenizer_path\n"
            "print(tokenizer_path(Path(sys.argv[2]), Path(sys.argv[3]), prepare=True))\n"
        )
        env = os.environ.copy()
        extra = engine_cfg.get("env")
        if isinstance(extra, dict):
            env.update({str(key): str(value) for key, value in extra.items()})
        try:
            proc = subprocess.run(
                [python, "-c", code, str(tool.parent), str(model), str(source)],
                cwd=str(source) if source.is_dir() else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BadRequest(f"tokenizer 导出失败：{exc}") from exc
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise BadRequest("tokenizer 导出失败：" + message[-800:])
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        tokenizer = Path(lines[-1]) if lines else Path("")
        if not tokenizer.is_dir() or not (tokenizer / "vocab.json").is_file():
            raise BadRequest(f"tokenizer 导出结果无效：{tokenizer}")
        return tokenizer

    @staticmethod
    def _fit_kvmem_budget(args: list[str], context: int) -> list[str]:
        """Keep the template's KVMem tuning when it fits the model's context."""
        def number(flag: str, default: int) -> int:
            if flag in args[:-1]:
                try:
                    return int(args[args.index(flag) + 1])
                except (TypeError, ValueError):
                    return default
            return default

        block = max(1, number("--kvmem-block-tokens", 64))
        budget = max(128, number("--kvmem-budget", 32768))
        reserve = max(block, number("--kvmem-gen-reserve", 4096))
        if budget + reserve <= context:
            return args
        reserve = max(block, min(reserve, (context // 4) // block * block))
        budget = max(128, min(budget, ((context - reserve) // block) * block))
        if budget + reserve > context:
            reserve = max(block, (context // 5) // block * block)
            budget = max(128, ((context - reserve) // block) * block)
        if budget + reserve > context or budget < 128:
            raise BadRequest(
                f"上下文 {context} 太小，无法容纳 KVMem 的预算与生成保留（至少需要 5 × block）"
            )
        args = _config_arg_replace(args, "--kvmem-budget", budget)
        args = _config_arg_replace(args, "--kvmem-gen-reserve", reserve)
        prefill = number("--prompt-batch", 512)
        if prefill > reserve:
            args = _config_arg_replace(args, "--prompt-batch", max(block, reserve))
        return args

    @staticmethod
    def _cache_startup_error(message: str) -> bool:
        text = str(message).lower()
        if any(marker in text for marker in (
            "out of memory",
            "cudamalloc failed",
            "failed to allocate",
            "显存不足",
        )):
            return False
        if "cache-type" in text or "cache type" in text or "quantized" in text:
            return True
        return "kv" in text and any(
            marker in text for marker in ("invalid choice", "unrecognized", "unsupported", "not supported")
        )

    def _smoke_completion(
        self,
        entry: Model,
        messages: list[dict[str, Any]],
        job: dict[str, Any],
    ) -> tuple[str, int]:
        manager = self.engines.get(entry.engine)
        if manager is None:
            raise EngineFailure(f"no engine configured for {entry.engine}")
        payload = {
            "model": entry.engine_model,
            "messages": messages,
            "max_tokens": 128,
            "temperature": 0,
            "stream": False,
        }
        status, body = manager._json_request(
            "POST", "/v1/chat/completions", entry=entry, json_body=payload, timeout=300
        )
        if status != 200 or not isinstance(body, dict):
            raise EngineFailure(
                f"冒烟测试请求失败（HTTP {status}）：{manager._error_from_body(status, body)}"
            )
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        message = message if isinstance(message, dict) else {}
        text = str(message.get("content") or "").strip()
        reasoning = str(message.get("reasoning_content") or "").strip()
        if not text and not reasoning:
            raise EngineFailure("冒烟测试返回了空内容")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        try:
            tokens = int(usage.get("prompt_tokens") or 0)
        except (TypeError, ValueError):
            tokens = 0
        return text or reasoning, tokens

    def _model_add_ninfer_worker(
        self,
        job: dict[str, Any],
        engine_id: str,
        model: Path,
        smoke: bool,
    ) -> None:
        """Add a .ninfer model as a single-model NInfer process.

        NInfer's router mode refuses KVMem, so each model gets its own
        ninfer-*.json (artifact, serve options and env) and the gateway starts
        the engine for that one model, exactly like a KVMem strata config.
        """
        engine_cfg = self.config.get("engines", {}).get(engine_id) or {}
        config_file: Path | None = None
        name: str | None = None
        try:
            self._job_step(job, "paths", "running")
            if not model.is_file():
                raise BadRequest(f"模型路径不存在：{model}")
            if model.suffix.lower() != ".ninfer":
                raise BadRequest("NInfer 模型路径必须是 .ninfer 文件")
            self._job_step(job, "paths", "ok", f"{model}；NInfer 单模型模式")

            self._job_step(job, "metadata", "running")
            name = slugify_model_id(job.get("name") or model.stem)
            if self.registry.resolve(name) is not None:
                raise BadRequest(f"模型 ID {name} 已存在")
            job["name"] = name
            template_path, template = ninfer_reference_config(engine_cfg)
            if template is None or template_path is None:
                raise BadRequest(
                    "找不到 NInfer 模板配置；请在管理页把 NInfer 模板指向一份可用的 ninfer-*.json"
                )
            self._job_step(
                job, "metadata", "ok",
                f"ID {name}；模板 {template_path.name}",
            )

            self._job_step(job, "kv", "running")
            self._job_step(
                job, "kv", "ok",
                "上下文、KVMem 与 KV 量化均继承模板；单模型模式无需 GGUF 元数据",
            )

            self._job_step(job, "config", "running")
            params = job.get("params") or self.default_sampling(engine_id)
            cleaned = self.params.replace(name, params, "ninfer")
            if not cleaned:
                cleaned = self.params.replace(name, self.default_sampling(engine_id), "ninfer")

            models_cfg = engine_cfg.get("models") if isinstance(engine_cfg.get("models"), dict) else {}
            config_dir = Path(str(models_cfg.get("dir") or ""))
            config_file = config_dir / f"ninfer-{slugify_model_id(name).lower()}.json"
            if config_file.exists():
                raise BadRequest(f"配置已存在：{config_file}")
            new_cfg = json.loads(json.dumps(template))
            new_cfg["model_name"] = name
            new_cfg["model_id"] = name
            new_cfg["display_name"] = model.stem
            new_cfg["aliases"] = []
            new_cfg["model"] = str(model)
            new_cfg.pop("artifact", None)
            new_cfg["notes"] = f"NInfer 单模型模式；参数复制自 {template_path.name}"
            if not new_cfg.get("exe"):
                command = engine_cfg.get("command")
                if isinstance(command, list) and command:
                    new_cfg["exe"] = str(command[0])
            if not new_cfg.get("cwd"):
                new_cfg["cwd"] = str(engine_cfg.get("cwd") or "")
            args = [str(item) for item in (new_cfg.get("args") or [])]
            args = _config_arg_replace(args, "--model-id", name)
            if "--model-id" not in args[:-1]:
                args = ["--model-id", name, *args]
            for router_flag in ("--host", "--port", "--api-key", "--models-dir", "--models-preset"):
                args = _config_arg_remove(args, router_flag)
            new_cfg["args"] = args
            write_json_atomic(config_file, new_cfg)
            job["config_path"] = str(config_file)
            self._job_step(
                job, "config", "ok",
                f"已写入 {config_file}；参数复制自 {template_path.name}",
            )

            self._job_step(job, "start", "running", "正在停止当前引擎…")
            self.ensure_no_async()
            with self.cond:
                engine_alive = bool(self.engine is not None and self.engine.alive())
            if engine_alive:
                self.stop_engine()
            self.registry.reload()
            entry = self.registry.resolve(name)
            if entry is None:
                raise EngineFailure("写入配置后注册表仍未发现该模型")
            if not entry.available:
                raise EngineFailure(f"引擎配置不可用：{entry.problem}")
            self._job_step(job, "start", "running", "正在启动引擎并加载模型…")
            self.preload(name)
            self._job_step(
                job, "start", "ok",
                f"{entry.engine} 已加载 {entry.id}（引擎模型 {entry.engine_model}）",
            )

            if smoke:
                self._job_step(job, "smoke", "running", "发送文本请求…")
                text, text_tokens = self._smoke_completion(
                    entry, [{"role": "user", "content": SMOKE_PROMPT}], job
                )
                self._job_step(
                    job, "smoke", "ok",
                    f"文本正常（prompt {text_tokens} tokens）：{text[:80]}",
                )
            else:
                self._job_step(job, "smoke", "ok", "跳过（未勾选冒烟测试）")
            self._job_finish(job, "ok")
            self.log.info("NInfer model add %s (%s) finished", job["id"], name)
        except Exception as exc:
            message = exc.message if isinstance(exc, GatewayError) else str(exc)
            self.log.exception("NInfer model add %s failed", job["id"])
            with self.model_jobs_lock:
                for step in job.get("steps") or []:
                    if step.get("status") == "running":
                        step["status"] = "fail"
                        if message:
                            step["detail"] = message[:300]
                if not any(step.get("status") == "fail" for step in job.get("steps") or []):
                    for step in job.get("steps") or []:
                        if step.get("status") == "pending":
                            step["status"] = "fail"
                            break
            self._rollback_model_add(job, "ninfer", name, None, config_file)
            self._job_finish(job, "failed", message)
        finally:
            with self.model_jobs_lock:
                job["status"] = job.get("status") if job.get("status") != "running" else "failed"
                job["finished_at"] = job.get("finished_at") or iso_now()

    def _model_add_worker(
        self,
        job: dict[str, Any],
        engine_id: str,
        kind: str,
        model: Path,
        vision: Path | None,
        smoke: bool,
        vision_gpu: bool = False,
    ) -> None:
        if kind == "ninfer_dir":
            self._model_add_ninfer_worker(job, engine_id, model, smoke)
            return
        engine_cfg = self.config.get("engines", {}).get(engine_id) or {}
        short = self._short_kind(kind)
        preset_path: Path | None = None
        config_file: Path | None = None
        name: str | None = None
        cache_override = False
        try:
            self._job_step(job, "paths", "running")
            if not model.is_file():
                raise BadRequest(f"模型路径不存在：{model}")
            if model.suffix.lower() != ".gguf":
                raise BadRequest("模型路径不是 .gguf 文件")
            if vision is not None:
                if not vision.is_file():
                    raise BadRequest(f"视觉路径不存在：{vision}")
                if vision.suffix.lower() != ".gguf":
                    raise BadRequest("视觉路径不是 .gguf 文件")
            self._job_step(
                job, "paths", "ok",
                str(model) + (f"；视觉 {vision}" if vision is not None else "；无视觉"),
            )

            self._job_step(job, "metadata", "running")
            info = gguf_model_info(model)
            if info is None:
                raise BadRequest("无法读取 GGUF 头：文件不是有效的 GGUF 模型")
            gguf_context = int(info.get("context_length") or 0)
            if gguf_context <= 0:
                raise BadRequest("GGUF 中没有 context_length；无法确定上下文上限")
            requested_context = job.get("context_length")
            if requested_context is None:
                context = gguf_context
                context_note = f"上下文 {context:,}（GGUF 上限）"
            else:
                context = int(requested_context)
                if context > gguf_context:
                    raise BadRequest(
                        f"上下文长度 {context:,} 超过 GGUF 上限 {gguf_context:,}"
                    )
                context_note = f"上下文 {context:,}（GGUF 上限 {gguf_context:,}）"
            name = slugify_model_id(job.get("name") or model.stem)
            if self.registry.resolve(name) is not None:
                raise BadRequest(f"模型 ID {name} 已存在")
            vision_flags = None
            if vision is not None:
                vision_flags = gguf_bool_metadata(
                    vision, {"clip.has_vision_encoder", "clip.has_audio_encoder"}
                )
                if vision_flags is None:
                    raise BadRequest("无法读取视觉文件头")
                if not vision_flags.get("clip.has_vision_encoder"):
                    raise BadRequest("视觉文件没有 clip.has_vision_encoder（不是视觉 mmproj）")
            job["name"] = name
            self._job_step(
                job, "metadata", "ok",
                f"架构 {info.get('architecture') or '未知'}；{context_note}；ID {name}"
                + ("；视觉 mmproj 有效" if vision is not None else ""),
            )

            self._job_step(job, "kv", "running")
            help_text = engine_help_text(engine_cfg)
            cache_k = "q8_0"
            cache_v = "q8_0"
            explicit_cache = False
            if short == "llama":
                requested_k = job.get("cache_type_k")
                requested_v = job.get("cache_type_v")
                explicit_cache = bool(requested_k or requested_v)
                if explicit_cache:
                    choices = parse_cache_type_choices(help_text)
                    cache_k = requested_k or "q8_0"
                    cache_v = requested_v or "q8_0"
                    for label, value in (("cache_type_k", cache_k), ("cache_type_v", cache_v)):
                        if value not in choices:
                            raise BadRequest(
                                f"{label} {value!r} 不受支持；可选：{', '.join(choices)}"
                            )
                    if help_text and not (
                        "--cache-type-k" in help_text and "--cache-type-v" in help_text
                    ):
                        raise BadRequest("该 llama.cpp 不支持 --cache-type-k/v，无法设置 K/V 量化精度")
                    cache_override = True
                else:
                    cache_override = "--cache-type-k" in help_text and "--cache-type-v" in help_text
            else:
                cache_override = False
                _, kv_template = kvmem_reference_config(engine_cfg)
                if kv_template is not None:
                    exe = Path(str(kv_template.get("exe") or ""))
                    if exe.is_file():
                        cwd = Path(str(kv_template.get("cwd") or engine_cfg.get("cwd") or "."))
                        cache_override = "--kv" in binary_help_text(exe, cwd)
            if cache_override:
                if short == "llama":
                    detail = (
                        f"K {cache_k} / V {cache_v}（自定义）"
                        if explicit_cache else "k8v8（K/V 均为 q8_0）"
                    )
                else:
                    detail = "q8_0（固定）"
                self._job_step(job, "kv", "ok", detail)
            else:
                note = "引擎帮助不可读，保守使用引擎默认" if not help_text else "引擎不支持 k8v8"
                self._job_step(job, "kv", "warn", f"{note}，使用引擎默认 KV 精度")

            self._job_step(job, "config", "running")
            params = job.get("params") or self.default_sampling(engine_id)
            try:
                cleaned = self.params.replace(name, params, short)
            except BadRequest:
                raise
            if not cleaned:
                cleaned = self.params.replace(name, self.default_sampling(engine_id), short)

            if short == "llama":
                models_cfg = engine_cfg.get("models") if isinstance(engine_cfg.get("models"), dict) else {}
                preset_raw = models_cfg.get("preset")
                if not preset_raw:
                    raise BadRequest("llama engine has no models.preset in config")
                preset_path = Path(str(preset_raw))
                values = {"c": str(context), "model": str(model)}
                if vision is not None:
                    values["mmproj"] = str(vision)
                    if not vision_gpu:
                        values["mmproj-offload"] = "off"
                if cache_override:
                    values["cache-type-k"] = cache_k
                    values["cache-type-v"] = cache_v
                append_llama_preset(preset_path, name, values)
                config_file = preset_path
                job["config_path"] = str(preset_path)
                self._job_step(
                    job, "config", "ok",
                    f"已写入 {preset_path} 的 [{name}] 段"
                    + (
                        f"；视觉 mmproj 走 {'GPU' if vision_gpu else 'CPU'}"
                        if vision is not None else ""
                    ),
                )
            else:
                models_cfg = engine_cfg.get("models") if isinstance(engine_cfg.get("models"), dict) else {}
                config_dir = Path(str(models_cfg.get("dir") or ""))
                template_path, template = kvmem_reference_config(engine_cfg)
                if template is None or template_path is None:
                    raise BadRequest("找不到 KVMem 模板配置（需要一份可用的 ornith strata-*.json）")
                tool = kvmem_tokenizer_tool(engine_cfg)
                if tool is None:
                    raise BadRequest("找不到 KVMem tokenizer 导出工具 tools/polystrata/assets.py")
                tokenizer = self._export_kvmem_tokenizer(engine_cfg, tool, model, job)
                config_file = config_dir / f"strata-{slugify_model_id(name).lower()}.json"
                if config_file.exists():
                    raise BadRequest(f"配置已存在：{config_file}")
                kv_cfg = json.loads(json.dumps(template))
                kv_cfg["model_name"] = name
                kv_cfg["display_name"] = str(info.get("name") or name)
                kv_cfg["aliases"] = []
                args = [str(item) for item in (kv_cfg.get("args") or [])]
                args = _config_arg_replace(args, "--model", str(model))
                args = _config_arg_replace(args, "--max-context", context)
                args = self._fit_kvmem_budget(args, context)
                if cache_override:
                    args = _config_arg_replace(args, "--kv", "q8_0")
                else:
                    args = _config_arg_remove(args, "--kv")
                kv_cfg["args"] = args
                kv_cfg["tokenizer"] = str(tokenizer)
                kv_cfg["tokenizer_preparation"] = "explicit_export"
                kv_cfg["warm_start"] = False
                old_log = Path(str(kv_cfg.get("log") or ""))
                if str(old_log) and str(old_log.parent) not in ("", "."):
                    kv_cfg["log"] = str(old_log.parent / f"serve-{name}.log")
                if vision is not None:
                    vision_cfg = dict(kv_cfg.get("vision") or {})
                    vision_cfg["mmproj"] = str(vision)
                    vision_cfg["model"] = str(model)
                    vision_cfg["gpu"] = bool(vision_gpu)
                    vision_cfg.pop("video", None)
                    if not vision_cfg.get("exe"):
                        vision_cfg["exe"] = str(Path(str(kv_cfg.get("exe"))).with_name("strata-vision"))
                    vision_cfg.setdefault("threads", max(1, (os.cpu_count() or 8) // 2))
                    vision_cfg.setdefault("max_tokens", 65536)
                    if not Path(str(vision_cfg["exe"])).is_file():
                        raise BadRequest(f"视觉编码器不存在：{vision_cfg['exe']}")
                    kv_cfg["vision"] = vision_cfg
                else:
                    kv_cfg.pop("vision", None)
                write_json_atomic(config_file, kv_cfg)
                job["config_path"] = str(config_file)
                self._job_step(
                    job, "config", "ok",
                    f"已写入 {config_file}；tokenizer {tokenizer}"
                    + (
                        f"；视觉走 {'GPU' if vision_gpu else 'CPU'}"
                        if vision is not None else "；无视觉"
                    ),
                )

            self._job_step(job, "start", "running", "正在停止当前引擎…")
            self.ensure_no_async()
            with self.cond:
                engine_alive = bool(self.engine is not None and self.engine.alive())
            if engine_alive:
                self.stop_engine()
            self.registry.reload()
            entry = self.registry.resolve(name)
            if entry is None:
                raise EngineFailure("写入配置后注册表仍未发现该模型")
            if not entry.available:
                raise EngineFailure(f"引擎配置不可用：{entry.problem}")
            self._job_step(job, "start", "running", "正在启动引擎并加载模型…")

            try:
                self.preload(name)
            except EngineFailure as exc:
                # Only the default q8_0 choice retries without a cache override;
                # an explicit q4/f16/... selection fails so the user sees why.
                retry_cache = cache_override and (
                    short != "llama" or (cache_k, cache_v) == ("q8_0", "q8_0")
                )
                if not (retry_cache and self._cache_startup_error(str(exc))):
                    raise
                self._job_step(job, "kv", "warn", "k8v8 启动失败，回退引擎默认精度后重试")
                if short == "llama" and preset_path is not None:
                    remove_llama_preset(preset_path, name)
                    values = {"c": str(context), "model": str(model)}
                    if vision is not None:
                        values["mmproj"] = str(vision)
                        if not vision_gpu:
                            values["mmproj-offload"] = "off"
                    append_llama_preset(preset_path, name, values)
                elif config_file is not None and config_file.is_file():
                    kv_cfg = read_json(config_file, {})
                    if isinstance(kv_cfg, dict):
                        kv_cfg["args"] = _config_arg_remove(
                            [str(item) for item in (kv_cfg.get("args") or [])], "--kv"
                        )
                        write_json_atomic(config_file, kv_cfg)
                self.registry.reload()
                self._job_step(job, "start", "running", "使用默认 KV 精度重新启动…")
                self.preload(name)

            self._job_step(
                job, "start", "ok",
                f"{entry.engine} 已加载 {entry.id}（引擎模型 {entry.engine_model}）",
            )

            if smoke:
                self._job_step(job, "smoke", "running", "发送文本请求…")
                text, text_tokens = self._smoke_completion(
                    entry, [{"role": "user", "content": SMOKE_PROMPT}], job
                )
                self._job_step(
                    job, "smoke", "ok",
                    f"文本正常（prompt {text_tokens} tokens）：{text[:80]}",
                )
                if vision is not None:
                    self._job_step(job, "vision_smoke", "running", "发送图片请求…")
                    messages = None
                    if not self.smoke_image_path.is_file():
                        self._job_step(
                            job, "vision_smoke", "warn",
                            f"默认测试图不存在：{self.smoke_image_path}",
                        )
                        text = vision_tokens = 0
                    else:
                        try:
                            image = base64.b64encode(self.smoke_image_path.read_bytes()).decode("ascii")
                        except OSError as exc:
                            raise BadRequest(f"无法读取默认测试图：{exc}") from exc
                        suffix = self.smoke_image_path.suffix.lower()
                        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": SMOKE_IMAGE_PROMPT},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime};base64,{image}"},
                                    },
                                ],
                            }
                        ]
                    try:
                        if messages is None:
                            raise EngineFailure(f"默认测试图不存在：{self.smoke_image_path}")
                        text, vision_tokens = self._smoke_completion(entry, messages, job)
                    except EngineFailure as exc:
                        self._job_step(job, "vision_smoke", "warn", f"视觉测试失败：{exc}")
                    else:
                        if text_tokens and vision_tokens and vision_tokens <= text_tokens:
                            self._job_step(
                                job, "vision_smoke", "warn",
                                f"图片请求正常但 prompt tokens 未增加（{vision_tokens} ≤ {text_tokens}），"
                                "图像可能未被编码",
                            )
                        else:
                            self._job_step(
                                job, "vision_smoke", "ok",
                                f"图片正常（prompt {vision_tokens} tokens）：{text[:80]}",
                            )
            else:
                self._job_step(job, "smoke", "ok", "跳过（未勾选冒烟测试）")
                if vision is not None:
                    self._job_step(job, "vision_smoke", "ok", "跳过（未勾选冒烟测试）")

            self._job_finish(job, "ok")
            self.log.info("model add %s (%s/%s) finished", job["id"], engine_id, name)
        except Exception as exc:
            message = exc.message if isinstance(exc, GatewayError) else str(exc)
            self.log.exception("model add %s failed", job["id"])
            with self.model_jobs_lock:
                for step in job.get("steps") or []:
                    if step.get("status") == "running":
                        step["status"] = "fail"
                        if message:
                            step["detail"] = message[:300]
                if not any(step.get("status") == "fail" for step in job.get("steps") or []):
                    for step in job.get("steps") or []:
                        if step.get("status") == "pending":
                            step["status"] = "fail"
                            break
            self._rollback_model_add(job, short, name, preset_path, config_file)
            self._job_finish(job, "failed", message)
        finally:
            with self.model_jobs_lock:
                job["status"] = job.get("status") if job.get("status") != "running" else "failed"
                job["finished_at"] = job.get("finished_at") or iso_now()

    def _rollback_model_add(
        self,
        job: dict[str, Any],
        short: str,
        name: str | None,
        preset_path: Path | None,
        config_file: Path | None,
    ) -> None:
        removed: list[str] = []
        if name:
            try:
                self.params.replace(name, {}, short)
            except Exception:
                self.log.exception("cannot remove sampling parameters of failed model %s", name)
            try:
                if short == "llama" and preset_path is not None and remove_llama_preset(preset_path, name):
                    removed.append(str(preset_path))
                elif config_file is not None and config_file.is_file():
                    config_file.unlink()
                    removed.append(str(config_file))
            except OSError:
                self.log.exception("cannot roll back configuration of failed model %s", name)
        job["rollback"] = bool(removed)
        job["removed"] = removed
        try:
            self.registry.reload()
        except Exception:
            self.log.exception("cannot reload the registry after a failed add")
        with self.cond:
            active = self.active
            target = self.target
        if name and ((active is not None and active.id == name) or (target is not None and target.id == name)):
            try:
                self.stop_engine()
            except Exception:
                self.log.exception("cannot stop the engine of a failed model %s", name)

    def delete_model(self, name: str) -> dict[str, Any]:
        entry = self.registry.resolve(name)
        if entry is None:
            raise ModelNotFound(f"model {name!r} is not in the registry")
        if entry.source not in ("preset", "strata", "ninfer"):
            raise BadRequest("该模型由目录扫描发现，没有可删除的配置文件")
        removed: list[str] = []
        if entry.source == "preset":
            preset_path = Path(str(entry.config_path or ""))
            if not preset_path.is_file() or not remove_llama_preset(preset_path, entry.engine_model):
                raise BadRequest(f"models.ini 中找不到 [{entry.engine_model}] 段")
            removed.append(str(preset_path))
        elif entry.source == "ninfer":
            config_file = Path(str(entry.config_path or ""))
            if not config_file.is_file():
                raise BadRequest(f"配置文件不存在：{config_file}")
            config_file.unlink()
            removed.append(str(config_file))
        else:
            config_file = Path(str(entry.config_path or ""))
            if not config_file.is_file():
                raise BadRequest(f"配置文件不存在：{config_file}")
            config_file.unlink()
            removed.append(str(config_file))
            engine_cfg = self.config.get("engines", {}).get(entry.engine) or {}
            models_cfg = engine_cfg.get("models") if isinstance(engine_cfg.get("models"), dict) else {}
            meta_path = Path(str(models_cfg.get("meta") or Path(str(models_cfg.get("dir") or "")) / "models.json"))
            meta = read_json(meta_path, {})
            if isinstance(meta, dict) and entry.id in meta:
                meta.pop(entry.id, None)
                write_json_atomic(meta_path, meta)
                removed.append(str(meta_path))
        with self.cond:
            active = self.active
            target = self.target
        if (active is not None and active.id == entry.id) or (target is not None and target.id == entry.id):
            self.ensure_no_async()
            self.stop_engine()
        self.registry.reload()
        self.params.drop(entry.id)
        still = self.registry.resolve(entry.id)
        self.log.info("deleted model config %s (%s); removed %s", entry.id, entry.engine, removed)
        return {
            "model": entry.id,
            "engine": entry.engine,
            "removed": removed,
            "still_listed": bool(still),
            "note": (
                "配置已删除；模型目录仍会被扫描到，可手动移动模型文件"
                if still else "配置已删除，模型文件未改动"
            ),
        }

    # -- per-model engine parameters (llama.cpp preset) ---------------------

    @staticmethod
    def _llama_model_file(
        engine_cfg: dict[str, Any], name: str, merged: dict[str, str]
    ) -> Path | None:
        models_cfg = engine_cfg.get("models") if isinstance(engine_cfg.get("models"), dict) else {}
        models_dir = Path(str(models_cfg.get("dir") or ""))
        raw = str(merged.get("model") or "").strip()
        if raw:
            path = Path(raw)
            if path.is_absolute():
                return path if path.is_file() else None
            for root in (models_dir / name, models_dir):
                candidate = root / path
                if candidate.is_file():
                    return candidate
            return None
        model_dir = models_dir / name
        try:
            files = sorted(model_dir.glob("*.gguf"))
        except OSError:
            files = []
        return files[0] if files else None

    def _engine_params_values(self, entry: Model, raw: Any) -> dict[str, str] | None:
        if raw is None:
            return None
        if not isinstance(raw, dict) or not raw:
            raise BadRequest("engine_params must be a non-empty object")
        if entry.engine == "kvmem":
            raise BadRequest("KVMem 模型固定使用模型上下文与 q8 量化，不支持修改上下文或 K/V 精度")
        if entry.engine == "ninfer":
            raise BadRequest("NInfer 的上下文与 KV 参数由模型配置决定，不支持在控制台修改")
        if entry.engine != "llama":
            raise BadRequest(f"引擎 {entry.engine!r} 不支持修改上下文或 K/V 精度")
        preset_path = Path(str(entry.config_path or ""))
        if entry.source != "preset" or not preset_path.is_file():
            raise BadRequest("该模型由目录扫描发现，没有可编辑的 models.ini 预设段")
        engine_cfg = self.config.get("engines", {}).get(entry.engine) or {}
        presets = parse_llama_presets(preset_path)
        section = presets.get(entry.engine_model) or {}
        if not section:
            raise BadRequest(f"models.ini 中找不到 [{entry.engine_model}] 段")
        merged = dict(presets.get("*") or {})
        merged.update(section)
        values: dict[str, str] = {}
        if "context_length" in raw:
            context_raw = raw.get("context_length")
            max_context: int | None = None
            model_file = self._llama_model_file(engine_cfg, entry.engine_model, merged)
            info = gguf_model_info(model_file) if model_file is not None else None
            if info:
                max_context = int(info.get("context_length") or 0) or None
            if context_raw is None or (isinstance(context_raw, str) and not context_raw.strip()):
                if max_context is None:
                    current = parse_int(merged.get("c") or merged.get("ctx-size"))
                    if not current:
                        raise BadRequest("无法读取 GGUF 上下文上限，请填写上下文长度")
                    context = current
                else:
                    context = max_context
            else:
                if (
                    isinstance(context_raw, bool)
                    or not isinstance(context_raw, (int, float))
                    or int(context_raw) != context_raw
                ):
                    raise BadRequest("context_length must be an integer")
                context = int(context_raw)
                if context < 256:
                    raise BadRequest("上下文长度至少为 256")
                if max_context is not None and context > max_context:
                    raise BadRequest(
                        f"上下文长度 {context:,} 超过 GGUF 上限 {max_context:,}"
                    )
            values["c"] = str(context)
        choices = llama_cache_type_choices(engine_cfg)
        for field, key in (("cache_type_k", "cache-type-k"), ("cache_type_v", "cache-type-v")):
            if field not in raw:
                continue
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise BadRequest(f"{field} must be a cache type")
            value = value.strip()
            if value not in choices:
                raise BadRequest(f"{field} {value!r} 不受支持；可选：{', '.join(choices)}")
            values[key] = value
        if not values:
            raise BadRequest("engine_params 没有可应用的字段")
        return values

    def save_model_params(
        self, entry: Model, values: dict[str, Any], engine_params: Any = None
    ) -> dict[str, Any]:
        """Validate and persist sampling plus llama preset parameters."""
        cleaned = normalize_sampling(values, entry.engine)
        engine_values = self._engine_params_values(entry, engine_params)
        saved = self.params.replace(entry.id, cleaned, entry.engine)
        result: dict[str, Any] = {
            "params": saved,
            "engine_params": None,
            "restarting": False,
            "restart_pending": False,
        }
        if not engine_values:
            return result
        preset_path = Path(str(entry.config_path or ""))
        update_llama_preset(preset_path, entry.engine_model, engine_values)
        self.registry.reload()
        fresh = self.registry.resolve(entry.id)
        result["engine_params"] = {
            "context_length": fresh.context_length if fresh else None,
            "cache_type_k": fresh.cache_type_k if fresh else None,
            "cache_type_v": fresh.cache_type_v if fresh else None,
        }
        started, pending, note = self._schedule_engine_restart(entry.engine)
        result["restarting"] = started
        result["restart_pending"] = pending
        if note:
            result["note"] = note
        return result

    def _schedule_engine_restart(self, engine_id: str) -> tuple[bool, bool, str | None]:
        """Restart a live engine so it re-reads models.ini.

        Returns (started, pending, note).  pending means the save is valid but the
        engine is busy; the new values then apply on the next engine start.
        """
        with self.cond:
            if self._shutting_down:
                return False, True, "网关正在关闭，新参数将在下次启动时生效"
            if self.async_operation is not None:
                return False, True, "另一个加载/卸载操作正在进行，新参数将在下次加载模型时生效"
            if self.switching:
                return False, True, "网关正在切换模型，新参数将在下次加载模型时生效"
            manager = self.engine
            if manager is None or not manager.alive() or manager.engine_id != engine_id:
                return False, False, "引擎未运行，新参数将在下次加载模型时生效"
            active = self.active
            reload_id = active.id if active is not None and active.engine == engine_id else None
            self.async_operation = {
                "kind": "restart",
                "model": reload_id,
                "engine": engine_id,
                "started_at": iso_now(),
            }
            self.phase = "queued"
            self.cond.notify_all()
        thread = threading.Thread(
            target=self._restart_engine_worker,
            args=(engine_id, reload_id),
            name=f"restart-{engine_id}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            with self.cond:
                self.async_operation = None
                self.phase = "error"
                self.last_error = f"cannot start restart thread: {exc}"
                self.cond.notify_all()
            return False, True, f"无法启动重启线程：{exc}；新参数将在下次加载模型时生效"
        return True, False, None

    def _restart_engine_worker(self, engine_id: str, reload_id: str | None) -> None:
        try:
            self.stop_engine()
            if reload_id:
                self.preload(reload_id)
        except Exception as exc:
            self.log.exception("engine restart after parameter change failed")
            with self.cond:
                self.phase = "error"
                self.last_error = str(exc)
        finally:
            with self.cond:
                self.async_operation = None
                self.cond.notify_all()

    # -- views -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self.cond:
            engine = self.engine
            active = self.active
            target = self.target
            pending_load = self.pending_load
            pending_unload = self.pending_unload
            operation = dict(self.async_operation) if self.async_operation else None
            webui = self.config.get("webui")
            if not isinstance(webui, dict):
                webui = {}
            data = {
                "service": "llm-router",
                "version": VERSION,
                "phase": self.phase,
                "switching": self.switching,
                "in_flight": self.in_flight,
                "queue": self.queue,
                "active_engine": active.engine if active else (engine.engine_id if engine else None),
                "active_model": active.id if active else None,
                "active_engine_model": active.engine_model if active else None,
                "target_model": target.id if target else None,
                "pending_load": pending_load.id if pending_load else None,
                "pending_unload": pending_unload.id if pending_unload else None,
                "async_operation": operation,
                "last_error": self.last_error,
                "last_switch": self.last_switch,
                "uptime_s": round(now() - self.started_at, 1),
                "models": len(self.registry.list()),
                "models_by_engine": self.registry.counts(),
                "ui_path": webui.get("ui_path") or "/ui",
            }
        data["engine"] = engine.status() if engine is not None else None
        data["engines"] = {
            engine_id: manager.status() for engine_id, manager in self.engines.items()
        }
        return data

    def _state_for(
        self,
        model: Model,
        active: Model | None,
        target: Model | None,
        switching: bool,
        phase: str,
        pending_load: Model | None,
        live: bool,
    ) -> str:
        if phase == "error" and target is not None and target.id == model.id:
            return "failed"
        if active is not None and active.id == model.id:
            return "loaded" if live else "failed"
        if target is not None and target.id == model.id:
            if switching or pending_load is not None or phase in (
                "queued",
                "starting",
                "loading",
                "stopping",
                "unloading",
            ):
                return "loading"
        return "unloaded"

    def model_list(self) -> dict[str, Any]:
        with self.cond:
            active = self.active
            target = self.target
            switching = self.switching
            phase = self.phase
            pending_load = self.pending_load
            live = bool(self.engine is not None and self.engine.alive())
            last_error = self.last_error
        data = []
        for model in self.registry.list():
            status = self._state_for(
                model, active, target, switching, phase, pending_load, live
            )
            item = model.openai(status, status == "loaded", self.params.get(model.id))
            if status == "failed" and last_error and target is not None and target.id == model.id:
                item["error"] = last_error
            data.append(item)
        return {"object": "list", "data": data}

    def model_item(self, model: Model) -> dict[str, Any]:
        for item in self.model_list().get("data") or []:
            if item.get("id") == model.id:
                return item
        return model.openai("unloaded", False, self.params.get(model.id))

    def params_view(self) -> dict[str, Any]:
        data = []
        for entry in self.registry.list():
            data.append(
                {
                    "model": entry.id,
                    "engine": entry.engine,
                    "params": self.params.get(entry.id),
                }
            )
        return {"object": "list", "data": data}


# ---------------------------------------------------------------------------
# HTTP front door


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"llm-router/{VERSION}"
    gateway: Gateway

    # -- basic helpers -----------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        self.gateway.log.debug("%s - %s", self.address_string(), fmt % args)

    def _cors(self) -> None:
        origin = self.gateway.config.get("cors_origin") or "*"
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        requested = self.headers.get("Access-Control-Request-Headers")
        self.send_header(
            "Access-Control-Allow-Headers",
            requested or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta",
        )
        self.send_header("Access-Control-Max-Age", "86400")

    def _json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self._response_started = True
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str = "server_error") -> None:
        if self._response_started:
            # Headers are already on the wire (a streamed upstream failed); a second
            # status line would corrupt the response.  Drop the connection instead.
            self.close_connection = True
            return
        extra = {"WWW-Authenticate": 'Bearer realm="llm-router"'} if status == 401 else None
        self._json(status, {"error": {"message": message, "type": kind}}, extra)

    def _read_body(self) -> bytes:
        cached = getattr(self, "_body_cache", None)
        if cached is not None:
            return cached
        max_body = int(self.gateway.config.get("max_body_bytes") or 268435456)
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            chunks = []
            total = 0
            while True:
                line = self.rfile.readline(65536)
                if not line:
                    raise BadRequest("incomplete chunked request body")
                size_text = line.split(b";", 1)[0].strip()
                try:
                    size = int(size_text, 16)
                except ValueError:
                    raise BadRequest("invalid chunked request body")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline(65536)
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                total += size
                if total > max_body:
                    raise RequestTooLarge("request body is too large")
                chunk = self.rfile.read(size)
                if len(chunk) != size:
                    raise BadRequest("incomplete chunked request body")
                chunks.append(chunk)
                self.rfile.read(2)
            self._body_cache = b"".join(chunks)
            return self._body_cache
        length_text = self.headers.get("Content-Length")
        if not length_text:
            self._body_cache = b""
            return self._body_cache
        try:
            length = int(length_text)
        except ValueError:
            raise BadRequest("invalid Content-Length")
        if length < 0:
            raise BadRequest("invalid Content-Length")
        if length > max_body:
            raise RequestTooLarge("request body is too large")
        body = self.rfile.read(length) if length else b""
        if len(body) != length:
            raise BadRequest("incomplete request body")
        self._body_cache = body
        return body

    # -- HTTP verbs --------------------------------------------------------

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        self._handle("GET")

    def do_HEAD(self) -> None:
        self._handle("HEAD")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_PUT(self) -> None:
        self._handle("PUT")

    def do_PATCH(self) -> None:
        self._handle("PATCH")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    def _handle(self, method: str) -> None:
        self._response_started = False
        self._body_cache = None
        try:
            # Consume the request body up front so that every early error response
            # leaves the keep-alive connection in a clean state for the next request.
            try:
                self._read_body()
            except GatewayError:
                self.close_connection = True
                raise
            self._route(method)
        except GatewayError as exc:
            if self._response_started:
                self.close_connection = True
                return
            try:
                self._error(exc.status, exc.message, exc.kind)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self.gateway.log.error("unhandled request error:\n%s", traceback.format_exc())
            if self._response_started:
                self.close_connection = True
                return
            try:
                self._error(500, "internal gateway error")
            except Exception:
                self.close_connection = True

    # -- web UI and static files -------------------------------------------

    def _webui_config(self) -> dict[str, Any]:
        value = self.gateway.config.get("webui")
        return value if isinstance(value, dict) else {}

    def _webui_path(self, key: str, default: str) -> Path:
        cfg = self._webui_config()
        path = Path(str(cfg.get(key) or default))
        return path if path.is_absolute() else self.gateway.base_dir / path

    def _ui_root(self) -> Path:
        cfg = self._webui_config()
        return self._webui_path("root", "webui") / str(cfg.get("ui_dir") or "llama-ui")

    def _console_path(self) -> Path:
        cfg = self._webui_config()
        return self._webui_path("root", "webui") / str(cfg.get("console") or "index.html")

    def _ui_path(self) -> str:
        value = str(self._webui_config().get("ui_path") or "/ui").strip()
        if not value or value == "/":
            return "/ui"
        if not value.startswith("/"):
            value = "/" + value
        return value.rstrip("/") or "/ui"

    @staticmethod
    def _is_ui_api(path: str) -> bool:
        for prefix in UI_API_PREFIXES:
            if path == prefix.rstrip("/"):
                return True
            if prefix.endswith("/"):
                if path.startswith(prefix):
                    return True
            elif path.startswith(prefix + "/"):
                return True
        return False

    def _request_api_key(self) -> str:
        auth = self.headers.get("Authorization", "")
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not given:
            given = (self.headers.get("x-api-key") or "").strip()
        return given

    def _check_api_key(self) -> None:
        key = self.gateway.auth_key
        if not key:
            return
        given = self._request_api_key()
        if given and hmac.compare_digest(given, key):
            return
        raise Unauthorized("invalid or missing API key")

    def _check_origin(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            return
        allowed = self.gateway.config.get("allowed_origins") or []
        if isinstance(allowed, str):
            allowed = [allowed]
        if origin in allowed:
            return
        host = (self.headers.get("Host") or "").strip().lower()
        if host and urlsplit(origin).netloc.lower() == host:
            return
        raise Forbidden("cross-origin control requests are not allowed")

    def _redirect(self, location: str, status: int = 302) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()
        self._response_started = True

    def _send_file(self, path: Path, method: str, cache: str = "default") -> None:
        try:
            stat = path.stat()
        except OSError:
            raise NotFound("not found")
        if not path.is_file():
            raise NotFound("not found")
        ctype = (
            STATIC_TYPES.get(path.suffix.lower())
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        if cache == "immutable":
            cache_control = "public, max-age=31536000, immutable"
        elif cache == "no-cache":
            cache_control = "no-cache, no-store, must-revalidate"
        else:
            cache_control = "public, max-age=3600"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Cache-Control", cache_control)
        self.send_header("Last-Modified", formatdate(stat.st_mtime, usegmt=True))
        self.send_header("X-Content-Type-Options", "nosniff")
        if path.name == "sw.js":
            self.send_header("Service-Worker-Allowed", self._ui_path() + "/")
        self.end_headers()
        self._response_started = True
        if method == "HEAD":
            return
        try:
            with path.open("rb") as fh:
                while True:
                    chunk = fh.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except OSError:
            self.gateway.log.exception("cannot read static file %s", path)
            self.close_connection = True

    def _serve_console(self, method: str) -> None:
        cfg = self._webui_config()
        path = self._console_path()
        if cfg.get("enabled", True) and path.is_file():
            self._send_file(path, method, "no-cache")
            return
        self._json(
            200,
            {
                "service": "llm-router",
                "version": VERSION,
                "status": "ok",
                "models": len(self.gateway.registry.list()),
                "api": "/v1",
                "control": "/router/status",
                "console": "not installed",
            },
        )

    def _serve_ui(self, method: str, path: str) -> None:
        root = self._ui_root().resolve()
        rel = unquote(path).lstrip("/")
        if "\x00" in rel:
            raise NotFound("not found")
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise NotFound("not found")
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            if Path(rel).suffix:
                raise NotFound("not found")
            candidate = root / "index.html"
            if not candidate.is_file():
                raise NotFound("web UI build is not installed")
        name = candidate.name
        parts = candidate.parts
        if "_app" in parts and "immutable" in parts:
            cache = "immutable"
        elif name in ("index.html", "sw.js", "manifest.webmanifest", "ui-build.json"):
            cache = "no-cache"
        else:
            cache = "default"
        self._send_file(candidate, method, cache)

    # -- SSE ---------------------------------------------------------------

    def _sse(self) -> None:
        hub = self.gateway.status_hub
        client = hub.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            self._response_started = True
            while not self.gateway.shutting_down():
                try:
                    event = client.get(timeout=15.0)
                except queue.Empty:
                    payload = b": ping\n\n"
                else:
                    if event is None:
                        break
                    payload = (
                        b"data: "
                        + json.dumps(event, ensure_ascii=False).encode("utf-8")
                        + b"\n\n"
                    )
                self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except OSError:
            self.close_connection = True
        finally:
            hub.unsubscribe(client)
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                self.close_connection = True

    # -- props -------------------------------------------------------------

    def _props(self, entry: Model | None) -> dict[str, Any]:
        with self.gateway.cond:
            active = self.gateway.active
            engine = self.gateway.engine
        target = entry or active
        if target is not None and target.context_length:
            context = target.context_length
        else:
            values = [
                m.context_length
                for m in self.gateway.registry.list()
                if m.context_length
            ]
            context = max(values) if values else 0
        modalities = {
            "vision": bool(target.multimodal) if target is not None else False,
            "audio": bool(target.audio) if target is not None else False,
            "video": bool(target.video) if target is not None else False,
        }
        props: dict[str, Any] = {
            "role": "router",
            "model_path": "none",
            "model_alias": target.id if target is not None else "",
            "build_info": f"llm-router {VERSION}",
            "total_slots": 1,
            "max_instances": 1,
            "models_autoload": False,
            "default_generation_settings": {"params": {}, "n_ctx": context},
            "ui_settings": {},
            "webui_settings": {},
            "modalities": modalities,
            "chat_template": "",
            "system_prompt": "",
        }
        if (
            active is not None
            and engine is not None
            and engine.alive()
            and (entry is None or entry.id == active.id)
        ):
            manager = self.gateway.engines.get(active.engine)
            if manager is not None:
                try:
                    status, body = manager._json_request(
                        "GET",
                        "/props?model=" + quote(active.engine_model) + "&autoload=false",
                        entry=active,
                        timeout=5.0,
                    )
                    if status == 200 and isinstance(body, dict):
                        props.update(body)
                        engine_modalities = body.get("modalities")
                        if isinstance(engine_modalities, dict):
                            # Engines such as kvmem only report the keys they
                            # know (vision); keep router-derived capabilities
                            # the engine omits so video/audio gates stay open.
                            merged = dict(modalities)
                            merged.update(engine_modalities)
                            props["modalities"] = merged
                except Exception:
                    self.gateway.log.exception("cannot read /props from %s", active.engine)
        # A router-mode UI must never treat the gateway as a single model.
        props["role"] = "router"
        props["model_alias"] = target.id if target is not None else ""
        settings = props.get("default_generation_settings")
        if not isinstance(settings, dict):
            settings = {}
        settings.setdefault("params", {})
        if not isinstance(settings.get("n_ctx"), (int, float)) or not settings.get("n_ctx"):
            settings["n_ctx"] = context
        props["default_generation_settings"] = settings
        if not isinstance(props.get("modalities"), dict):
            props["modalities"] = modalities
        if not isinstance(props.get("ui_settings"), dict):
            props.setdefault("webui_settings", {})
        return props

    # -- routing -----------------------------------------------------------

    def _json_body(self, required: bool = False) -> dict[str, Any] | None:
        body = self._read_body()
        if not body:
            if required:
                raise BadRequest("send a JSON object")
            return None
        try:
            payload = json.loads(body)
        except ValueError:
            raise BadRequest("request body is not valid JSON")
        if not isinstance(payload, dict):
            raise BadRequest("send a JSON object")
        return payload

    def _route(self, method: str) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path or "/"
        ui_path = self._ui_path()

        if path == ui_path:
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            location = ui_path + "/"
            if parsed.query:
                location += "?" + parsed.query
            self._redirect(location)
            return
        if path.startswith(ui_path + "/"):
            rest = path[len(ui_path):] or "/"
            if rest == "/":
                if method not in ("GET", "HEAD"):
                    raise MethodNotAllowed("method not allowed")
                self._serve_ui(method, "/")
                return
            if self._is_ui_api(rest):
                self._api_route(method, rest, parsed.query)
                return
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._serve_ui(method, rest)
            return
        if path in ("/", ""):
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._serve_console(method)
            return
        self._api_route(method, path, parsed.query)

    def _api_route(self, method: str, path: str, query: str) -> None:
        normalized = path.rstrip("/") or "/"

        # One key for every model; the inference API and the UI props probe are
        # protected.  The console and control endpoints stay same-origin +
        # trusted-LAN only, so a forgotten key can still be reset from the console.
        if (
            (normalized.startswith("/v1/") and normalized != "/v1/health")
            or normalized == "/props"
        ):
            self._check_api_key()

        if method in ("POST", "PUT", "PATCH", "DELETE") and (
            normalized in (
                "/models",
                "/models/load",
                "/models/unload",
                "/router/reload",
                "/router/load",
                "/router/unload",
                "/router/stop",
            )
            or normalized.startswith("/router/")
        ):
            self._check_origin()

        if normalized in ("/health", "/v1/health"):
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._json(200, {"status": "ok", **self.gateway.status()})
            return
        if normalized == "/version":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._json(200, {"version": VERSION})
            return
        if normalized == "/v1/models":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._json(200, self.gateway.model_list())
            return
        if normalized == "/models":
            if method in ("GET", "HEAD"):
                self._json(200, self.gateway.model_list())
                return
            self.close_connection = True
            raise Unsupported("model download/delete is not supported by this gateway")
        if normalized.startswith("/v1/models/"):
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            model_id = unquote(path[len("/v1/models/"):]).strip("/")
            entry = self.gateway.registry.resolve(model_id)
            if entry is None:
                raise ModelNotFound(f"model {model_id!r} is not in the registry")
            self._json(200, self.gateway.model_item(entry))
            return
        if normalized == "/models/load":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            name = payload.get("model")
            if not isinstance(name, str) or not name.strip():
                raise BadRequest('send {"model": "..."}')
            entry, started = self.gateway.schedule_load(name.strip())
            self._json(
                202 if started else 200,
                {
                    "success": True,
                    "status": "loading" if started else "loaded",
                    "model": entry.id,
                    "engine": entry.engine,
                },
            )
            return
        if normalized == "/models/unload":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=False) or {}
            name = payload.get("model")
            if name is not None and not isinstance(name, str):
                raise BadRequest('send {"model": "..."} or an empty body')
            entry, started = self.gateway.schedule_unload(name)
            self._json(
                202 if started else 200,
                {
                    "success": True,
                    "status": "unloading" if started else "unloaded",
                    "model": entry.id if entry is not None else None,
                },
            )
            return
        if normalized == "/models/sse":
            if method != "GET":
                raise MethodNotAllowed("method not allowed")
            self._sse()
            return
        if normalized in ("/props", "/v1/props"):
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            values = parse_qs(query).get("model")
            entry = None
            if values:
                entry = self.gateway.registry.resolve(values[0])
                if entry is None:
                    raise ModelNotFound(f"model {values[0]!r} is not in the registry")
            self._json(200, self._props(entry))
            return
        if normalized == "/slots":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._json(200, [])
            return
        if normalized == "/tools":
            if method in ("GET", "HEAD"):
                self._json(200, [])
                return
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=False) or {}
            tool = payload.get("tool") if isinstance(payload, dict) else None
            # The gateway deliberately does not expose engine-side shell/file tools.
            # Answer with a JSON error so the web UI can degrade gracefully.
            self._json(
                200,
                {
                    "error": f"server tool {tool or 'unknown'!r} is not available on this gateway",
                },
            )
            return
        if normalized == "/v1/streams/lookup":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            self._read_body()
            self._json(200, [])
            return
        if normalized == "/v1/stream":
            raise NotFound("stream resume is not supported by this gateway")
        if normalized == "/router/status":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._json(200, self.gateway.status())
            return
        if normalized == "/router/auth":
            if method in ("GET", "HEAD"):
                self._json(200, self.gateway.auth_view())
                return
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            clear = payload.get("clear") is True
            new_key = payload.get("api_key")
            if not clear:
                if not isinstance(new_key, str) or not new_key.strip():
                    raise BadRequest('send {"api_key": "..."} or {"clear": true}')
                new_key = new_key.strip()
            current = self._request_api_key()
            if not current and isinstance(payload.get("current_api_key"), str):
                current = str(payload["current_api_key"]).strip()
            server_key = self.gateway.auth_key
            if server_key:
                if not current:
                    raise Unauthorized("current API key is required to change the server key")
                if not hmac.compare_digest(current, server_key):
                    raise Unauthorized("current API key is incorrect")
            result = self.gateway.set_auth_key(None if clear else new_key)
            self._json(200, {"status": "ok", **result})
            return
        if normalized == "/router/params":
            if method in ("GET", "HEAD"):
                self._json(200, self.gateway.params_view())
                return
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            name = payload.get("model")
            if not isinstance(name, str) or not name.strip():
                raise BadRequest('send {"model": "...", "params": {...}}')
            entry = self.gateway.registry.resolve(name.strip())
            if entry is None:
                raise ModelNotFound(f"model {name!r} is not in the registry")
            values = payload.get("params")
            if values is None:
                values = {}
            if not isinstance(values, dict):
                raise BadRequest("params must be an object")
            result = self.gateway.save_model_params(
                entry, values, payload.get("engine_params")
            )
            self._json(
                200,
                {"status": "ok", "model": entry.id, "engine": entry.engine, **result},
            )
            return
        if normalized == "/router/models/manage":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._json(200, self.gateway.model_manage_view())
            return
        if normalized == "/router/models/add":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            job = self.gateway.start_model_add(payload)
            self._json(202, job)
            return
        if normalized == "/router/models/job":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            values = parse_qs(query).get("id")
            job_id = values[0].strip() if values else ""
            if not job_id:
                raise BadRequest('send ?id=<job id>')
            job = self.gateway.get_model_job(job_id)
            if job is None:
                raise NotFound("model job not found")
            self._json(200, job)
            return
        if normalized == "/router/models/delete":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            name = payload.get("model")
            if not isinstance(name, str) or not name.strip():
                raise BadRequest('send {"model": "..."}')
            result = self.gateway.delete_model(name.strip())
            self._json(200, {"status": "ok", **result})
            return
        if normalized == "/router/engines/paths":
            if method in ("GET", "HEAD"):
                self._json(200, self.gateway.engine_paths_view())
                return
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            engine = payload.get("engine")
            if not isinstance(engine, str) or not engine.strip():
                raise BadRequest('send {"engine": "llama", "paths": {...}}')
            paths = payload.get("paths")
            if not isinstance(paths, dict):
                raise BadRequest("paths must be an object")
            result = self.gateway.save_engine_paths(
                engine.strip(), paths, payload.get("create") is True
            )
            self._json(200, result)
            return
        if normalized == "/router/engines/restart":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            engine = payload.get("engine")
            if not isinstance(engine, str) or not engine.strip():
                raise BadRequest('send {"engine": "llama"}')
            result = self.gateway.restart_engine_paths(engine.strip())
            self._json(202 if result.get("restarting") else 200, result)
            return
        if normalized == "/router/proxy":
            if method in ("GET", "HEAD"):
                self._json(200, self.gateway.proxy_view())
                return
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            self._json(200, self.gateway.save_proxy(payload))
            return
        if normalized == "/router/proxy/test":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            self._json(200, self.gateway.test_proxy(payload))
            return
        if normalized == "/router/downloads":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            self._json(200, self.gateway.downloads_view())
            return
        if normalized == "/router/downloads/settings":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            self._json(200, self.gateway.save_download_settings(payload))
            return
        if normalized == "/router/downloads/search":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            values = parse_qs(query)
            query_text = (values.get("q") or [""])[0]
            limit_text = (values.get("limit") or [None])[0]
            source_text = (values.get("sources") or [None])[0]
            self._json(
                200,
                self.gateway.search_downloads(query_text, limit_text, source_text),
            )
            return
        if normalized == "/router/downloads/plan":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            self._json(200, self.gateway.plan_download(payload))
            return
        if normalized == "/router/downloads/start":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            self._json(202, self.gateway.start_download(payload))
            return
        if normalized == "/router/downloads/job":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            values = parse_qs(query).get("id")
            job_id = values[0].strip() if values else ""
            if not job_id:
                raise BadRequest('send ?id=<download job id>')
            job = self.gateway.download_job_view(job_id)
            if job is None:
                raise NotFound("download job not found")
            self._json(200, job)
            return
        if normalized == "/router/downloads/cancel":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            job_id = payload.get("id")
            if not isinstance(job_id, str) or not job_id.strip():
                raise BadRequest('send {"id": "..."}')
            job = self.gateway.cancel_download(job_id.strip())
            if job is None:
                raise NotFound("download job not found")
            self._json(202 if job.get("status") == "running" else 200, job)
            return
        if normalized == "/router/reload":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            self._read_body()
            self.gateway.registry.reload()
            self._json(200, {"status": "ok", "models": len(self.gateway.registry.list())})
            return
        if normalized == "/router/load":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            payload = self._json_body(required=True) or {}
            name = payload.get("model")
            if not isinstance(name, str) or not name.strip():
                raise BadRequest('send {"model": "..."}')
            self.gateway.ensure_no_async()
            entry = self.gateway.preload(name.strip())
            self._json(200, {"status": "loaded", "model": entry.id, "engine": entry.engine})
            return
        if normalized == "/router/unload":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            self._read_body()
            self.gateway.ensure_no_async()
            self.gateway.unload_model()
            self._json(200, {"status": "unloaded"})
            return
        if normalized == "/router/stop":
            if method != "POST":
                raise MethodNotAllowed("method not allowed")
            self._read_body()
            self.gateway.ensure_no_async()
            self.gateway.stop_engine()
            self._json(200, {"status": "stopped"})
            return
        if normalized.startswith("/router/"):
            raise NotFound("unknown router endpoint")
        if normalized == "/favicon.ico":
            if method not in ("GET", "HEAD"):
                raise MethodNotAllowed("method not allowed")
            icon = self._ui_root() / "favicon.ico"
            if icon.is_file():
                self._send_file(icon, method, "default")
            else:
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                self._response_started = True
            return
        self._proxy(method, path, query)

    # -- proxy -------------------------------------------------------------

    def _proxy(self, method: str, path: str, query: str) -> None:
        started = time.monotonic()
        body = self._read_body()
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        payload: Any = None
        if body and (ctype == "application/json" or path in OPENAI_JSON_PATHS):
            try:
                payload = json.loads(body)
            except ValueError:
                if path in OPENAI_JSON_PATHS:
                    raise BadRequest("request body is not valid JSON")
        model_name = None
        payload_changed = False
        if isinstance(payload, dict) and isinstance(payload.get("model"), str):
            model_name = payload["model"]
        if not model_name and path in ("/props", "/v1/props"):
            values = parse_qs(query).get("model")
            if values:
                model_name = values[0]
        if not model_name:
            snapshot = self.gateway.status()
            model_name = snapshot.get("active_model") or self.gateway.config.get("default_model")
            if not model_name:
                if path in MODEL_REQUIRED_PATHS:
                    raise BadRequest("model is required in the request body")
                raise BadRequest("no model is loaded; call /router/load first or name a model")
            if isinstance(payload, dict) and path in MODEL_REQUIRED_PATHS:
                payload["model"] = model_name
                payload_changed = True
        entry = self.gateway.acquire(model_name)
        if (
            isinstance(payload, dict)
            and path.rstrip("/") in SAMPLING_GENERATION_PATHS
            and self.gateway.params.apply(entry.engine, entry.id, payload)
        ):
            payload_changed = True
        session = requests.Session()
        session.trust_env = False
        resp = None
        try:
            upstream_headers = self._upstream_headers(entry, body, payload, ctype)
            if isinstance(payload, dict) and (
                payload_changed or payload.get("model") != entry.engine_model
            ):
                payload["model"] = entry.engine_model
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                upstream_headers["Content-Type"] = "application/json"
            manager = self.gateway.engines.get(entry.engine)
            if manager is None:
                raise EngineFailure(f"no engine configured for {entry.engine}")
            url = f"http://{manager.host}:{manager.port}{path}"
            if query:
                url += "?" + query
            resp = session.request(
                method,
                url,
                headers=upstream_headers,
                data=body if body else None,
                stream=True,
                allow_redirects=False,
                timeout=(
                    float(self.gateway.timeouts.get("proxy_connect_timeout_s") or 10),
                    float(self.gateway.timeouts.get("proxy_read_timeout_s") or 3600),
                ),
            )
            status = resp.status_code
            self._send_upstream(resp, method, payload)
            self.gateway.log.info(
                "%s %s -> %s [%s] HTTP %s %.2fs",
                method,
                self.path,
                entry.id,
                entry.engine,
                status,
                time.monotonic() - started,
            )
        except requests.RequestException as exc:
            self.gateway.log.error("upstream request %s %s failed: %s", method, self.path, exc)
            if self._response_started:
                self.close_connection = True
            else:
                try:
                    self._error(502, f"upstream engine request failed: {exc}")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
        finally:
            if resp is not None:
                resp.close()
            session.close()
            self.gateway.release(entry)

    def _upstream_headers(self, entry: Model, body: bytes, payload: Any, ctype: str) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP or lower in ("origin", "referer"):
                continue
            if lower in ("authorization", "x-api-key"):
                continue
            if lower == "accept-encoding":
                continue
            headers[key] = value
        headers["Accept-Encoding"] = "identity"
        if body and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json" if ctype == "application/json" else "application/octet-stream"
        engine_cfg = self.gateway.config.get("engines", {}).get(entry.engine, {})
        key = first_key(engine_cfg.get("api_key"))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _send_upstream(self, resp: requests.Response, method: str, payload: Any) -> None:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        is_sse = "text/event-stream" in ctype
        wants_stream = isinstance(payload, dict) and payload.get("stream") is True
        is_stream = resp.status_code == 200 and (is_sse or wants_stream)
        self.send_response(resp.status_code)
        for key, value in resp.headers.items():
            lower = key.lower()
            if lower in RESPONSE_SKIP:
                continue
            self.send_header(key, value)
        self._cors()
        if is_stream:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            if self.request_version == "HTTP/1.0":
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                self._response_started = True
                try:
                    for chunk in resp.iter_content(chunk_size=4096):
                        if chunk:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except (requests.RequestException, OSError):
                    self.close_connection = True
                return
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self._response_started = True
            try:
                for chunk in resp.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                    self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (requests.RequestException, OSError):
                self.close_connection = True
            return
        data = resp.content
        if resp.status_code in (204, 304):
            self.end_headers()
            self._response_started = True
            return
        length = len(data)
        if method == "HEAD":
            upstream_length = resp.headers.get("Content-Length")
            if upstream_length and upstream_length.isdigit():
                length = int(upstream_length)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self._response_started = True
        if method != "HEAD" and data:
            self.wfile.write(data)


# ---------------------------------------------------------------------------
# main


def setup_logging(base_dir: Path, config: dict[str, Any]) -> logging.Logger:
    log_path = Path(str(config.get("log_file") or "logs/router.log"))
    if not log_path.is_absolute():
        log_path = base_dir / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("llm-router")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(sh)
    return logger


DEFAULT_CONFIG: dict[str, Any] = {
    "listen": {"host": "0.0.0.0", "port": 8080},
    "engines": {},
    "timeouts": {"queue_s": 600},
    "log_file": "logs/router.log",
    "state_file": "state.json",
    "pid_file": "router.pid",
    "model_params_file": "model-params.json",
    "api_keys_file": "api-keys.json",
    "proxy": {
        "enabled": False,
        "type": "http",
        "host": "",
        "port": 0,
        "username": "",
        "password": "",
    },
    "downloads": {
        "dir": "",
        "sources": {
            "huggingface": {"use_proxy": False},
            "hf-mirror": {"use_proxy": False},
            "modelscope": {"use_proxy": False},
        },
    },
}


def load_config(path: Path) -> dict[str, Any]:
    """Load config.json, creating a minimal one on first start."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, DEFAULT_CONFIG)
        print(f"created default config: {path}")
        print("no engine configured yet; open the management page in the web UI to add one")
        return json.loads(json.dumps(DEFAULT_CONFIG))
    data = read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"cannot read config: {path} (expected a JSON object)")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="llm-router gateway")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    parser.add_argument("--host", help="listen host override (not written to config.json)")
    parser.add_argument("--port", type=int, help="listen port override (not written to config.json)")
    parser.add_argument("--check", action="store_true", help="scan and print models, then exit")
    args = parser.parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    config = load_config(config_path)
    log = setup_logging(base_dir, config)
    registry = Registry(config, base_dir, log)
    if args.check:
        params = SamplingStore(
            resolve_path(base_dir, config.get("model_params_file") or "model-params.json"),
            log,
            registry,
        )
        print(json.dumps(
            {
                "models": len(registry.list()),
                "by_engine": registry.counts(),
                "items": [m.openai("unloaded", False, params.get(m.id)) for m in registry.list()],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    timeouts = config.get("timeouts", {})
    state_path = base_dir / str(config.get("state_file") or "state.json")
    engines: dict[str, EngineManager] = {}
    for engine_id, engine_cfg in config.get("engines", {}).items():
        engines[engine_id] = EngineManager(engine_id, engine_cfg, base_dir, timeouts, state_path, log)
    gateway = Gateway(config, registry, engines, state_path, base_dir, log, config_path)
    Handler.gateway = gateway

    listen = config.get("listen", {})
    host = str(args.host or listen.get("host") or "0.0.0.0")
    port = int(args.port or listen.get("port") or 8080)
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        log.error("cannot bind %s:%s: %s (is the old service still running?)", host, port, exc)
        return 2
    server.daemon_threads = True
    server.allow_reuse_address = True
    server.timeout = 0.5
    pid_path = base_dir / str(config.get("pid_file") or "router.pid")
    try:
        pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    except OSError:
        log.warning("cannot write pid file %s", pid_path)
    cleanup_stale_state(state_path, log)
    stop_event = threading.Event()

    def _signal(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)
    log.info("llm-router %s listening on %s:%s; %d models, no engine loaded", VERSION, host, port, len(registry.list()))
    try:
        while not stop_event.is_set():
            try:
                server.handle_request()
            except OSError:
                if stop_event.is_set():
                    break
                log.exception("accept error")
        log.info("shutting down")
    finally:
        try:
            server.server_close()
        except Exception:
            pass
        gateway.shutdown()
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass
        log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
