#!/usr/bin/env python3
"""llm-router: one OpenAI-compatible front door for the llama.cpp router and KVMem.

The gateway owns the public port.  It starts with no engine and no model loaded; the
requested model selects the engine.  A cross-engine switch stops the current engine and
starts the other one; a same-engine switch uses the engine's own model-switch API.

Only the Python standard library and requests are used.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import logging
import mimetypes
import os
import queue
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

import requests


VERSION = "0.3.0"

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
# per-model sampling defaults

SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
)
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
}
# A llama.cpp preset that encodes the reference defaults (the [*] block plus
# the LFM3-2B-omni section); KVMem's reference config is Ornith 1.5.
LLAMA_REFERENCE_MODEL = "LFM3-2B-omni"
KVMEM_REFERENCE_MODEL = "ornith-1.5-35b-a3b-vl-thinking"
SMOKE_PROMPT = "Reply with the single word: OK"
SMOKE_IMAGE_PROMPT = "What shape is in this image? Answer in one word."


def sampling_aliases(key: str) -> tuple[str, ...]:
    if key == "repetition_penalty":
        return ("repetition_penalty", "repeat_penalty")
    if key == "repeat_last_n":
        return ("repeat_last_n", "penalty_last_n")
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


def normalize_sampling(values: dict[str, Any], engine: str) -> dict[str, Any]:
    """Validate one model's sampling set.  None / '' clears a field; unknown fields are rejected."""
    if not isinstance(values, dict):
        raise BadRequest("params must be an object")
    out: dict[str, Any] = {}
    for key, raw in values.items():
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        if key not in SAMPLING_FIELDS:
            raise BadRequest(f"unknown sampling parameter {key!r}")
        if key == "temperature":
            out[key] = _sampling_number(key, raw, 0.0, 2.0)
        elif key == "top_p":
            out[key] = _sampling_number(key, raw, 0.0, 1.0, low_exclusive=True)
        elif key == "top_k":
            if engine == "kvmem":
                out[key] = _sampling_number(key, raw, 1, 64, integer=True)
            else:
                out[key] = _sampling_number(key, raw, 0, 1000, integer=True)
        elif key == "min_p":
            out[key] = _sampling_number(key, raw, 0.0, 1.0)
        elif key == "repetition_penalty":
            out[key] = _sampling_number(key, raw, 0.5, 2.0)
        elif key == "repeat_last_n":
            if engine == "kvmem":
                out[key] = _sampling_number(key, raw, 1, 131072, integer=True)
            else:
                out[key] = _sampling_number(key, raw, -1, 131072, integer=True)
        elif key == "presence_penalty":
            out[key] = _sampling_number(key, raw, -2.0, 2.0)
        elif key == "frequency_penalty":
            out[key] = _sampling_number(key, raw, -2.0, 2.0)
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
            "owned_by": "llama.cpp" if self.engine == "llama" else "KVMem",
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
            "multimodal": self.multimodal,
            "video": self.video,
            "audio": self.audio,
            "reasoning": self.reasoning,
            "notes": self.notes,
            "quantization": self.quantization,
            "source": self.source,
            "config_path": self.config_path,
            "deletable": self.source in ("preset", "strata"),
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

    def _build_command(self, entry: Model) -> list[str]:
        command = self.cfg.get("command")
        if not isinstance(command, list) or not command:
            raise EngineFailure(f"{self.engine_id}: no command in config")
        if self.kind == "kvmem" and not entry.config_path:
            raise EngineFailure(f"{self.engine_id}: {entry.id} has no config path")
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
        if self.kind == "kvmem" and entry.config_path:
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
        self.current_model = entry.engine_model if self.kind == "kvmem" else None
        self._write_state()
        try:
            if self.kind == "llama_router":
                self._wait_llama_ready()
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
                if state == "loaded":
                    self.current_model = target_id
                    return
                if state == "unloaded" and isinstance(target.get("status"), dict) and target["status"].get("failed"):
                    detail = target["status"].get("exit_code")
                    raise EngineFailure(f"llama failed to load {entry.id} (exit code {detail}); log: {tail_file(self.log_path, 60)}")
            if time.monotonic() >= deadline:
                raise EngineFailure(f"llama did not load {entry.id} within {int(timeout)}s; log: {tail_file(self.log_path, 60)}")
            if time.monotonic() - last_note > 15:
                last_note = time.monotonic()
                self.log.info("loading llama model %s", entry.id)
            time.sleep(1.0)

    def load(self, entry: Model) -> None:
        if self.current_model == entry.engine_model and self.alive():
            self.current_entry = entry
            return
        if not self.alive():
            raise EngineFailure(f"{self.engine_id}: engine is not running")
        if self.kind == "llama_router":
            self._load_llama_model(entry)
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
        if state == "loaded":
            return
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
        return {"llama_dir": "llama", "kvmem_dir": "kvmem"}.get(str(kind), "")

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
                    "default_params": self.default_sampling(engine_id),
                }
            elif kind == "kvmem_dir":
                template_path, template = kvmem_reference_config(engine_cfg)
                out[engine_id] = {
                    "engine": engine_id,
                    "kind": "kvmem",
                    "label": "KVMem",
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
        if kind not in ("llama_dir", "kvmem_dir"):
            raise BadRequest(f"engine {engine_id!r} cannot add models")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise BadRequest('send {"engine": "...", "model": "/path/model.gguf"}')
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
        steps = [
            {"key": "paths", "label": "检查模型路径", "status": "pending", "detail": ""},
            {"key": "metadata", "label": "读取 GGUF 元数据", "status": "pending", "detail": ""},
            {"key": "kv", "label": "选择 KV 精度", "status": "pending", "detail": ""},
            {"key": "config", "label": "写入引擎配置", "status": "pending", "detail": ""},
            {"key": "start", "label": "启动引擎并加载模型", "status": "pending", "detail": ""},
            {"key": "smoke", "label": "冒烟测试", "status": "pending", "detail": ""},
        ]
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
            context = int(info.get("context_length") or 0)
            if context <= 0:
                raise BadRequest("GGUF 中没有 context_length；无法确定上下文上限")
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
                f"架构 {info.get('architecture') or '未知'}；上下文上限 {context:,}；ID {name}"
                + ("；视觉 mmproj 有效" if vision is not None else ""),
            )

            self._job_step(job, "kv", "running")
            help_text = engine_help_text(engine_cfg)
            if short == "llama":
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
                self._job_step(job, "kv", "ok", "k8v8（K/V 均为 q8_0）")
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
                    values["cache-type-k"] = "q8_0"
                    values["cache-type-v"] = "q8_0"
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
                if not (cache_override and self._cache_startup_error(str(exc))):
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
        if entry.source not in ("preset", "strata"):
            raise BadRequest("该模型由目录扫描发现，没有可删除的配置文件")
        removed: list[str] = []
        if entry.source == "preset":
            preset_path = Path(str(entry.config_path or ""))
            if not preset_path.is_file() or not remove_llama_preset(preset_path, entry.engine_model):
                raise BadRequest(f"models.ini 中找不到 [{entry.engine_model}] 段")
            removed.append(str(preset_path))
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
            saved = self.gateway.params.replace(entry.id, values, entry.engine)
            self._json(
                200,
                {"status": "ok", "model": entry.id, "engine": entry.engine, "params": saved},
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


def load_config(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"cannot read config: {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="llm-router gateway")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
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
    host = str(listen.get("host") or "0.0.0.0")
    port = int(listen.get("port") or 8080)
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
