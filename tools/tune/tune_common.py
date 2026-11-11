#!/usr/bin/env python3
"""Shared plumbing for the per-engine KVMem tuning scripts.

The three tuning entry points (``tune_kvmem_polystrata.py``,
``tune_kvmem_ninfer.py`` and ``tune_kvmem_llama.py``) stay independent; this
module only carries the hardware probe, the streaming GGUF/.ninfer header
readers, engine process control and the OpenAI-compatible benchmark client.

Nothing here imports ``router.py``: the scripts must be runnable on their own
from a shell for debugging.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def emit(event: str, **fields: Any) -> None:
    """One JSON event line on stdout; the router forwards these to the UI."""
    payload = {"event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def progress(message: str, pct: int | None = None, **extra: Any) -> None:
    emit("progress", message=message, pct=pct, **extra)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def write_json_atomic(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 12.0) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def probe_gpus() -> list[dict[str, Any]]:
    """NVIDIA first, ROCm second; an empty list means "unknown"."""
    gpus: list[dict[str, Any]] = []
    report = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if report is not None and report.returncode == 0:
        for line in report.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            index = _as_int(parts[0])
            total = _as_int(parts[2])
            free = _as_int(parts[3])
            if index is None or not total:
                continue
            gpus.append({
                "index": index,
                "name": parts[1],
                "vram_total_mib": total,
                "vram_free_mib": free if free is not None else total,
                "vendor": "nvidia",
            })
        if gpus:
            return gpus

    report = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if report is not None and report.returncode == 0:
        try:
            data = json.loads(report.stdout)
        except ValueError:
            data = {}
        if isinstance(data, dict):
            for key, device in data.items():
                if not isinstance(device, dict):
                    continue
                total_raw = (
                    device.get("VRAM Total Memory (B)")
                    or device.get("vram_total")
                    or device.get("VRAM Total Memory")
                )
                used_raw = (
                    device.get("VRAM Total Used Memory (B)")
                    or device.get("vram_used")
                    or device.get("VRAM Total Used Memory")
                )
                total = _as_int(total_raw)
                used = _as_int(used_raw)
                if not total:
                    continue
                # Values shown by rocm-smi in bytes; newer JSON already uses MiB.
                if total > 1024 * 1024:
                    total //= 1024 * 1024
                    used = (used or 0) // (1024 * 1024)
                index = _as_int(str(key).replace("card", "").strip())
                gpus.append({
                    "index": index if index is not None else len(gpus),
                    "name": str(device.get("Card series") or device.get("Card SKU") or key),
                    "vram_total_mib": total,
                    "vram_free_mib": max(0, total - (used or 0)),
                    "vendor": "amd",
                })
    return gpus


def probe_ram() -> dict[str, int | None]:
    total = available = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return {"ram_total_mib": total, "ram_available_mib": available}


def logical_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def physical_cores() -> int:
    """Unique core ids, not SMT threads: the engine docs tune threads by cores."""
    seen: set[str] = set()
    root = Path("/sys/devices/system/cpu")
    if root.is_dir():
        for entry in root.glob("cpu[0-9]*"):
            try:
                online = entry / "online"
                if online.is_file() and online.read_text().strip() == "0":
                    continue
                siblings = (entry / "topology" / "thread_siblings_list")
                if siblings.is_file():
                    seen.add(siblings.read_text().strip())
            except OSError:
                continue
    if seen:
        return len(seen)
    report = _run(["lscpu", "-p=CPU,CORE,SOCKET"])
    if report is not None and report.returncode == 0:
        pairs = set()
        for line in report.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                pairs.add((parts[1], parts[2]))
        if pairs:
            return len(pairs)
    return max(1, (os.cpu_count() or 2) // 2)


def probe_hardware() -> dict[str, Any]:
    gpus = probe_gpus()
    ram = probe_ram()
    cores = physical_cores()
    return {
        "gpus": gpus,
        "vram_total_mib": max((gpu["vram_total_mib"] for gpu in gpus), default=None),
        "vram_free_mib": max((gpu["vram_free_mib"] for gpu in gpus), default=None),
        "ram_total_mib": ram["ram_total_mib"],
        "ram_available_mib": ram["ram_available_mib"],
        "cpu_physical_cores": cores,
        "cpu_logical_cores": logical_cpus(),
        "cpu_model": _cpu_model(),
        "os": f"{os.uname().sysname} {os.uname().release}",
    }


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# model headers (streaming; never load multi-GB files into memory)
# ---------------------------------------------------------------------------

_GGUF_VALUE_SIZES = {
    0: 1,   # u8
    1: 1,   # i8
    2: 2,   # u16
    3: 2,   # i16
    4: 4,   # u32
    5: 4,   # i32
    6: 4,   # f32
    7: 1,   # bool
    10: 8,  # u64
    11: 8,  # i64
    12: 8,  # f64
}


class _HeaderReader:
    def __init__(self, handle, size: int):
        self.handle = handle
        self.size = size

    def read(self, count: int) -> bytes:
        if count < 0 or self.handle.tell() + count > self.size:
            raise ValueError("truncated model header")
        data = self.handle.read(count)
        if len(data) != count:
            raise ValueError("truncated model header")
        return data

    def skip(self, count: int) -> None:
        if count < 0:
            raise ValueError("negative skip")
        target = self.handle.tell() + count
        if target > self.size:
            raise ValueError("truncated model header")
        self.handle.seek(target)

    def u8(self) -> int:
        return self.read(1)[0]

    def u32(self) -> int:
        return int.from_bytes(self.read(4), "little")

    def u64(self) -> int:
        return int.from_bytes(self.read(8), "little")

    def i32(self) -> int:
        return int.from_bytes(self.read(4), "little", signed=True)

    def string(self) -> str:
        length = self.u64()
        if length > 64 * 1024 * 1024:
            raise ValueError("unreasonable string length in model header")
        return self.read(length).decode("utf-8", "replace")


def _gguf_value(reader: _HeaderReader, vtype: int, *, keep_arrays: bool = False) -> Any:
    if vtype == 8:
        return reader.string()
    if vtype == 9:
        item_type = reader.u32()
        count = reader.u64()
        if not keep_arrays or item_type in (9,) or count > 8192:
            if item_type == 8:
                for _ in range(count):
                    reader.skip(reader.u64())
            elif item_type in _GGUF_VALUE_SIZES:
                reader.skip(count * _GGUF_VALUE_SIZES[item_type])
            elif item_type == 9:
                raise ValueError("nested arrays are not supported")
            else:
                raise ValueError(f"unsupported GGUF array type {item_type}")
            return None
        return [_gguf_value(reader, item_type, keep_arrays=keep_arrays) for _ in range(count)]
    if vtype == 7:
        return bool(reader.u8())
    if vtype in (0, 2, 4, 10):
        size = _GGUF_VALUE_SIZES[vtype]
        return int.from_bytes(reader.read(size), "little")
    if vtype in (1, 3, 5, 11):
        size = _GGUF_VALUE_SIZES[vtype]
        return int.from_bytes(reader.read(size), "little", signed=True)
    if vtype == 6:
        import struct
        return struct.unpack("<f", reader.read(4))[0]
    if vtype == 12:
        import struct
        return struct.unpack("<d", reader.read(8))[0]
    raise ValueError(f"unsupported GGUF value type {vtype}")


def read_gguf_header(path: str | Path) -> dict[str, Any] | None:
    """Stream the GGUF metadata and tensor directory.

    The tokenizer arrays are skipped with seeks, so a 20+ GB GGUF costs a few
    hundred small reads rather than a full load.
    """
    target = Path(path)
    try:
        size = target.stat().st_size
        with open(target, "rb") as handle:
            reader = _HeaderReader(handle, size)
            magic = reader.read(4)
            if magic != b"GGUF":
                return None
            version = reader.u32()
            tensor_count = reader.u64()
            kv_count = reader.u64()
            if kv_count > 1_000_000 or tensor_count > 10_000_000:
                return None
            wanted = {
                "general.architecture",
                "general.name",
                "general.size_label",
                "general.file_type",
            }
            meta: dict[str, Any] = {}
            for _ in range(kv_count):
                key = reader.string()
                vtype = reader.u32()
                keep = (
                    key in wanted
                    or key.endswith(".context_length")
                    or key.endswith(".block_count")
                    or key.endswith(".embedding_length")
                    or key.endswith(".attention.head_count_kv")
                    or key.endswith(".attention.head_count")
                    or key.endswith(".attention.key_length")
                    or key.endswith(".attention.value_length")
                    or key.endswith(".expert_count")
                    or key.endswith(".expert_used_count")
                )
                value = _gguf_value(reader, vtype, keep_arrays=keep)
                if keep and value is not None:
                    meta[key] = value
            tensor_names: list[str] = []
            for _ in range(tensor_count):
                name = reader.string()
                n_dims = reader.u32()
                dims = [reader.u64() for _ in range(n_dims)]
                _type = reader.u32()
                reader.skip(8)  # data offset
                tensor_names.append(name)
    except (OSError, ValueError, OverflowError) as exc:
        emit("debug", message=f"cannot read GGUF header {target}: {exc}")
        return None

    arch = str(meta.get("general.architecture") or "")
    context = meta.get(f"{arch}.context_length") if arch else None
    if context is None:
        for key, value in meta.items():
            if key.endswith(".context_length") and isinstance(value, int):
                context = value
                break
    block_count = meta.get(f"{arch}.block_count") if arch else None
    if block_count is None:
        for key, value in meta.items():
            if key.endswith(".block_count") and isinstance(value, int):
                block_count = value
                break
    head_count_kv = meta.get(f"{arch}.attention.head_count_kv")
    head_count = meta.get(f"{arch}.attention.head_count")
    embedding = meta.get(f"{arch}.embedding_length")
    head_dim = meta.get(f"{arch}.attention.key_length")

    def _max_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, list) and value:
            numbers = [item for item in value if isinstance(item, int)]
            if numbers:
                return max(numbers)
        return None

    kv_heads = _max_int(head_count_kv)
    heads = _max_int(head_count)
    if head_dim is None and isinstance(embedding, int) and heads:
        head_dim = max(1, embedding // heads)

    lowered = "\n".join(tensor_names).lower()
    has_nextn = (
        "nextn" in lowered
        or "eh_proj" in lowered
        or "draft_head" in lowered
        or "mtp" in lowered
    )
    return {
        "path": str(target),
        "size_bytes": size,
        "size_mib": round(size / (1024 * 1024), 1),
        "format": "gguf",
        "version": version,
        "architecture": arch or None,
        "name": meta.get("general.name"),
        "context_length": int(context) if isinstance(context, int) else None,
        "block_count": int(block_count) if isinstance(block_count, int) else None,
        "embedding_length": embedding if isinstance(embedding, int) else None,
        "head_count": heads,
        "head_count_kv": kv_heads,
        "head_dim": int(head_dim) if isinstance(head_dim, int) else None,
        "has_nextn": bool(has_nextn),
        "tensor_count": len(tensor_names),
    }


def read_ninfer_header(path: str | Path) -> dict[str, Any] | None:
    """Read the JSON config embedded in a .ninfer artifact header."""
    target = Path(path)
    try:
        size = target.stat().st_size
        with open(target, "rb") as handle:
            head = handle.read(32)
            if not head.startswith(b"NINFER"):
                return None
            version = head[7]
            length = int.from_bytes(head[8:16], "little")
            if length <= 0 or length > 64 * 1024 * 1024 or 32 + length > size:
                return None
            payload = handle.read(length)
    except OSError:
        return None
    try:
        data = json.loads(payload.decode("utf-8", "replace"))
    except ValueError:
        return None
    components = data.get("components") if isinstance(data, dict) else None
    components = components if isinstance(components, dict) else {}
    text = components.get("text") if isinstance(components.get("text"), dict) else {}
    config = text.get("config") if isinstance(text.get("config"), dict) else {}
    context = config.get("max_position_embeddings") or config.get("max_context")
    layers = config.get("num_hidden_layers")
    lowered = json.dumps(components, ensure_ascii=False).lower()
    has_mtp = any(
        token in lowered for token in ("nextn", "mtp", "draft", "dflash", "spec")
    )
    return {
        "path": str(target),
        "size_bytes": size,
        "size_mib": round(size / (1024 * 1024), 1),
        "format": "ninfer",
        "version": version,
        "architecture": (
            (config.get("architectures") or [None])[0]
            if isinstance(config.get("architectures"), list) and config.get("architectures")
            else config.get("model_type")
        ),
        "context_length": int(context) if isinstance(context, (int, float)) else None,
        "block_count": int(layers) if isinstance(layers, (int, float)) else None,
        "has_nextn": bool(has_mtp),
        "components": sorted(components.keys()),
    }


# ---------------------------------------------------------------------------
# config / command construction
# ---------------------------------------------------------------------------

def repo_config(repo: Path | None = None) -> dict[str, Any]:
    data = read_json((repo or REPO_ROOT) / "config.json", {})
    return data if isinstance(data, dict) else {}


def find_engine_slot(repo: Path, kind: str) -> tuple[str, dict[str, Any]] | None:
    engines = repo_config(repo).get("engines")
    if not isinstance(engines, dict):
        return None
    for name, slot in engines.items():
        if isinstance(slot, dict) and slot.get("kind") == kind:
            return str(name), slot
    return None


def _prepend(env: dict[str, str], key: str, values: list[str], sep: str = ":") -> None:
    values = [value for value in values if value]
    if not values:
        return
    current = env.get(key)
    merged = list(dict.fromkeys(values))
    if current:
        merged.append(current)
    env[key] = sep.join(merged)


def engine_env(
    repo: Path,
    kind: str,
    slot: dict[str, Any],
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for source in (slot.get("env"), (model_cfg or {}).get("env")):
        if isinstance(source, dict):
            env.update({str(key): str(value) for key, value in source.items()})

    lib_dirs: list[str] = []
    lib_root = repo / "bin" / "libs" / "kvmem"
    for candidate in (lib_root / "cuda", lib_root / "default"):
        if candidate.is_dir():
            lib_dirs.append(str(candidate))
    # Toolchain runtimes (notably CUDA 13.x cuBLAS) are not installed beside
    # the binaries; add the managed toolchain lib directories explicitly.
    toolchains = repo / "toolchains"
    if toolchains.is_dir():
        for entry in sorted(toolchains.iterdir()):
            if not entry.is_dir():
                continue
            for sub in ("lib64", "lib", "targets/x86_64-linux/lib"):
                candidate = entry / sub
                if candidate.is_dir():
                    lib_dirs.append(str(candidate))
    _prepend(env, "LD_LIBRARY_PATH", lib_dirs)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def expand_slot_command(
    slot: dict[str, Any],
    config_path: Path,
    host: str,
    port: int,
    repo: Path,
) -> list[str]:
    command = slot.get("command")
    if not isinstance(command, list) or not command:
        raise RuntimeError("engine slot has no command")
    out = []
    cwd = str(slot.get("cwd") or repo)
    for token in command:
        value = str(token)
        value = value.replace("{config}", str(config_path))
        value = value.replace("{host}", host)
        value = value.replace("{port}", str(port))
        value = value.replace("{cwd}", cwd)
        out.append(value)
    return out


# ---------------------------------------------------------------------------
# argv helpers
# ---------------------------------------------------------------------------

def args_value(args: list[str], *flags: str) -> str | None:
    for index, token in enumerate(args):
        if token in flags:
            if index + 1 < len(args):
                return str(args[index + 1])
            return None
    return None


def args_switch(args: list[str], flag: str) -> bool:
    return any(token == flag for token in args)


def _value_follows(args: list[str], index: int) -> bool:
    if index + 1 >= len(args):
        return False
    nxt = args[index + 1]
    if not nxt.startswith("-"):
        return True
    # Negative numbers are values; -fa / --flash-attn are switches.
    return bool(re.match(r"^-\d", nxt))


def args_remove(args: list[str], *flags: str) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in flags:
            index += 1
            if _value_follows(args, index - 1):
                index += 1
            continue
        out.append(token)
        index += 1
    return out


def args_set(args: list[str], *flags_and_value: Any) -> list[str]:
    """args_set(args, "-ngl", value) or args_set(args, "--kv", None)."""
    flag = str(flags_and_value[0])
    value = flags_and_value[1] if len(flags_and_value) > 1 else None
    out = args_remove(args, flag)
    if value is not None:
        out += [flag, str(value)]
    return out


def args_set_switch(args: list[str], flag: str, enabled: bool, off_flag: str | None = None) -> list[str]:
    out = args_remove(args, flag)
    if off_flag:
        out = args_remove(out, off_flag)
    if enabled:
        out.append(flag)
    elif off_flag:
        out.append(off_flag)
    return out


def args_replace_aliases(args: list[str], aliases: list[str], value: Any) -> list[str]:
    out = args_remove(args, *aliases)
    if value is not None:
        out += [aliases[0], str(value)]
    return out


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def align(value: int, step: int) -> int:
    return max(step, value // step * step)


# ---------------------------------------------------------------------------
# engine process
# ---------------------------------------------------------------------------

class EngineError(RuntimeError):
    pass


class EngineProcess:
    def __init__(
        self,
        cmd: list[str],
        cwd: str | Path,
        env: dict[str, str],
        log_path: str | Path,
        label: str = "engine",
    ):
        self.cmd = cmd
        self.cwd = str(cwd)
        self.env = env
        self.log_path = Path(log_path)
        self.label = label
        self.proc: subprocess.Popen | None = None
        self._handle = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.log_path, "ab")
        self._handle.write(
            f"\n=== tune start {iso_now()} ===\n$ {' '.join(self.cmd)}\n".encode()
        )
        self._handle.flush()
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=self.env,
            stdout=self._handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def poll(self) -> int | None:
        return self.proc.poll() if self.proc is not None else None

    def log_tail(self, limit: int = 2000) -> str:
        try:
            data = self.log_path.read_bytes()
        except OSError:
            return ""
        text = data[-limit:].decode("utf-8", "replace")
        return text.strip()

    def stop(self, grace: float = 15.0) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    proc.terminate()
                except OSError:
                    pass
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.2)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


# ---------------------------------------------------------------------------
# HTTP benchmark client
# ---------------------------------------------------------------------------

def _opener() -> urllib.request.OpenerDirector:
    # Local engines only: ignore HTTP(S)_PROXY from the environment.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_json(url: str, payload: Any = None, timeout: float = 30.0) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with _opener().open(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode("utf-8", "replace")) if body else None


def wait_ready(
    base_url: str,
    engine: EngineProcess,
    timeout: float,
    on_tick: Callable[[float], None] | None = None,
) -> str:
    started = time.monotonic()
    deadline = started + timeout
    last_error = ""
    while time.monotonic() < deadline:
        code = engine.poll()
        if code is not None:
            raise EngineError(
                f"引擎在就绪前退出（code {code}）：{engine.log_tail()[-1200:]}"
            )
        for path in ("/health", "/v1/models"):
            try:
                data = http_json(base_url + path, timeout=3)
                if path == "/v1/models" or data is not None:
                    models = (data or {}).get("data") if isinstance(data, dict) else None
                    if path == "/v1/models":
                        if not isinstance(models, list) or not models:
                            continue
                        return str(models[0].get("id") or "model")
                    return "model"
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
                last_error = "not ready"
        if on_tick is not None:
            on_tick(time.monotonic() - started)
        time.sleep(1.0)
    raise EngineError(
        f"引擎在 {int(timeout)} 秒内未就绪（{last_error}）：{engine.log_tail()[-1200:]}"
    )


def chat_completion(
    base_url: str,
    model_id: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: float = 600.0,
) -> dict[str, Any]:
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "stream": False,
    }
    started = time.monotonic()
    data = http_json(base_url + "/v1/chat/completions", body, timeout=timeout)
    elapsed = max(0.001, time.monotonic() - started)
    if not isinstance(data, dict):
        raise EngineError("chat completion returned no JSON object")
    choices = data.get("choices") or []
    message = (choices[0] or {}).get("message") if choices else {}
    text = message.get("content") if isinstance(message, dict) else ""
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "text": text or "",
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "elapsed_s": round(elapsed, 3),
    }


_LONG_PARAGRAPH = (
    "KV cache compression keeps long conversations inside a small graphics "
    "card. A router sends each request to the engine that owns the model, and "
    "the engine keeps the attention history in host memory while it decodes. "
    "The benchmark below measures prompt ingestion and token generation on "
    "this exact machine so the suggested parameters are grounded in measured "
    "numbers rather than copied from another host. "
)


def long_prompt(tokens: int = 2048) -> str:
    # Roughly 1.3 tokens per word; the server reports the exact count.
    words = _LONG_PARAGRAPH.split()
    target = max(32, int(tokens / 1.3))
    pieces: list[str] = []
    while len(pieces) < target:
        pieces.extend(words)
    return " ".join(pieces[:target]) + "\n\nSummarize the text above in one short sentence."


def benchmark(
    base_url: str,
    model_id: str,
    decode_tokens: int = 96,
    prefill_tokens: int = 2048,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Measure decode and prefill; failures bubble up to the candidate record."""
    warm = chat_completion(
        base_url, model_id,
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=8, timeout=min(timeout, 300),
    )
    decode = chat_completion(
        base_url, model_id,
        [{"role": "user", "content": "Count from 1 to 60, one number per line."}],
        max_tokens=decode_tokens, timeout=timeout,
    )
    decode_tps = (
        decode["completion_tokens"] / decode["elapsed_s"]
        if decode["completion_tokens"] > 0 else 0.0
    )
    result = {
        "warmup_s": warm["elapsed_s"],
        "decode_tps": round(decode_tps, 2),
        "decode_tokens": decode["completion_tokens"],
        "decode_s": decode["elapsed_s"],
    }
    prefill = chat_completion(
        base_url, model_id,
        [{"role": "user", "content": long_prompt(prefill_tokens)}],
        max_tokens=1, timeout=timeout,
    )
    prompt_tokens = prefill["prompt_tokens"]
    result.update({
        "prefill_tps": round(prompt_tokens / prefill["elapsed_s"], 2) if prompt_tokens else 0.0,
        "prefill_tokens": prompt_tokens,
        "prefill_s": prefill["elapsed_s"],
    })
    return result


def score(candidate: dict[str, Any]) -> float:
    """Decode first; prefill breaks ties within 2 percent."""
    decode = float(candidate.get("decode_tps") or 0.0)
    prefill = float(candidate.get("prefill_tps") or 0.0)
    return decode + min(prefill, decode * 0.02) * 0.5


def best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [item for item in candidates if item.get("ok")]
    if not usable:
        return None
    return max(usable, key=score)


# ---------------------------------------------------------------------------
# candidate runner
# ---------------------------------------------------------------------------

def run_candidates(
    candidates: list[dict[str, Any]],
    launch: Callable[[dict[str, Any], int, Path], tuple[EngineProcess, str, str]],
    workdir: Path,
    startup_timeout: float,
    bench_timeout: float,
    decode_tokens: int = 96,
    prefill_tokens: int = 2048,
    max_seconds: float = 1500.0,
) -> list[dict[str, Any]]:
    """Start one engine per candidate, benchmark it, stop it, repeat."""
    started = time.monotonic()
    for index, candidate in enumerate(candidates):
        if time.monotonic() - started > max_seconds:
            candidate.setdefault("ok", False)
            candidate["error"] = "调优总时长上限已到，未运行该候选"
            emit("candidate", index=index, **candidate)
            continue
        label = candidate["label"]
        progress(f"[{index + 1}/{len(candidates)}] 启动引擎：{label}")
        port = free_port()
        engine: EngineProcess | None = None
        try:
            engine, base_url, model_id = launch(candidate, port, workdir / f"c{index + 1}")
            progress(f"[{index + 1}/{len(candidates)}] 等待 {label} 就绪…")
            wait_ready(base_url, engine, startup_timeout)
            progress(f"[{index + 1}/{len(candidates)}] 基准测试：{label}")
            metrics = benchmark(
                base_url, model_id,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                timeout=bench_timeout,
            )
            candidate.update(metrics)
            candidate["ok"] = True
            candidate.pop("error", None)
            progress(
                f"{label}：decode {candidate['decode_tps']} tok/s，"
                f"prefill {candidate.get('prefill_tps', 0)} tok/s"
            )
        except (EngineError, urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError, ValueError) as exc:
            candidate["ok"] = False
            candidate["error"] = str(exc)[:600]
            progress(f"{label} 失败：{candidate['error'][:200]}")
        finally:
            if engine is not None:
                engine.stop()
            # Give the driver a moment to release VRAM before the next model.
            time.sleep(2.0)
        emit("candidate", index=index, **candidate)
    return candidates


def diff_params(
    before_args: list[str],
    after_args: list[str],
    specs: list[tuple[str, str, Any, str | None]],
) -> list[dict[str, Any]]:
    """Build the UI diff.

    Each spec is ``(key, label, flags, kind)`` where flags is one flag or a
    tuple of aliases, and kind is ``value``, ``switch`` or ``off``.
    """
    changes: list[dict[str, Any]] = []
    for key, label, flag, kind in specs:
        flags = tuple(flag) if isinstance(flag, (tuple, list)) else (flag,)
        before = args_value(before_args, *flags) if flags else None
        after = args_value(after_args, *flags) if flags else None
        if kind in ("switch", "off"):
            before = "on" if flags and args_switch(before_args, flags[0]) else "off"
            after = "on" if flags and args_switch(after_args, flags[0]) else "off"
        if before == after:
            continue
        changes.append({
            "key": key,
            "label": label,
            "before": before,
            "after": after,
        })
    return changes
