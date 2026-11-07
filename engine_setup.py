#!/usr/bin/env python3
"""Engine provisioning for llm-router: toolchains, sources, builds and updates.

Everything below the project directory is owned by this module:

    engine/<engine>      git checkout of the engine source
    bin/                 compiled executables (`bin/libs/<engine>[/cuda|rocm]` for shared objects)
    config/<engine>/     model / engine configuration files
    log/                 engine and setup logs
    toolchains/          downloaded CUDA / ROCm toolkits
    tmp/downloads/       download staging area

The module never imports router.py; the gateway calls it from its HTTP handlers
and keeps the ``setup`` section of config.json in sync through ``persist``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import requests


# ---------------------------------------------------------------------------
# requirements

# llama.cpp 0.9.0 hard-fails in ggml/src/ggml-hip/CMakeLists.txt below HIP 6.1;
# CUDA has no hard CMake floor, the upstream docs build with 11.7.
MIN_CUDA = (11, 7)
MIN_ROCM = (6, 1)

CUDA_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "version": "11.8.0",
        "url": "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run",
        "filename": "cuda_11.8.0_520.61.05_linux.run",
        "bytes": 4_400_000_000,
    },
    {
        "version": "12.9.1",
        "url": "https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run",
        "filename": "cuda_12.9.1_575.57.08_linux.run",
        "bytes": 5_100_000_000,
    },
    {
        "version": "13.5.0",
        "url": "https://developer.download.nvidia.com/compute/cuda/13.5.0/local_installers/cuda_13.5.0_linux.run",
        "filename": "cuda_13.5.0_linux.run",
        "bytes": 6_000_000_000,
    },
)

# TheRock distribution tarballs.  AMD's relocatable packages start at 7.10
# (TheRock tags therock-7.10 … therock-10.1); ROCm 6.x was only ever shipped as
# distro packages, so MIN_ROCM stays 6.1 for manually installed toolkits while
# the download list starts at 7.10.  Channels differ per series: 10.x lives on
# the stable index, 7.14.x only on the multi-arch index, older 7.x on the
# legacy tarball index.  The file-name suffix is not uniform (-all/-dgpu/-dcgpu
# or bare), so the exact names are listed per family.
ROCM_TARBALL = "https://repo.amd.com/rocm/tarball/"
ROCM_CHANNELS: dict[str, str] = {
    "10.1.0": "https://stable.repo.amd.com/rocm/core/tarball/",
    "10.0.0": "https://stable.repo.amd.com/rocm/core/tarball/",
    "7.14.1": "https://repo.amd.com/rocm/tarball-multi-arch/",
    "7.14.0": "https://repo.amd.com/rocm/tarball-multi-arch/",
}

ROCM_PACKAGES: dict[str, tuple[tuple[str, str], ...]] = {
    "gfx101X": (
        ("7.13.0", "therock-dist-linux-gfx101X-dgpu-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx101X-dgpu-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx101X-dgpu-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx101X-dgpu-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx101X-dgpu-10.1.0.tar.gz"),
    ),
    "gfx103X": (
        ("7.13.0", "therock-dist-linux-gfx103X-all-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx103X-all-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx103X-all-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx103X-all-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx103X-all-10.1.0.tar.gz"),
    ),
    "gfx110X": (
        ("7.10.0", "therock-dist-linux-gfx110X-dgpu-7.10.0.tar.gz"),
        ("7.11.0", "therock-dist-linux-gfx110X-all-7.11.0.tar.gz"),
        ("7.12.0", "therock-dist-linux-gfx110X-all-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx110X-all-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx110X-all-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx110X-all-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx110X-all-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx110X-all-10.1.0.tar.gz"),
    ),
    "gfx1150": (
        ("7.10.0", "therock-dist-linux-gfx1150-7.10.0.tar.gz"),
        ("7.11.0", "therock-dist-linux-gfx1150-7.11.0.tar.gz"),
        ("7.12.0", "therock-dist-linux-gfx1150-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx1150-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx1150-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx1150-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx1150-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx1150-10.1.0.tar.gz"),
    ),
    "gfx1151": (
        ("7.10.0", "therock-dist-linux-gfx1151-7.10.0.tar.gz"),
        ("7.11.0", "therock-dist-linux-gfx1151-7.11.0.tar.gz"),
        ("7.12.0", "therock-dist-linux-gfx1151-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx1151-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx1151-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx1151-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx1151-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx1151-10.1.0.tar.gz"),
    ),
    "gfx1152": (
        ("7.11.0", "therock-dist-linux-gfx1152-7.11.0.tar.gz"),
        ("7.12.0", "therock-dist-linux-gfx1152-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx1152-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx1152-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx1152-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx1152-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx1152-10.1.0.tar.gz"),
    ),
    "gfx1153": (
        ("7.14.0", "therock-dist-linux-gfx1153-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx1153-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx1153-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx1153-10.1.0.tar.gz"),
    ),
    "gfx120X": (
        ("7.10.0", "therock-dist-linux-gfx120X-all-7.10.0.tar.gz"),
        ("7.11.0", "therock-dist-linux-gfx120X-all-7.11.0.tar.gz"),
        ("7.12.0", "therock-dist-linux-gfx120X-all-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx120X-all-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx120X-all-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx120X-all-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx120X-all-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx120X-all-10.1.0.tar.gz"),
    ),
    "gfx125X": (
        ("7.14.0", "therock-dist-linux-gfx125X-dcgpu-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx125X-dcgpu-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx125X-dcgpu-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx125X-dcgpu-10.1.0.tar.gz"),
    ),
    "gfx908": (
        ("7.12.0", "therock-dist-linux-gfx908-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx908-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx908-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx908-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx908-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx908-10.1.0.tar.gz"),
    ),
    "gfx90X": (
        ("7.10.0", "therock-dist-linux-gfx90X-dcgpu-7.10.0.tar.gz"),
        ("7.11.0", "therock-dist-linux-gfx90X-dcgpu-7.11.0.tar.gz"),
    ),
    "gfx90A": (
        ("7.12.0", "therock-dist-linux-gfx90a-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx90a-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx90a-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx90a-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx90a-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx90a-10.1.0.tar.gz"),
    ),
    "gfx94X": (
        ("7.10.0", "therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz"),
        ("7.11.0", "therock-dist-linux-gfx94X-dcgpu-7.11.0.tar.gz"),
        ("7.12.0", "therock-dist-linux-gfx94X-dcgpu-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx94X-dcgpu-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx94X-dcgpu-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx94X-dcgpu-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx94X-dcgpu-10.1.0.tar.gz"),
    ),
    "gfx950": (
        ("7.10.0", "therock-dist-linux-gfx950-dcgpu-7.10.0.tar.gz"),
        ("7.11.0", "therock-dist-linux-gfx950-dcgpu-7.11.0.tar.gz"),
        ("7.12.0", "therock-dist-linux-gfx950-dcgpu-7.12.0.tar.gz"),
        ("7.13.0", "therock-dist-linux-gfx950-dcgpu-7.13.0.tar.gz"),
        ("7.14.0", "therock-dist-linux-gfx950-dcgpu-7.14.0.tar.gz"),
        ("7.14.1", "therock-dist-linux-gfx950-dcgpu-7.14.1.tar.gz"),
        ("10.0.0", "therock-dist-linux-gfx950-dcgpu-10.0.0.tar.gz"),
        ("10.1.0", "therock-dist-linux-gfx950-dcgpu-10.1.0.tar.gz"),
    ),
}

# NInfer's static FFmpeg build needs meson and nasm; meson comes from pip,
# nasm is built once into toolchains/nasm when the host lacks it.
NASM_VERSION = "2.16.03"
NASM_URL = f"https://www.nasm.us/pub/nasm/releasebuilds/{NASM_VERSION}/nasm-{NASM_VERSION}.tar.xz"

ROCM_FAMILIES: tuple[tuple[str, str], ...] = (
    ("gfx1153", "gfx1153"),
    ("gfx1152", "gfx1152"),
    ("gfx1151", "gfx1151"),
    ("gfx1150", "gfx1150"),
    ("gfx110", "gfx110X"),
    ("gfx125", "gfx125X"),
    ("gfx120", "gfx120X"),
    ("gfx103", "gfx103X"),
    ("gfx101", "gfx101X"),
    ("gfx950", "gfx950"),
    ("gfx942", "gfx94X"),
    ("gfx940", "gfx94X"),
    ("gfx90a", "gfx90A"),
    ("gfx908", "gfx908"),
)

ENGINE_CATALOG: dict[str, dict[str, Any]] = {
    "llama": {
        "label": "llama.cpp",
        "repo": "https://github.com/ggml-org/llama.cpp",
        "needs": (),
        "submodules": False,
        "artifacts": {
            "cuda": (("llama-server", "llama-server"), ("llama-mtmd-cli", "llama-mtmd-cli")),
            "rocm": (("llama-server", "llama-server"), ("llama-mtmd-cli", "llama-mtmd-cli")),
        },
    },
    "strata": {
        "label": "Strata",
        "repo": "https://github.com/Niko1221/Strata",
        "needs": ("llama",),
        "submodules": True,
        "artifacts": {
            "cuda": (("strata", "strata"), ("strata-vision", "strata-vision")),
            "rocm": (("strata", "strata"), ("strata-vision", "strata-vision")),
        },
    },
    "kvmem": {
        "label": "KVMem",
        "repo": "https://github.com/kvmem/kvmem",
        "needs": ("llama",),
        "submodules": True,
        "artifacts": {
            "cuda": (
                ("llama-kvmem-server", "kvmem-llama-server"),
                ("strata-poly", "kvmem-strata-poly"),
                ("ninfer-serve", "kvmem-ninfer-serve"),
            ),
            "rocm": (
                ("llama-kvmem-server", "kvmem-llama-server"),
                ("strata-poly", "kvmem-strata-poly"),
            ),
        },
    },
}

ENGINE_IDS = ("llama", "strata", "kvmem")
VARIANT_LABEL = {"cuda": "CUDA", "rocm": "ROCm"}


class SetupError(Exception):
    """A user-facing setup failure."""


# ---------------------------------------------------------------------------
# small helpers


def parse_version(text: Any) -> tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    if not match:
        return ()
    return tuple(int(part) for part in match.groups() if part is not None)


def version_text(value: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in value)


def version_ok(value: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    if not value:
        return False
    width = max(len(value), len(minimum))
    a = value + (0,) * (width - len(value))
    b = minimum + (0,) * (width - len(minimum))
    return a >= b


def proxy_url(proxy_cfg: dict[str, Any] | None) -> str | None:
    if not isinstance(proxy_cfg, dict):
        return None
    kind = str(proxy_cfg.get("type") or "http").strip().lower()
    scheme = "socks5h" if kind == "socks5" else "http"
    host = str(proxy_cfg.get("host") or "").strip()
    try:
        port = int(proxy_cfg.get("port") or 0)
    except (TypeError, ValueError):
        return None
    if not host or not port:
        return None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    username = str(proxy_cfg.get("username") or "")
    password = str(proxy_cfg.get("password") or "")
    auth = ""
    if username:
        from urllib.parse import quote

        auth = quote(username, safe="")
        if password:
            auth += ":" + quote(password, safe="")
        auth += "@"
    return f"{scheme}://{auth}{host}:{port}"


def apply_proxy(env: dict[str, str], proxy_cfg: dict[str, Any] | None, use_proxy: bool) -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    if not use_proxy:
        return
    url = proxy_url(proxy_cfg)
    if not url:
        return
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[key] = url


def requests_proxies(proxy_cfg: dict[str, Any] | None, use_proxy: bool) -> dict[str, str] | None:
    url = proxy_url(proxy_cfg) if use_proxy else None
    if not url:
        return None
    return {"http": url, "https": url}


def normalize_remote(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if text.startswith("git@") and ":" in text:
        host, path = text[4:].split(":", 1)
        text = f"https://{host}/{path}"
    text = text.rstrip("/")
    if text.endswith(".git"):
        text = text[:-4]
    if text.startswith("http://"):
        text = "https://" + text[len("http://"):]
    return text.lower()


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


def tail(path: Path, lines: int = 30) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(data[-lines:])
    except OSError:
        return ""


def _read_sys(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return ""


def _glob_lib(root: Path, pattern: str) -> list[Path]:
    found: list[Path] = []
    bases = [root / "lib64", root / "lib"]
    # Distro packages (for example a /usr toolkit) put the runtime in the
    # multiarch directory instead of lib/ or lib64/.
    bases.extend(sorted((root / "lib").glob("*-linux-gnu*")))
    for base in bases:
        if base.is_dir():
            found.extend(sorted(base.glob(pattern)))
    return found


# ---------------------------------------------------------------------------
# GPU discovery


def _sysfs_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    seen: set[str] = set()
    drm = Path("/sys/class/drm")
    if not drm.is_dir():
        return gpus
    for card in sorted(drm.glob("card[0-9]*")):
        device = card / "device"
        try:
            real = os.path.realpath(device)
        except OSError:
            continue
        if real in seen:
            continue
        vendor = _read_sys(device / "vendor").lower()
        klass = _read_sys(device / "class").lower()
        if vendor not in ("0x10de", "0x1002"):
            continue
        if klass and not (klass.startswith("0x0300") or klass.startswith("0x0380")):
            continue
        seen.add(real)
        gpus.append(
            {
                "vendor": "nvidia" if vendor == "0x10de" else "amd",
                "name": "NVIDIA GPU" if vendor == "0x10de" else "AMD GPU",
                "arch": "",
            }
        )
    return gpus


def nvidia_devices() -> list[dict[str, Any]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, cap = line.rpartition(",")
        out.append(
            {
                "vendor": "nvidia",
                "name": (name or line).strip(),
                "arch": (cap or "").strip(),
            }
        )
    return out


def amd_arch() -> str:
    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        try:
            proc = subprocess.run([rocminfo], capture_output=True, text=True, timeout=15)
            match = re.search(r"\bgfx[0-9A-Za-z]+\b", proc.stdout)
            if match:
                return match.group(0)
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


_GPU_CACHE: tuple[float, list[dict[str, Any]]] | None = None


def detect_gpus(force: bool = False) -> list[dict[str, Any]]:
    global _GPU_CACHE
    if not force and _GPU_CACHE is not None and time.time() - _GPU_CACHE[0] < 10:
        return [dict(item) for item in _GPU_CACHE[1]]
    gpus = _sysfs_gpus()
    nvidia = nvidia_devices()
    for item in gpus:
        if item["vendor"] == "nvidia":
            if nvidia:
                item["name"] = nvidia[0]["name"]
                item["arch"] = nvidia[0]["arch"]
        elif item["vendor"] == "amd":
            arch = amd_arch()
            if arch:
                item["arch"] = arch
                item["name"] = f"AMD GPU ({arch})"
    if not gpus:
        for item in nvidia:
            gpus.append({"vendor": "nvidia", **item})
        if not gpus and amd_arch():
            arch = amd_arch()
            gpus.append({"vendor": "amd", "name": f"AMD GPU ({arch})", "arch": arch})
    # One entry per vendor/architecture pair is enough for toolchain gating.
    unique: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in gpus:
        marker = (item["vendor"], str(item.get("arch") or item.get("name") or ""))
        if marker in seen_keys:
            continue
        seen_keys.add(marker)
        unique.append(item)
    gpus = unique
    _GPU_CACHE = (time.time(), [dict(item) for item in gpus])
    return gpus


def rocm_family(arch: str) -> str:
    arch = str(arch or "").strip().lower()
    for prefix, family in ROCM_FAMILIES:
        if arch.startswith(prefix.lower()):
            return family
    return ""


def rocm_catalog(arch: str) -> list[dict[str, Any]]:
    """Published TheRock packages for the detected GPU family, oldest first."""
    entries: list[dict[str, Any]] = []
    for version, filename in ROCM_PACKAGES.get(rocm_family(arch), ()):
        if not version_ok(parse_version(version), MIN_ROCM):
            continue
        channel = ROCM_CHANNELS.get(version, ROCM_TARBALL)
        entries.append(
            {
                "version": version,
                "url": channel + filename,
                "filename": filename,
                "bytes": 3_750_000_000,
            }
        )
    return entries


def nvidia_arch_for_ninfer(cap: str) -> str:
    text = str(cap or "").strip()
    match = re.match(r"(\d+)\.(\d+)", text)
    if not match:
        return "89"
    major, minor = int(match.group(1)), int(match.group(2))
    if major >= 12:
        return "120a"
    return f"{major}{minor}"


# ---------------------------------------------------------------------------
# toolchain inspection


def _cuda_candidate(path: Path) -> Path:
    """Accept /usr/local, /usr/local/cuda or a versioned toolkit root."""
    if (path / "bin" / "nvcc").is_file():
        return path
    for name in ("cuda",):
        candidate = path / name
        if (candidate / "bin" / "nvcc").is_file():
            return candidate
    matches = sorted(path.glob("cuda-*"), reverse=True)
    for candidate in matches:
        if (candidate / "bin" / "nvcc").is_file():
            return candidate
    return path


def _rocm_candidate(path: Path) -> Path:
    if (path / "bin" / "hipcc").is_file() or (path / ".info" / "version").is_file():
        return path
    for name in ("rocm",):
        candidate = path / name
        if (candidate / "bin" / "hipcc").is_file():
            return candidate
    matches = sorted(path.glob("core-*"), reverse=True)
    for candidate in matches:
        if (candidate / "bin" / "hipcc").is_file():
            return candidate
    return path


def inspect_cuda(path_value: Any) -> dict[str, Any]:
    raw = str(path_value or "").strip()
    out: dict[str, Any] = {
        "kind": "cuda", "path": raw, "ok": False, "version": "", "reason": "",
        "nvcc": "", "headers": False, "runtime": False, "blas": False, "minimum": version_text(MIN_CUDA),
    }
    if not raw:
        out["reason"] = "未配置 CUDA 路径"
        return out
    path = Path(raw).expanduser()
    if not path.exists():
        out["reason"] = f"路径不存在：{path}"
        return out
    root = _cuda_candidate(path)
    nvcc = root / "bin" / "nvcc"
    if not nvcc.is_file():
        out["reason"] = f"未找到 nvcc，{root} 不是 CUDA 编译包"
        return out
    out["nvcc"] = str(nvcc)
    env = os.environ.copy()
    env["PATH"] = f"{root / 'bin'}:{env.get('PATH', '')}"
    try:
        proc = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=20, env=env)
        match = re.search(r"release\s+(\d+\.\d+)", proc.stdout)
        version = match.group(1) if match else ""
    except (OSError, subprocess.SubprocessError) as exc:
        out["reason"] = f"无法运行 nvcc：{exc}"
        return out
    if not version:
        out["reason"] = "无法从 nvcc --version 读取 CUDA 版本"
        return out
    out["version"] = version
    header = root / "include" / "cuda_runtime.h"
    if not header.is_file():
        header = next(iter(sorted(root.glob("targets/*/include/cuda_runtime.h"))), header)
    out["headers"] = header.is_file()
    out["runtime"] = bool(_glob_lib(root, "libcudart.so*"))
    out["blas"] = bool(_glob_lib(root, "libcublas.so*"))
    missing = []
    if not out["headers"]:
        missing.append("cuda_runtime.h")
    if not out["runtime"]:
        missing.append("libcudart")
    if not out["blas"]:
        missing.append("libcublas")
    if missing:
        out["reason"] = "CUDA 编译包不完整，缺少：" + "、".join(missing)
        return out
    if not version_ok(parse_version(version), MIN_CUDA):
        out["reason"] = f"CUDA {version} 低于最低要求 {version_text(MIN_CUDA)}"
        return out
    out["ok"] = True
    out["resolved_path"] = str(root)
    return out


def _toolchain_candidates(kind: str, base_dir: Path) -> list[Path]:
    """Standard places an installed toolkit can live.

    Managed downloads land in ``toolchains/<kind>-<version>``; everything else
    is the host's normal install location or whatever ``nvcc``/``hipcc`` on
    PATH resolves to."""
    roots: list[Path] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        raw = str(value or "").strip()
        if not raw:
            return
        path = Path(raw).expanduser()
        if not path.is_dir():
            return
        key = os.path.realpath(str(path))
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    pattern = "cuda-*" if kind == "cuda" else "rocm-*"
    try:
        for path in sorted((Path(base_dir) / "toolchains").glob(pattern), reverse=True):
            add(path)
    except OSError:
        pass
    which = shutil.which("nvcc" if kind == "cuda" else "hipcc")
    if which:
        add(Path(which).resolve().parent.parent)
    if kind == "cuda":
        for env in ("CUDA_HOME", "CUDA_PATH"):
            add(os.environ.get(env))
        add("/usr/local/cuda")
        try:
            for path in sorted(Path("/usr/local").glob("cuda-*"), reverse=True):
                add(path)
        except OSError:
            pass
        add("/opt/cuda")
    else:
        add(os.environ.get("ROCM_PATH"))
        add("/opt/rocm")
        try:
            for path in sorted(Path("/opt").glob("rocm-*"), reverse=True):
                add(path)
        except OSError:
            pass
    return roots


def detect_installed_toolchain(kind: str, base_dir: Path) -> dict[str, Any]:
    """The best usable toolkit of this kind, or why a present one is unusable.

    Managed installs and host locations are inspected; the newest complete
    toolkit wins.  An installation that exists but is older than the minimum
    (or misses compiler pieces) stays visible as installed-but-unusable so the
    UI can ask for a reinstall instead of pretending nothing is there."""
    inspect = inspect_cuda if kind == "cuda" else inspect_rocm
    label = "CUDA" if kind == "cuda" else "ROCm"
    minimum = MIN_CUDA if kind == "cuda" else MIN_ROCM
    label_minimum = version_text(minimum)
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _toolchain_candidates(kind, base_dir):
        report = inspect(str(path))
        root = str(report.get("resolved_path") or report.get("path") or path)
        key = os.path.realpath(root)
        if key in seen:
            continue
        seen.add(key)
        report["path"] = root
        report["installed"] = bool(
            report.get("version") or report.get("nvcc" if kind == "cuda" else "hipcc")
        )
        reports.append(report)
    working = [item for item in reports if item["ok"]]
    if working:
        best = max(working, key=lambda item: parse_version(item["version"]))
        best["installed"] = True
        return best
    old = [
        item for item in reports
        if item.get("version") and not version_ok(parse_version(item["version"]), minimum)
    ]
    if old:
        best = max(old, key=lambda item: parse_version(item["version"]))
        best["installed"] = True
        best["needs_reinstall"] = True
        best["reason"] = (
            f"已安装 {label} {best['version']}（{best['path']}），"
            f"低于最低要求 {label_minimum}，必须重新安装"
        )
        return best
    broken = [item for item in reports if item.get("installed") and not item["ok"]]
    if broken:
        best = broken[0]
        best["needs_reinstall"] = True
        best["reason"] = f"已检测到 {label}（{best['path']}），{best['reason']}，必须重新安装"
        return best
    return {
        "kind": kind, "path": "", "ok": False, "version": "", "installed": False,
        "reason": f"未检测到 {label}，请选择版本安装", "minimum": label_minimum,
    }


def inspect_rocm(path_value: Any) -> dict[str, Any]:
    raw = str(path_value or "").strip()
    out: dict[str, Any] = {
        "kind": "rocm", "path": raw, "ok": False, "version": "", "reason": "",
        "hipcc": "", "headers": False, "runtime": False, "blas": False, "minimum": version_text(MIN_ROCM),
    }
    if not raw:
        out["reason"] = "未配置 ROCm 路径"
        return out
    path = Path(raw).expanduser()
    if not path.exists():
        out["reason"] = f"路径不存在：{path}"
        return out
    root = _rocm_candidate(path)
    hipcc = root / "bin" / "hipcc"
    header = root / "include" / "hip" / "hip_runtime.h"
    version = ""
    info = root / ".info" / "version"
    if info.is_file():
        version = _read_sys(info)
    elif (root / "lib" / "rocm_version").is_file():
        version = _read_sys(root / "lib" / "rocm_version")
    if not version and hipcc.is_file():
        try:
            proc = subprocess.run([str(hipcc), "--version"], capture_output=True, text=True, timeout=20)
            match = re.search(r"(?:HIP|ROCm)[^\d]*(\d+\.\d+(?:\.\d+)?)", proc.stdout)
            if match:
                version = match.group(1)
        except (OSError, subprocess.SubprocessError):
            pass
    if not hipcc.is_file() and not version:
        out["reason"] = f"未找到 hipcc 或版本文件，{root} 不是 ROCm 编译包"
        return out
    out["hipcc"] = str(hipcc) if hipcc.is_file() else ""
    out["version"] = version
    out["headers"] = header.is_file()
    out["runtime"] = bool(_glob_lib(root, "libamdhip64.so*") or _glob_lib(root, "libamdhip64.so"))
    out["blas"] = bool(_glob_lib(root, "libhipblas.so*") or _glob_lib(root, "libhipblas.so"))
    missing = []
    if not hipcc.is_file():
        missing.append("hipcc")
    if not out["headers"]:
        missing.append("hip/hip_runtime.h")
    if not out["runtime"]:
        missing.append("libamdhip64")
    if not out["blas"]:
        missing.append("libhipblas")
    if missing:
        out["reason"] = "ROCm 编译包不完整，缺少：" + "、".join(missing)
        return out
    if not version_ok(parse_version(version), MIN_ROCM):
        out["reason"] = f"ROCm {version} 低于最低要求 {version_text(MIN_ROCM)}"
        return out
    out["ok"] = True
    out["resolved_path"] = str(root)
    return out


# ---------------------------------------------------------------------------
# engine setup


class EngineSetup:
    def __init__(
        self,
        base_dir: Path,
        log: Any,
        config: dict[str, Any],
        persist: Callable[[], None] | None = None,
        proxy_cfg: Callable[[], dict[str, Any] | None] | None = None,
        stop_engine: Callable[[], bool] | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.log = log
        self.config = config
        self.persist = persist
        self.proxy_cfg = proxy_cfg or (lambda: None)
        self.stop_engine = stop_engine
        self.dir_engine = self.base_dir / "engine"
        self.dir_bin = self.base_dir / "bin"
        self.dir_config = self.base_dir / "config"
        self.dir_log = self.base_dir / "log"
        self.dir_toolchains = self.base_dir / "toolchains"
        self.dir_tmp = self.base_dir / "tmp" / "downloads"
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._procs: dict[str, subprocess.Popen[Any]] = {}
        self._job_logs: dict[str, Path] = {}
        self._report_cache: tuple[float, dict[str, Any]] | None = None
        self._ensure_dirs()

    # -- layout ------------------------------------------------------------
    def _ensure_dirs(self) -> None:
        for path in (
            self.dir_engine,
            self.dir_bin,
            self.dir_config,
            self.dir_log,
            self.dir_log / "setup",
            self.dir_toolchains,
            self.dir_tmp,
            self.dir_config / "llama.cpp",
            self.dir_config / "strata",
            self.dir_config / "polystrata",
            self.dir_config / "ninfer",
            self.dir_config / "kvmem-llama",
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.log.warning("cannot create %s: %s", path, exc)

    # -- toolchains --------------------------------------------------------
    def toolchains(self) -> dict[str, Any]:
        if self._report_cache is not None and time.time() - self._report_cache[0] < 8:
            return self._report_cache[1]
        gpus = detect_gpus()
        vendors = {item["vendor"] for item in gpus}
        cuda = detect_installed_toolchain("cuda", self.base_dir)
        rocm = detect_installed_toolchain("rocm", self.base_dir)
        result = {
            "gpus": gpus,
            "show_cuda": "nvidia" in vendors,
            "show_rocm": "amd" in vendors,
            "cuda": cuda,
            "rocm": rocm,
            "cuda_catalog": list(CUDA_CATALOG),
            "rocm_catalog": rocm_catalog(self._amd_arch(gpus)),
            "rocm_family": rocm_family(self._amd_arch(gpus)),
            "minimum": {"cuda": version_text(MIN_CUDA), "rocm": version_text(MIN_ROCM)},
        }
        self._report_cache = (time.time(), result)
        return result

    @staticmethod
    def _amd_arch(gpus: list[dict[str, Any]]) -> str:
        for item in gpus:
            if item.get("vendor") == "amd" and item.get("arch"):
                return str(item["arch"])
        return "gfx1100"

    def active_variants(self) -> dict[str, dict[str, Any]]:
        """Usable toolchains for the GPUs on this machine, in build order."""
        reports = self.toolchains()
        variants: dict[str, dict[str, Any]] = {}
        if reports["show_cuda"] and reports["cuda"]["ok"]:
            variants["cuda"] = reports["cuda"]
        if reports["show_rocm"] and reports["rocm"]["ok"]:
            variants["rocm"] = reports["rocm"]
        return variants

    # -- status ------------------------------------------------------------
    def _engine_dir(self, engine_id: str) -> Path:
        return self.dir_engine / ("llama.cpp" if engine_id == "llama" else "Strata" if engine_id == "strata" else "kvmem")

    def _git(self, cwd: Path, *args: str, env: dict[str, str] | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, env=env, check=False,
        )

    def _repo_state(self, engine_id: str) -> dict[str, Any]:
        spec = ENGINE_CATALOG[engine_id]
        root = self._engine_dir(engine_id)
        out: dict[str, Any] = {
            "dir": str(root),
            "exists": root.is_dir(),
            "repo": False,
            "remote": "",
            "expected_remote": spec["repo"],
            "remote_ok": False,
            "version": "",
            "dirty": False,
            "source_ok": False,
            "reason": "",
        }
        if not root.is_dir():
            return out
        if not (root / ".git").exists():
            out["reason"] = f"{root} 不是 git 仓库，请重新下载引擎"
            return out
        out["repo"] = True
        proc = self._git(root, "remote", "get-url", "origin", timeout=20)
        remote = proc.stdout.strip() if proc.returncode == 0 else ""
        out["remote"] = remote
        out["remote_ok"] = normalize_remote(remote) == normalize_remote(spec["repo"])
        if not out["remote_ok"]:
            out["reason"] = f"git remote 不匹配（{remote or '无 origin'}），请重新下载引擎"
        proc = self._git(root, "describe", "--tags", "--always", timeout=20)
        out["version"] = proc.stdout.strip() if proc.returncode == 0 else ""
        # KVMem keeps its integration patches inside the backend submodules,
        # so submodule modifications are expected and not a user edit.
        status_args = ["status", "--porcelain"]
        if engine_id == "kvmem":
            status_args.append("--ignore-submodules=all")
        proc = self._git(root, *status_args, timeout=30)
        out["dirty"] = bool(proc.stdout.strip()) if proc.returncode == 0 else False
        out["submodules_ok"] = True
        if spec["submodules"]:
            sub = self._git(root, "submodule", "status", "--recursive", timeout=120)
            missing = [line for line in sub.stdout.splitlines() if line.startswith("-")]
            out["submodules_ok"] = sub.returncode == 0 and not missing
            if missing and not out["reason"]:
                names = ", ".join(line.split()[1] for line in missing[:4])
                out["reason"] = f"子模块未下载：{names}，请重新下载引擎"
        out["source_ok"] = out["remote_ok"] and out["submodules_ok"]
        return out

    def _bin_name(self, engine_id: str, artifact: str, variant: str | None) -> str:
        spec = ENGINE_CATALOG[engine_id]
        final = artifact
        for source, target in spec["artifacts"].get(variant or "cuda", ()):
            if source == artifact:
                final = target
                break
        return f"{final}-{variant}" if variant and self._dual() else final

    def _dual(self) -> bool:
        return len(self.active_variants()) > 1

    def _bin_paths(self, engine_id: str, variant: str) -> list[tuple[str, Path]]:
        spec = ENGINE_CATALOG[engine_id]
        return [
            (target, self.dir_bin / self._bin_name(engine_id, source, variant))
            for source, target in spec["artifacts"].get(variant, ())
        ]

    def engine_status(self, engine_id: str) -> dict[str, Any]:
        spec = ENGINE_CATALOG[engine_id]
        repo = self._repo_state(engine_id)
        variants = self.active_variants()
        variant_views: dict[str, Any] = {}
        complete = bool(variants)
        for variant in variants:
            binaries = [
                {"name": path.name, "path": str(path), "exists": path.is_file()}
                for _, path in self._bin_paths(engine_id, variant)
            ]
            ok = all(item["exists"] for item in binaries)
            complete = complete and ok
            variant_views[variant] = {
                "label": VARIANT_LABEL[variant],
                "binaries": binaries,
                "complete": ok,
            }
        if not repo["exists"]:
            state = "missing"
        elif not repo["remote_ok"]:
            state = "invalid"
        elif complete:
            state = "built"
        else:
            state = "source"
        notes = []
        if repo["reason"]:
            notes.append(repo["reason"])
        if repo["dirty"]:
            notes.append("源码目录有未提交的修改，更新会被拒绝")
        if repo["exists"] and repo["remote_ok"] and not variants:
            notes.append("未配置可用的 CUDA / ROCm 工具链，无法编译")
        deps = []
        for dep in spec["needs"]:
            dep_repo = self._repo_state(dep)
            deps.append(
                {
                    "engine": dep,
                    "label": ENGINE_CATALOG[dep]["label"],
                    "ok": dep_repo["source_ok"],
                    "dir": dep_repo["dir"],
                }
            )
        return {
            "engine": engine_id,
            "label": spec["label"],
            "source": spec["repo"],
            "dir": repo["dir"],
            "state": state,
            "repo": repo,
            "deps": deps,
            "variants": variant_views,
            "notes": notes,
            "dual": len(variants) > 1,
        }

    def view(self) -> dict[str, Any]:
        report = self.toolchains()
        return {
            **report,
            "engines": [self.engine_status(engine_id) for engine_id in ENGINE_IDS],
            "dirs": {
                "engine": str(self.dir_engine),
                "bin": str(self.dir_bin),
                "config": str(self.dir_config),
                "log": str(self.dir_log),
            },
            "jobs": self.jobs_snapshot(),
        }

    # -- jobs --------------------------------------------------------------
    def jobs_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_job(job) for job in self._jobs.values()]

    @staticmethod
    def _public_job(job: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if not key.startswith("_")
        }

    def job_view(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public_job(job) if job else None

    def start_job(
        self,
        action: str,
        engine_id: str | None,
        use_proxy: bool,
        *,
        force: bool = False,
        kind: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        if action not in ("install", "update", "rebuild", "toolchain"):
            raise SetupError("未知的安装动作")
        if action == "toolchain":
            if kind not in ("cuda", "rocm"):
                raise SetupError("工具链类型必须是 cuda 或 rocm")
            if not version:
                raise SetupError("缺少工具链版本")
        else:
            if engine_id not in ENGINE_CATALOG:
                raise SetupError("未知的引擎")
        job_id = uuid.uuid4().hex[:12]
        log_path = self.dir_log / "setup" / f"{job_id}.log"
        steps = self._initial_steps(action, engine_id, kind)
        job: dict[str, Any] = {
            "id": job_id,
            "action": action,
            "engine": engine_id,
            "kind": kind,
            "version": version,
            "use_proxy": bool(use_proxy),
            "force": bool(force),
            "status": "running",
            "error": "",
            "detail": "",
            "percent": 0,
            "bytes": 0,
            "total_bytes": 0,
            "steps": steps,
            "log_path": str(log_path),
            "started": time.time(),
            "finished": None,
            "_cancel": threading.Event(),
            "_proc": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._job_logs[job_id] = log_path
        thread = threading.Thread(
            target=self._job_worker, args=(job,), name=f"setup-{job_id}", daemon=True
        )
        thread.start()
        return self._public_job(job)

    def _initial_steps(self, action: str, engine_id: str | None, kind: str | None) -> list[dict[str, Any]]:
        if action == "toolchain":
            labels = [("download", "下载"), ("install", "安装"), ("verify", "校验")]
        elif action == "install":
            labels = [("source", "下载源码")]
            if engine_id and ENGINE_CATALOG[engine_id]["submodules"]:
                labels.append(("submodules", "下载子模块"))
            labels.append(("deps", "Python 依赖"))
            labels.extend((f"build:{variant}", f"编译 {VARIANT_LABEL[variant]}") for variant in self.active_variants())
            labels.append(("binaries", "安装可执行文件"))
            labels.append(("config", "生成配置"))
        else:
            labels = [("check", "检查更新"), ("pull", "拉取源码")]
            if engine_id and ENGINE_CATALOG[engine_id]["submodules"]:
                labels.append(("submodules", "更新子模块"))
            labels.append(("backup", "备份可执行文件"))
            labels.extend((f"build:{variant}", f"编译 {VARIANT_LABEL[variant]}") for variant in self.active_variants())
            labels.append(("finish", "完成"))
        return [
            {"key": key, "label": label, "status": "pending", "detail": ""}
            for key, label in labels
        ]

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job["_cancel"].set()
            proc = job.get("_proc")
        if proc is not None and proc.poll() is None:
            self._terminate(proc)
        return self.job_view(job_id)

    @staticmethod
    def _terminate(proc: subprocess.Popen[Any]) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                proc.terminate()
            except OSError:
                return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    proc.kill()
                except OSError:
                    pass

    def _step(self, job: dict[str, Any], key: str, status: str, detail: str = "") -> None:
        for step in job["steps"]:
            if step["key"] == key:
                step["status"] = status
                if detail:
                    step["detail"] = detail
                break
        if status == "running":
            job["detail"] = next((s["label"] for s in job["steps"] if s["key"] == key), "")
        self.log.info("setup %s %s: %s %s", job["id"], key, status, detail)

    def _job_worker(self, job: dict[str, Any]) -> None:
        log_path = self._job_logs[job["id"]]
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if job["action"] == "toolchain":
                self._run_toolchain_job(job)
            elif job["action"] == "install":
                self._run_install(job)
            else:
                self._run_update(job)
            if job["_cancel"].is_set():
                job["status"] = "cancelled"
                job["detail"] = "已取消"
            else:
                job["status"] = "ok"
                job["percent"] = 100
                job["detail"] = job["detail"] or "完成"
        except SetupError as exc:
            if job["_cancel"].is_set():
                job["status"] = "cancelled"
                job["detail"] = "已取消"
            else:
                job["status"] = "error"
                job["error"] = str(exc)
                job["detail"] = str(exc)
            for step in job["steps"]:
                if step["status"] == "running":
                    step["status"] = "error"
                    step["detail"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            self.log.exception("setup job %s failed", job["id"])
            job["status"] = "error"
            job["error"] = f"{exc}"
            job["detail"] = f"{exc}"
            for step in job["steps"]:
                if step["status"] == "running":
                    step["status"] = "error"
                    step["detail"] = f"{exc}"
        finally:
            job["finished"] = time.time()
            job["_proc"] = None

    # -- process helpers ---------------------------------------------------
    def _env(self, variant: str | None, use_proxy: bool) -> dict[str, str]:
        env = os.environ.copy()
        apply_proxy(env, self.proxy_cfg(), use_proxy)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        if variant:
            report = self.active_variants().get(variant) or {}
            root = Path(str(report.get("resolved_path") or report.get("path") or ""))
            if root.is_dir():
                env["PATH"] = f"{root / 'bin'}:{env.get('PATH', '')}"
                libs = [str(root / "lib64"), str(root / "lib")]
                env["LD_LIBRARY_PATH"] = ":".join(
                    [item for item in libs if Path(item).is_dir()]
                    + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
                )
                if variant == "cuda":
                    env["CUDA_HOME"] = str(root)
                    env["CUDA_PATH"] = str(root)
                    env["CUDAToolkit_ROOT"] = str(root)
                else:
                    env["ROCM_PATH"] = str(root)
                    env["HIP_PATH"] = str(root)
        return env

    def _run(
        self,
        job: dict[str, Any],
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        step: str | None = None,
    ) -> None:
        if job["_cancel"].is_set():
            raise SetupError("已取消")
        log_path = self._job_logs[job["id"]]
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            job["_proc"] = proc
            while proc.poll() is None:
                if job["_cancel"].is_set():
                    self._terminate(proc)
                    raise SetupError("已取消")
                time.sleep(0.2)
            code = proc.returncode
            job["_proc"] = None
        if code != 0:
            where = f"（{step}）" if step else ""
            raise SetupError(
                f"命令失败{where}：{' '.join(cmd[:4])}… 退出码 {code}\n{tail(log_path, 12)}"
            )

    # -- downloads ---------------------------------------------------------
    def _download(
        self,
        job: dict[str, Any],
        url: str,
        target: Path,
        expected_bytes: int = 0,
    ) -> Path:
        if job["_cancel"].is_set():
            raise SetupError("已取消")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        if target.is_file() and target.stat().st_size > 0 and (
            not expected_bytes or target.stat().st_size >= expected_bytes * 0.98
        ):
            job["detail"] = f"已存在，跳过下载：{target.name}"
            job["percent"] = 15
            return target
        headers = {"User-Agent": "llm-router-setup/1.0"}
        proxies = requests_proxies(self.proxy_cfg(), job["use_proxy"])
        try:
            with requests.get(
                url, stream=True, timeout=(15, 120), headers=headers, proxies=proxies,
            ) as response:
                if response.status_code != 200:
                    raise SetupError(f"下载失败：HTTP {response.status_code} {url}")
                total = int(response.headers.get("Content-Length") or expected_bytes or 0)
                job["total_bytes"] = total
                done = 0
                last = time.time()
                speed = 0.0
                with part.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if job["_cancel"].is_set():
                            raise SetupError("已取消")
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        job["bytes"] = done
                        if total:
                            job["percent"] = min(99, int(done * 100 / total))
                        now = time.time()
                        if now - last >= 1.0:
                            speed = done / max(now - job["started"], 1)
                            job["detail"] = (
                                f"{done / 1e9:.2f} / {total / 1e9:.2f} GB，"
                                f"{speed / 1e6:.1f} MB/s"
                            )
                            last = now
        except requests.RequestException as exc:
            raise SetupError(f"下载失败：{exc}") from exc
        os.replace(part, target)
        return target

    # -- toolchain jobs ----------------------------------------------------
    def _run_toolchain_job(self, job: dict[str, Any]) -> None:
        kind = job["kind"]
        version = str(job["version"])
        catalog = list(CUDA_CATALOG) if kind == "cuda" else self.toolchains()["rocm_catalog"]
        entry = next((item for item in catalog if item["version"] == version), None)
        if entry is None:
            raise SetupError(f"未知的 {kind.upper()} 版本：{version}")
        target = self.dir_toolchains / f"{kind}-{version}"
        self._step(job, "download", "running")
        archive = self._download(job, entry["url"], self.dir_tmp / entry["filename"], int(entry.get("bytes") or 0))
        self._step(job, "download", "ok", entry["filename"])
        self._check_disk(target.parent, max(int(entry.get("bytes") or 0) * 3, 8_000_000_000))
        self._step(job, "install", "running")
        if kind == "cuda":
            self._install_cuda(job, archive, target)
        else:
            self._install_rocm(job, archive, target)
        self._step(job, "install", "ok", str(target))
        self._step(job, "verify", "running")
        report = inspect_cuda(str(target)) if kind == "cuda" else inspect_rocm(str(target))
        if not report["ok"]:
            raise SetupError(f"安装后校验失败：{report['reason']}")
        # The managed toolchains/ directory is auto-detected on the next scan,
        # so no path needs to be persisted.
        self._report_cache = None
        self._step(job, "verify", "ok", f"{kind.upper()} {report['version']}")

    def _check_disk(self, path: Path, needed: int) -> None:
        try:
            free = shutil.disk_usage(str(path)).free
        except OSError:
            return
        if free < needed:
            raise SetupError(
                f"磁盘空间不足：{path} 只剩 {free / 1e9:.1f} GB，安装至少需要 {needed / 1e9:.1f} GB"
            )

    def _install_cuda(self, job: dict[str, Any], archive: Path, target: Path) -> None:
        if target.is_dir() and inspect_cuda(str(target))["ok"]:
            job["detail"] = "工具链已安装，跳过"
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        env = self._env(None, job["use_proxy"])
        env["TMPDIR"] = str(self.dir_tmp / "tmp")
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        self._run(
            job,
            [
                "sh", str(archive), "--silent", "--toolkit", "--toolkitpath", str(target),
                "--no-opengl-libs", "--no-man-page", "--no-drm", "--override",
                "--tmpdir", env["TMPDIR"],
            ],
            cwd=archive.parent, env=env, step="CUDA runfile",
        )

    def _install_rocm(self, job: dict[str, Any], archive: Path, target: Path) -> None:
        if target.is_dir() and inspect_rocm(str(target))["ok"]:
            job["detail"] = "工具链已安装，跳过"
            return
        staging = self.dir_tmp / f"rocm-{job['id']}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        env = self._env(None, job["use_proxy"])
        self._run(job, ["tar", "-xzf", str(archive), "-C", str(staging)], env=env, step="解压 ROCm")
        entries = [entry for entry in staging.iterdir()]
        source = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        if staging.exists() and source != staging:
            shutil.rmtree(staging, ignore_errors=True)

    # -- engine source -----------------------------------------------------
    def _update_submodules(self, job: dict[str, Any], engine_id: str, root: Path, env: dict[str, str]) -> None:
        if not ENGINE_CATALOG[engine_id]["submodules"]:
            return
        self._step(job, "submodules", "running")
        # KVMem's backends carry an applied patch series; a forced update
        # returns them to the pinned upstream commits before the next prepare.
        force = ["--force"] if engine_id == "kvmem" else []
        result = self._git(root, "submodule", "update", "--init", "--recursive", *force, "--depth", "1", env=env, timeout=3600)
        if result.returncode != 0:
            result = self._git(root, "submodule", "update", "--init", "--recursive", *force, env=env, timeout=3600)
        if result.returncode != 0:
            raise SetupError(f"子模块下载失败：{(result.stderr or result.stdout)[-600:]}")
        self._step(job, "submodules", "ok")

    def _source_state(self, engine_id: str) -> dict[str, Any]:
        return self._repo_state(engine_id)

    def _ensure_source(self, job: dict[str, Any], engine_id: str) -> Path:
        spec = ENGINE_CATALOG[engine_id]
        root = self._engine_dir(engine_id)
        self._step(job, "source", "running")
        if root.exists() and not (root / ".git").exists():
            if not job.get("force"):
                raise SetupError(f"{root} 已存在但不是 git 仓库；请勾选重新下载或手动删除")
            shutil.rmtree(root, ignore_errors=True)
        if root.exists():
            state = self._repo_state(engine_id)
            if not state["remote_ok"]:
                if not job.get("force"):
                    raise SetupError(state["reason"] or "源码 remote 不匹配，请重新下载引擎")
                shutil.rmtree(root, ignore_errors=True)
            else:
                self._step(job, "source", "ok", "源码已存在")
                self._update_submodules(job, engine_id, root, self._env(None, job["use_proxy"]))
                return root
        env = self._env(None, job["use_proxy"])
        self._run(
            job,
            ["git", "clone", "--depth", "1", spec["repo"], str(root)],
            cwd=self.dir_engine, env=env, step="git clone",
        )
        self._step(job, "source", "ok", spec["repo"])
        self._update_submodules(job, engine_id, root, env)
        return root

    def _requirements(self, engine_id: str, root: Path) -> tuple[Path, Path] | None:
        if engine_id == "strata":
            return root, root / "requirements.txt"
        if engine_id == "kvmem":
            return root / "backends" / "polystrata", root / "backends" / "polystrata" / "requirements.txt"
        return None

    def _python_deps(self, job: dict[str, Any], engine_id: str, root: Path) -> None:
        self._step(job, "deps", "running")
        target = self._requirements(engine_id, root)
        if target is None:
            self._step(job, "deps", "ok", "不需要 Python 依赖")
            return
        cwd, requirements = target
        if not requirements.is_file():
            self._step(job, "deps", "warn", f"未找到 {requirements}，跳过")
            return
        venv = root / ".venv"
        python = venv / "bin" / "python"
        env = self._env(None, job["use_proxy"])
        if not python.is_file():
            self._run(job, [sys.executable, "-m", "venv", str(venv)], env=env, step="venv")
        self._run(job, [str(python), "-m", "pip", "install", "--upgrade", "pip"], env=env, step="pip")
        self._run(job, [str(python), "-m", "pip", "install", "-r", str(requirements)], cwd=cwd, env=env, step="pip")
        self._step(job, "deps", "ok", str(requirements))

    # -- builds ------------------------------------------------------------
    def _cmake_generator(self) -> list[str]:
        return ["-G", "Ninja"] if shutil.which("ninja") else []

    @staticmethod
    def _hip_compiler(tool_root: Path) -> list[str]:
        """Point CMAKE_HIP_COMPILER at the ROCm clang, not the hipcc wrapper.

        CMake 4 rejects hipcc outright ("CMAKE_HIP_COMPILER is set to the
        hipcc wrapper ... use Clang directly"), and clang works on 3.x too.
        """
        for candidate in (
            tool_root / "lib" / "llvm" / "bin" / "clang++",
            tool_root / "llvm" / "bin" / "clang++",
            tool_root / "bin" / "amdclang++",
        ):
            if candidate.is_file():
                return [f"-DCMAKE_HIP_COMPILER={candidate}"]
        return []

    def _build_variant(self, job: dict[str, Any], engine_id: str, root: Path, variant: str) -> None:
        step = f"build:{variant}"
        self._step(job, step, "running")
        report = self.active_variants()[variant]
        tool_root = Path(str(report.get("resolved_path") or report.get("path")))
        env = self._env(variant, job["use_proxy"])
        cmake = shutil.which("cmake")
        if not cmake:
            raise SetupError("找不到 cmake，请先安装 cmake（>=3.20）与 ninja")
        # Compilation needs the machine's RAM (and the card for probes); stop
        # a running engine first, then size the job pool to what is free.
        self._stop_engine_before_build(job)
        jobs = self._build_jobs(engine_id, job)
        if engine_id == "llama":
            build_dir = root / ("build-cuda" if variant == "cuda" else "build-hip")
            configure = [
                cmake, "-S", str(root), "-B", str(build_dir), *self._cmake_generator(),
                "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF",
                "-DGGML_NATIVE=ON",
            ]
            if variant == "cuda":
                configure += [
                    "-DGGML_CUDA=ON", "-DGGML_HIP=OFF",
                    f"-DCMAKE_CUDA_COMPILER={tool_root / 'bin' / 'nvcc'}",
                    "-DCMAKE_CUDA_ARCHITECTURES=native",
                ]
            else:
                configure += [
                    "-DGGML_HIP=ON", "-DGGML_CUDA=OFF",
                    *self._hip_compiler(tool_root),
                    f"-DCMAKE_HIP_ARCHITECTURES={self._amd_arch(detect_gpus())}",
                ]
            self._run(job, configure, env=env, step=step)
            self._run(
                job,
                [cmake, "--build", str(build_dir), "--config", "Release", "--parallel", jobs,
                 "--target", "llama-server", "llama-mtmd-cli"],
                env=env, step=step,
            )
        elif engine_id == "strata":
            llama_root = self._engine_dir("llama")
            if not (llama_root / "CMakeLists.txt").is_file():
                raise SetupError("Strata 需要 llama.cpp 源码，请先安装 llama.cpp")
            build_dir = root / ("build-cuda" if variant == "cuda" else "build-hip")
            configure = [
                cmake, "-S", str(root), "-B", str(build_dir), *self._cmake_generator(),
                "-DCMAKE_BUILD_TYPE=Release",
                # Strata pins the ggml commit it is verified against (latest llama.cpp
                # changed the MMQ template API); let it fetch that pinned checkout.
                "-DSTRATA_GGML_DIR=",
            ]
            if variant == "cuda":
                configure += [
                    "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_ENABLE_HIP=OFF",
                    # HIP-only option; keep the cache explicit for reused build dirs.
                    "-DSTRATA_PREFILL_MMQ=OFF",
                    f"-DCMAKE_CUDA_COMPILER={tool_root / 'bin' / 'nvcc'}",
                    "-DCMAKE_CUDA_ARCHITECTURES=native",
                ]
            else:
                configure += [
                    "-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF",
                    # HIP-only option; CUDA configure rejects it.
                    "-DSTRATA_PREFILL_MMQ=ON",
                    *self._hip_compiler(tool_root),
                    f"-DCMAKE_HIP_ARCHITECTURES={self._amd_arch(detect_gpus())}",
                ]
            self._run(job, configure, env=env, step=step)
            self._run(job, [cmake, "--build", str(build_dir), "--config", "Release", "--parallel", jobs, "--target", "strata"], env=env, step=step)
            vision_dir = root / ("build-vision-cuda" if variant == "cuda" else "build-vision-hip")
            # The vision tool must use the same llama.cpp/ggml checkout as the
            # engine build (the pinned one Strata fetched), not a newer one.
            pinned = build_dir / "_deps" / "strata_llamacpp-src"
            vision_llama = pinned if (pinned / "CMakeLists.txt").is_file() else llama_root
            vision = [
                cmake, "-S", str(root / "tools" / "vision"), "-B", str(vision_dir), *self._cmake_generator(),
                "-DCMAKE_BUILD_TYPE=Release", f"-DLLAMA_DIR={vision_llama}",
            ]
            if variant == "cuda":
                vision += ["-DSTRATA_VISION_CUDA=ON", "-DSTRATA_VISION_HIP=OFF",
                           f"-DCMAKE_CUDA_COMPILER={tool_root / 'bin' / 'nvcc'}",
                           "-DCMAKE_CUDA_ARCHITECTURES=native"]
            else:
                vision += ["-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_VISION_HIP=ON",
                           *self._hip_compiler(tool_root),
                           f"-DSTRATA_VISION_HIP_ARCHS={self._amd_arch(detect_gpus())}"]
            self._run(job, vision, env=env, step=step)
            self._run(job, [cmake, "--build", str(vision_dir), "--config", "Release", "--parallel", jobs, "--target", "strata-vision"], env=env, step=step)
        else:
            self._build_kvmem(job, root, variant, tool_root, env, step, jobs)
        self._step(job, step, "ok")

    def _build_kvmem(
        self,
        job: dict[str, Any],
        root: Path,
        variant: str,
        tool_root: Path,
        env: dict[str, str],
        step: str,
        jobs: str,
    ) -> None:
        backends = root / "backends"
        if not (backends / "polystrata" / "CMakeLists.txt").is_file():
            raise SetupError("kvmem 子模块缺失，请重新下载 kvmem（含子模块）")
        cmake = shutil.which("cmake")
        python = root / ".venv" / "bin" / "python"
        if not python.is_file():
            python = Path(sys.executable)
        # KVMem is a patch series over pinned upstream backends.  The llama.cpp
        # script prepares its own backend; PolyStrata and NInfer must be
        # prepared here or the build produces plain upstream workers that
        # reject the KVMem flags written into the model configs.
        prepare = root / "scripts" / "prepare-backends.py"
        if not prepare.is_file():
            raise SetupError("kvmem 缺少 scripts/prepare-backends.py")
        # A canceled or failed prepare can leave the patch-added files behind
        # as untracked entries.  Recorded patch states never match such a
        # tree, so drop them (ignored build directories stay).
        for name in ("llamacpp", "polystrata", "ninfer"):
            sub = backends / name
            if (sub / ".git").exists():
                self._git(sub, "reset", "--hard", env=env, timeout=300)
                self._git(sub, "clean", "-fd", env=env, timeout=600)
        for backend in ("polystrata", "ninfer") if variant == "cuda" else ("polystrata",):
            self._run(job, [str(python), str(prepare), "--backend", backend],
                      cwd=root, env=env, step=step)
        # PolyStrata runtime
        poly_dir = backends / "polystrata" / ("build-cuda" if variant == "cuda" else "build-hip")
        poly = [
            cmake, "-S", str(backends / "polystrata"), "-B", str(poly_dir), *self._cmake_generator(),
            "-DCMAKE_BUILD_TYPE=Release",
            # PolyStrata's CUDA kernels match the ggml commit it pins, not the
            # KVMem llama.cpp submodule; let it fetch that pinned checkout.
            "-DSTRATA_GGML_DIR=",
            # KVMem's patches only compile the adapter when this points at the
            # portable core; without it the worker is plain PolyStrata and
            # rejects every --kvmem-* option.
            f"-DSTRATA_KVMEM_SOURCE_DIR={root / 'kvmem'}",
        ]
        if variant == "cuda":
            poly += ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_ENABLE_HIP=OFF",
                     # HIP-only option; keep the cache explicit for reused build dirs.
                     "-DSTRATA_PREFILL_MMQ=OFF",
                     f"-DCMAKE_CUDA_COMPILER={tool_root / 'bin' / 'nvcc'}",
                     "-DCMAKE_CUDA_ARCHITECTURES=native"]
        else:
            poly += ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF",
                     # HIP-only option; CUDA configure rejects it.
                     "-DSTRATA_PREFILL_MMQ=ON",
                     *self._hip_compiler(tool_root),
                     f"-DCMAKE_HIP_ARCHITECTURES={self._amd_arch(detect_gpus())}"]
        self._run(job, poly, env=env, step=step)
        self._run(job, [cmake, "--build", str(poly_dir), "--config", "Release", "--parallel", jobs, "--target", "strata-poly"], env=env, step=step)
        if variant == "cuda":
            ninfer_dir = backends / "ninfer" / "build-cuda"
            self._ensure_ninfer_tools(job, root, env, step)
            media_dir = backends / "ninfer" / "third_party" / "media-sources"
            if not list(media_dir.glob("ffmpeg-*.tar.xz")):
                fetch = backends / "ninfer" / "scripts" / "fetch-media-sources.sh"
                if not fetch.is_file():
                    raise SetupError("ninfer 缺少 scripts/fetch-media-sources.sh")
                self._run(job, ["bash", str(fetch)], cwd=backends / "ninfer", env=env, step=step)
            ninfer = [
                cmake, "-S", str(backends / "ninfer"), "-B", str(ninfer_dir), *self._cmake_generator(),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DNINFER_KVMEM_SOURCE_DIR={root / 'kvmem'}",
                f"-DCMAKE_CUDA_COMPILER={tool_root / 'bin' / 'nvcc'}",
                f"-DCMAKE_CUDA_ARCHITECTURES={nvidia_arch_for_ninfer(self._nvidia_cap())}",
            ]
            self._run(job, ninfer, env=env, step=step)
            self._run(job, [cmake, "--build", str(ninfer_dir), "--config", "Release", "--parallel", jobs, "--target", "ninfer-serve"], env=env, step=step)
        # llama-kvmem-server (the KVMem llama.cpp integration)
        build_dir = root / ("build-cuda" if variant == "cuda" else "build-rocm")
        if variant == "cuda":
            script = root / "scripts" / "build-cuda.sh"
            if not script.is_file():
                raise SetupError("kvmem 缺少 scripts/build-cuda.sh")
            env = dict(env)
            env.setdefault("CUDA_HOME", str(tool_root))
            # KVMem's own CMakeLists inherits this from the environment as an
            # UNINITIALIZED cache entry; CMake then refuses to resolve "native"
            # at generate time ("no NVIDIA GPU was detected").  Pass the real
            # architecture instead of relying on native detection.
            env["CMAKE_CUDA_ARCHITECTURES"] = f"{nvidia_arch_for_ninfer(self._nvidia_cap())}-real"
            env.setdefault("BUILD_DIR", str(build_dir))
            env.setdefault("NPROC", jobs)
            self._run(job, ["bash", str(script)], cwd=root, env=env, step=step)
        else:
            script = root / "scripts" / "build-rocm.py"
            if not script.is_file():
                raise SetupError("kvmem 缺少 scripts/build-rocm.py")
            self._run(
                job,
                [str(python), str(script), "--linux", "--rocm", str(tool_root),
                 "--gpu-targets", self._amd_arch(detect_gpus()), "--build-dir", str(build_dir),
                 "--jobs", jobs],
                cwd=root, env=env, step=step,
            )

    def _ensure_ninfer_tools(
        self, job: dict[str, Any], root: Path, env: dict[str, str], step: str
    ) -> None:
        """NInfer links FFmpeg and dav1d statically: meson + nasm are required."""
        venv_bin = root / ".venv" / "bin"
        python = venv_bin / "python"
        if python.is_file():
            env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
            if not (venv_bin / "meson").is_file():
                self._run(job, [str(python), "-m", "pip", "install", "meson"], env=env, step=step)
        nasm_root = self.dir_toolchains / "nasm"
        nasm_bin = nasm_root / "bin" / "nasm"
        if not nasm_bin.is_file():
            archive = self.dir_tmp / f"nasm-{NASM_VERSION}.tar.xz"
            self._download(job, NASM_URL, archive)
            source = self.dir_toolchains / f"nasm-{NASM_VERSION}"
            if source.exists():
                shutil.rmtree(source)
            try:
                with tarfile.open(archive, "r:xz") as tar:
                    tar.extractall(self.dir_toolchains, filter="data")
            except TypeError:  # Python < 3.12 has no extraction filters
                with tarfile.open(archive, "r:xz") as tar:
                    tar.extractall(self.dir_toolchains)
            self._run(job, ["./configure", f"--prefix={nasm_root}"], cwd=source, env=env, step=step)
            self._run(job, ["make", f"-j{max(2, min(os.cpu_count() or 4, 16))}"], cwd=source, env=env, step=step)
            self._run(job, ["make", "install"], cwd=source, env=env, step=step)
        env["PATH"] = str(nasm_root / "bin") + os.pathsep + env.get("PATH", "")

    def _build_jobs(self, engine_id: str, job: dict[str, Any] | None = None) -> str:
        """Parallel compile jobs sized to the RAM this machine can spare.

        Heavy CUDA/HIP translation units peak at ~3-4 GB each.  Keep a reserve
        for the desktop and refuse to start when not even one job fits, so a
        busy or small host fails fast instead of being OOM-killed.
        """
        cpu = os.cpu_count() or 4
        cap = 12 if engine_id == "kvmem" else 16
        per_job_mb = 4096 if engine_id == "kvmem" else 3072
        total_mb, avail_mb = self._meminfo_mb()
        if avail_mb <= 0:
            return str(max(1, min(cpu, cap)))
        reserve_mb = max(2048, int(total_mb * 0.10))
        budget = avail_mb - reserve_mb
        count = int(min(cpu, cap, budget // per_job_mb))
        if count < 1:
            hint = (
                "（引擎已停止，释放后仍不足）"
                if job is not None and job.get("_engine_was_running")
                else "（引擎未运行，无法再释放资源）"
            )
            raise SetupError(
                f"可用内存不足，已中止编译{hint}：当前可用 {avail_mb / 1024:.1f} GB，"
                f"每个编译任务约需 {per_job_mb / 1024:.1f} GB，"
                f"还要为系统保留 {reserve_mb / 1024:.1f} GB"
            )
        if job is not None:
            job["detail"] = (
                f"{count} 个并行编译任务（可用内存 {avail_mb / 1024:.1f} GB，"
                f"每任务约 {per_job_mb / 1024:.1f} GB）"
            )
        return str(count)

    @staticmethod
    def _meminfo_mb() -> tuple[int, int]:
        total = avail = 0
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                if total and avail:
                    break
        except (OSError, ValueError):
            pass
        return total, avail

    def _stop_engine_before_build(self, job: dict[str, Any]) -> None:
        """A running engine holds the card and tens of GB of RAM: stop it so
        the compiler gets the machine, then measure what is actually free."""
        if job.get("_engine_checked"):
            return
        job["_engine_checked"] = True
        stopped = False
        if self.stop_engine is not None:
            job["detail"] = "编译前检查运行中的引擎"
            try:
                stopped = bool(self.stop_engine())
            except Exception as exc:
                self.log.exception(
                    "cannot stop the engine before building %s", job.get("engine")
                )
                raise SetupError(f"编译前无法停止正在运行的引擎：{exc}") from exc
        job["_engine_was_running"] = stopped
        if stopped:
            self.log.info(
                "setup %s: stopped the running engine before compiling", job.get("id")
            )
            job["detail"] = "已停止正在运行的引擎，正在计算编译并行度"

    def _nvidia_cap(self) -> str:
        for item in detect_gpus():
            if item.get("vendor") == "nvidia" and item.get("arch"):
                return str(item["arch"])
        return "8.9"

    # -- install -----------------------------------------------------------
    def _run_install(self, job: dict[str, Any]) -> None:
        engine_id = job["engine"]
        variants = self.active_variants()
        if not variants:
            raise SetupError("请先在引擎页顶部配置可用的 CUDA 或 ROCm 路径")
        for dep in ENGINE_CATALOG[engine_id]["needs"]:
            dep_state = self._repo_state(dep)
            if not dep_state["source_ok"]:
                raise SetupError(
                    f"{ENGINE_CATALOG[engine_id]['label']} 依赖 {ENGINE_CATALOG[dep]['label']}，请先安装 {ENGINE_CATALOG[dep]['label']}"
                )
        root = self._ensure_source(job, engine_id)
        self._python_deps(job, engine_id, root)
        for variant in variants:
            self._build_variant(job, engine_id, root, variant)
        self._step(job, "binaries", "running")
        self._install_artifacts(job, engine_id, root)
        self._step(job, "binaries", "ok")
        self._step(job, "config", "running")
        self._write_configs(engine_id, root)
        self._step(job, "config", "ok")
        job["detail"] = "安装完成"

    def _locate(self, root: Path, names: tuple[str, ...], allow_glob: bool = True) -> Path | None:
        for name in names:
            direct = root / name
            if direct.is_file():
                return direct
        if allow_glob:
            for name in names:
                for found in root.rglob(name):
                    if found.is_file() and os.access(found, os.X_OK):
                        return found
        return None

    def _copy_artifact(
        self, job: dict[str, Any], engine_id: str, source: Path, target: Path, variant: str
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.copy2(source, target)
        target.chmod(0o755)
        # Runtime objects built next to the executable (libllama/libggml/...).
        # Namespace them per engine: KVMem ships its own pinned llama.cpp build
        # whose shared objects are ABI-incompatible with upstream llama.cpp, so
        # a shared bin/libs/default would let one engine load the other's libs.
        lib_dir = self.dir_bin / "libs" / engine_id / (variant if self._dual() else "default")
        lib_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.so", "*.so.*", "*.dylib"):
            for item in source.parent.glob(pattern):
                if item.is_file():
                    shutil.copy2(item, lib_dir / item.name)
        job["detail"] = f"{target.name} ← {source}"

    def _find_artifact(self, engine_id: str, root: Path, artifact: str, variant: str) -> Path | None:
        if engine_id == "llama":
            build = root / ("build-cuda" if variant == "cuda" else "build-hip")
            return self._locate(build, (f"bin/{artifact}", artifact))
        if engine_id == "strata":
            if artifact == "strata":
                build = root / ("build-cuda" if variant == "cuda" else "build-hip")
                return self._locate(build, ("strata", "bin/strata"))
            build = root / ("build-vision-cuda" if variant == "cuda" else "build-vision-hip")
            return self._locate(build, ("strata-vision", "strata-vision-hip", "bin/strata-vision", "bin/strata-vision-hip"))
        if artifact == "strata-poly":
            build = root / "backends" / "polystrata" / ("build-cuda" if variant == "cuda" else "build-hip")
            return self._locate(build, ("strata-poly", "bin/strata-poly"))
        if artifact == "ninfer-serve":
            build = root / "backends" / "ninfer" / "build-cuda"
            return self._locate(build, ("ninfer-serve", "apps/ninfer-serve", "bin/ninfer-serve"))
        build = root / ("build-cuda" if variant == "cuda" else "build-rocm")
        return self._locate(build, ("llama-kvmem-server", "bin/llama-kvmem-server"))
    def _install_artifacts(self, job: dict[str, Any], engine_id: str, root: Path) -> None:
        for variant in self.active_variants():
            spec = ENGINE_CATALOG[engine_id]
            for artifact, _target in spec["artifacts"].get(variant, ()):
                source = self._find_artifact(engine_id, root, artifact, variant)
                if source is None:
                    raise SetupError(f"编译完成但找不到 {artifact}（{VARIANT_LABEL[variant]}）")
                target = self.dir_bin / self._bin_name(engine_id, artifact, variant)
                self._copy_artifact(job, engine_id, source, target, variant)
        if engine_id == "kvmem":
            self._sync_poly_alias()

    # -- config stubs ------------------------------------------------------
    def _poly_alias(self, variant: str | None) -> Path:
        """Path the PolyStrata model configs run.

        kvmem's backends/polystrata/serve/server.py tells the PolyStrata
        worker from the Strata project's engine by an exact "strata-poly"
        file stem (poly_engine); the renamed KVMem build keeps its required
        name (bin/kvmem-strata-poly) and is exposed under that stem here.
        """
        base = self.dir_bin / "kvmem"
        if variant and self._dual():
            return base / variant / "strata-poly"
        return base / "strata-poly"

    def _sync_poly_alias(self) -> None:
        for variant in self.active_variants():
            real = self.dir_bin / self._bin_name("kvmem", "strata-poly", variant)
            if not real.is_file():
                continue
            alias = self._poly_alias(variant)
            alias.parent.mkdir(parents=True, exist_ok=True)
            if alias.is_symlink() or alias.exists():
                alias.unlink()
            alias.symlink_to(os.path.relpath(real, alias.parent))

    def _polystrata_exe(self) -> Path:
        variants = self.active_variants()
        variant = "cuda" if "cuda" in variants else "rocm" if "rocm" in variants else None
        if variant is None:
            return self.dir_bin / "kvmem-strata-poly"
        return self._poly_alias(variant)

    def _write_configs(self, engine_id: str, root: Path) -> None:
        if engine_id == "llama":
            preset = self.dir_config / "llama.cpp" / "model.ini"
            if not preset.is_file():
                preset.write_text(
                    "; llama.cpp models preset\n; 在这里为每个模型添加一段，例如：\n"
                    "; [my-model]\n; model = /path/to/model.gguf\n; c = 8192\n",
                    encoding="utf-8",
                )
        elif engine_id == "strata":
            template = self.dir_config / "strata" / "strata-template.json"
            if not template.is_file():
                bin_path = self.dir_bin / self._bin_name("strata", "strata", "cuda" if "cuda" in self.active_variants() else "rocm")
                write_json_atomic(
                    template,
                    {
                        "exe": str(bin_path),
                        "args": ["--native", "", "--max-context", "8192"],
                        "cwd": str(root),
                        "log": str(self.dir_log / "strata-engine.log"),
                        "model_name": "template",
                        "host": "127.0.0.1",
                        "port": 18084,
                    },
                )
            else:
                # A template without "log" makes serve.server discard the
                # child engine's stderr; early stubs lacked it.
                existing = read_json(template)
                if isinstance(existing, dict) and not str(existing.get("log") or "").strip():
                    existing["log"] = str(self.dir_log / "strata-engine.log")
                    write_json_atomic(template, existing)
        elif engine_id == "kvmem":
            config_dir = self.dir_config / "polystrata"
            meta = config_dir / "models.json"
            if not meta.is_file():
                write_json_atomic(meta, {})
            reference = config_dir / "strata-reference.json"
            vision = self.dir_bin / self._bin_name("strata", "strata-vision", "cuda" if "cuda" in self.active_variants() else "rocm")
            if not reference.is_file():
                write_json_atomic(
                    reference,
                    {
                        "exe": str(self._polystrata_exe()),
                        "display_name": "参考模板",
                        "aliases": [],
                        "args": [
                            "--model", "", "--max-context", "32768",
                            "--prompt-batch", "512", "--window", "1",
                            "--no-draft", "--lookup", "0",
                            "--kvmem-budget", "16384", "--kvmem-gen-reserve", "4096",
                            "--kvmem-block-tokens", "32", "--kvmem-host-mib", "4096",
                            "--kvmem-sink-tokens", "64", "--kvmem-recent-tokens", "64",
                            "--kv", "q8_0", "--vram-reserve-mib", "512",
                        ],
                        "cwd": str(root / "backends" / "polystrata"),
                        "vision": {"exe": str(vision)},
                        "log": str(self.dir_log / "polystrata-reference.log"),
                        "host": "127.0.0.1",
                        "port": 18081,
                    },
                )
            else:
                # Early builds wrote bin/kvmem-strata-poly straight into the
                # template; serve.server then fails to see the PolyStrata
                # worker and adds an unsupported --vision.  Point new models
                # at the alias and refresh a missing encoder.
                existing = read_json(reference)
                if isinstance(existing, dict):
                    changed = False
                    exe = str(self._polystrata_exe())
                    if existing.get("exe") != exe:
                        existing["exe"] = exe
                        changed = True
                    vision_cfg = existing.get("vision")
                    if isinstance(vision_cfg, dict) and not vision_cfg.get("exe"):
                        vision_cfg["exe"] = str(vision)
                        changed = True
                    if changed:
                        write_json_atomic(reference, existing)
            ninfer_template = self.dir_config / "ninfer" / "ninfer.template.json"
            if not ninfer_template.is_file():
                ninfer_exe = self.dir_bin / self._bin_name("kvmem", "ninfer-serve", "cuda")
                write_json_atomic(
                    ninfer_template,
                    {
                        "model_name": "template",
                        "model_id": "template",
                        "display_name": "NInfer 模板（新增模型时替换模型路径）",
                        "model": "/path/to/model.ninfer",
                        "args": [
                            "--model-id", "template", "--max-context", "131072",
                            "--max-concurrency", "1", "--kvmem",
                            "--kvmem-budget", "36864", "--kvmem-gen-reserve", "4096",
                            "--kv-dtype", "nvfp4",
                        ],
                        "exe": str(ninfer_exe),
                        "cwd": str(root),
                        "env": {},
                        "context_length": 131072,
                        "default_max_tokens": 4096,
                        "aliases": [],
                    },
                )
            else:
                # Early builds wrote llama.cpp's --kv-dtype q8_0 into this
                # template; NInfer rejects that value and would refuse to start.
                existing = read_json(ninfer_template)
                args = existing.get("args") if isinstance(existing, dict) else None
                if isinstance(args, list):
                    fixed = False
                    for index, token in enumerate(args[:-1]):
                        if str(token) == "--kv-dtype" and str(args[index + 1]) == "q8_0":
                            args[index + 1] = "nvfp4"
                            fixed = True
                    if fixed:
                        write_json_atomic(ninfer_template, existing)
            kvllama_template = self.dir_config / "kvmem-llama" / "kvmem-llama.template.json"
            if not kvllama_template.is_file():
                write_json_atomic(
                    kvllama_template,
                    {
                        "args": [
                            "-c", "8192", "-ngl", "999", "--flash-attn", "on",
                            "--kvmem", "--kvmem-budget", "32768",
                            "--kvmem-block-tokens", "32", "--kvmem-gen-reserve", "1024",
                            "--kv-dtype", "q8_0", "--no-ui",
                        ]
                    },
                )
        elif engine_id == "ninfer":
            pass

    # -- update ------------------------------------------------------------
    def check_updates(self, use_proxy: bool = False) -> dict[str, Any]:
        env = self._env(None, use_proxy)
        results: dict[str, Any] = {}
        for engine_id in ENGINE_IDS:
            root = self._engine_dir(engine_id)
            if not (root / ".git").exists():
                results[engine_id] = {"state": "missing", "message": "源码未安装"}
                continue
            state = self._repo_state(engine_id)
            if not state["remote_ok"]:
                results[engine_id] = {"state": "invalid", "message": state["reason"]}
                continue
            fetch = self._git(root, "fetch", "--prune", "--quiet", env=env, timeout=600)
            if fetch.returncode != 0:
                results[engine_id] = {
                    "state": "error",
                    "message": f"git fetch 失败：{(fetch.stderr or fetch.stdout).strip()[:400]}",
                }
                continue
            upstream = self._git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", timeout=20)
            if upstream.returncode != 0:
                results[engine_id] = {"state": "error", "message": "没有上游分支"}
                continue
            behind = self._git(root, "rev-list", "--count", "HEAD..@{u}", timeout=30)
            commits = self._git(root, "log", "--oneline", "--max-count", "8", "HEAD..@{u}", timeout=30)
            count = int(behind.stdout.strip() or 0) if behind.returncode == 0 else 0
            results[engine_id] = {
                "state": "update" if count else "current",
                "behind": count,
                "commits": [line for line in commits.stdout.splitlines() if line.strip()] if count else [],
                "version": state["version"],
                "dirty": state["dirty"],
                "message": f"有 {count} 个新提交" if count else "已是最新版",
            }
        return results

    def _run_update(self, job: dict[str, Any]) -> None:
        engine_id = job["engine"]
        root = self._engine_dir(engine_id)
        self._step(job, "check", "running")
        state = self._repo_state(engine_id)
        if not state["exists"]:
            raise SetupError("源码未安装，请先安装引擎")
        if not state["remote_ok"]:
            raise SetupError(state["reason"])
        if state["dirty"]:
            raise SetupError("源码目录有未提交的修改，拒绝更新；请先提交或还原")
        env = self._env(None, job["use_proxy"])
        fetch = self._git(root, "fetch", "--prune", "--quiet", env=env, timeout=600)
        if fetch.returncode != 0:
            raise SetupError(f"git fetch 失败：{(fetch.stderr or fetch.stdout)[-500:]}")
        behind = self._git(root, "rev-list", "--count", "HEAD..@{u}", timeout=30)
        count = int(behind.stdout.strip() or 0) if behind.returncode == 0 else 0
        if not count:
            self._step(job, "check", "ok", "已是最新版")
            job["detail"] = "已是最新版"
            for step in job["steps"]:
                if step["status"] == "pending":
                    step["status"] = "skip"
            return
        self._step(job, "check", "ok", f"{count} 个新提交")
        old_head = self._git(root, "rev-parse", "HEAD", timeout=20).stdout.strip()
        self._step(job, "pull", "running")
        merge = self._git(root, "merge", "--ff-only", "@{u}", env=env, timeout=600)
        if merge.returncode != 0:
            raise SetupError(f"git pull 失败：{(merge.stderr or merge.stdout)[-500:]}")
        self._step(job, "pull", "ok")
        self._update_submodules(job, engine_id, root, env)
        self._step(job, "backup", "running")
        backups: list[tuple[Path, Path]] = []
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for variant in self.active_variants():
            for _target, path in self._bin_paths(engine_id, variant):
                if path.is_file():
                    backup = path.with_name(path.name + f".bak-{stamp}")
                    shutil.copy2(path, backup)
                    backups.append((path, backup))
        # Shared objects are installed in place; move the engine's directory
        # aside so a failed rebuild keeps the old library set runnable.
        libs_dir = self.dir_bin / "libs" / engine_id
        libs_backup: Path | None = None
        if libs_dir.is_dir():
            candidate = libs_dir.with_name(f"{libs_dir.name}.bak-{stamp}")
            try:
                shutil.move(str(libs_dir), str(candidate))
                libs_backup = candidate
            except OSError:
                libs_backup = None
        self._step(job, "backup", "ok", f"{len(backups)} 个文件")
        try:
            for variant in self.active_variants():
                self._build_variant(job, engine_id, root, variant)
            self._install_artifacts(job, engine_id, root)
            self._write_configs(engine_id, root)
        except Exception:
            for original, backup in backups:
                try:
                    shutil.copy2(backup, original)
                except OSError:
                    pass
            if libs_backup is not None:
                try:
                    if libs_dir.exists():
                        shutil.rmtree(libs_dir)
                    shutil.move(str(libs_backup), str(libs_dir))
                except OSError:
                    pass
            if old_head:
                self._git(root, "reset", "--hard", old_head, timeout=120)
                force = ["--force"] if engine_id == "kvmem" else []
                self._git(root, "submodule", "update", "--init", "--recursive", *force, timeout=1800)
            raise
        for _original, backup in backups:
            backup.unlink(missing_ok=True)
        if libs_backup is not None:
            shutil.rmtree(libs_backup, ignore_errors=True)
        self._step(job, "finish", "ok", "更新完成")
        job["detail"] = "更新完成"


# ---------------------------------------------------------------------------
# module self-check

if __name__ == "__main__":
    logging_stub = type("L", (), {"info": staticmethod(lambda *a, **k: None), "warning": staticmethod(lambda *a, **k: None), "exception": staticmethod(lambda *a, **k: None)})()
    setup = EngineSetup(Path.cwd(), logging_stub, {})
    print(json.dumps(setup.view(), ensure_ascii=False, indent=2))
