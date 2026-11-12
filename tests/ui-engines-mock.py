#!/usr/bin/env python3
"""Mock gateway for the engine provisioning / add-model UI test.

Serves the current console against a fake /router/setup payload (toolchains,
install-product engine cards, setup jobs and proxy switches) plus the
/router/models/manage data used by the two-column add-model picker.  Engine
slot ids use the renamed kvmem_polystrata / kvmem_ninfer names; install-product
ids (llama / strata / kvmem) are unchanged.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
PORT = 18097

MANAGE_ENGINES = {
    "llama": {
        "engine": "llama", "kind": "llama", "label": "llama.cpp",
        "category": "llama.cpp", "name": "llama.cpp",
        "models_dir": "/mnt/ai/llm/llm-router/models",
        "preset": "/mnt/ai/llm/llm-router/models.ini",
        "cache_types": ["q8_0", "f16", "f32"], "defaults": {},
        "default_params": {"temperature": 0.7},
    },
    "kvmem_llama": {
        "engine": "kvmem_llama", "kind": "kvmem_llama", "label": "KVMem + llama.cpp",
        "category": "kvmem", "name": "llama.cpp",
        "models_dir": "/mnt/ai/llm/llm-router/models-kvmem-llama",
        "template": "/mnt/ai/llm/llm-router/models-kvmem-llama/kvmem-llama.template.json",
        "cache_types": ["q8_0", "f16"], "defaults": {"vision_gpu": False},
        "default_params": {},
    },
    "kvmem_polystrata": {
        "engine": "kvmem_polystrata", "kind": "kvmem_polystrata",
        "label": "KVMem + polystrata",
        "category": "kvmem", "name": "polystrata",
        "models_dir": "/mnt/ai/llm/kvmem/local_config", "template": None,
        "defaults": {"vision_gpu": False}, "default_params": {},
    },
    "kvmem_ninfer": {
        "engine": "kvmem_ninfer", "kind": "kvmem_ninfer", "label": "KVMem + ninfer",
        "category": "kvmem", "name": "ninfer",
        "models_dir": "/mnt/ai/llm/kvmem/local_config",
        "template": "/mnt/ai/llm/kvmem/local_config/ninfer.template.json",
        "defaults": {"vision": False, "vision_gpu": False}, "default_params": {},
    },
}

STATUS = {
    "engines": {
        "llama": {"kind": "llama_router", "alive": True, "host": "127.0.0.1", "port": 18080,
                  "pid": 41001, "model": "Qwen3-8B", "started_at": "2026-11-06T10:00:00+08:00"},
        "kvmem_llama": {"kind": "kvmem_llama", "alive": True, "host": "127.0.0.1", "port": 18083,
                        "pid": 41002, "model": "Gemma-3-27B", "started_at": "2026-11-06T10:02:00+08:00"},
        "kvmem_polystrata": {"kind": "kvmem_polystrata", "alive": False, "host": "127.0.0.1",
                             "port": 18081, "pid": None, "model": None, "started_at": None},
        "kvmem_ninfer": {"kind": "kvmem_ninfer", "alive": False, "host": "127.0.0.1",
                         "port": 18082, "pid": None, "model": None, "started_at": None},
    },
    "active_engine": "llama",
    "active_model": "Qwen3-8B",
    "switching": False,
    "async_operation": None,
}

SETUP_ENGINES = [
    {
        "engine": "llama", "label": "llama.cpp", "state": "built",
        "source": "github.com/ggml-org/llama.cpp",
        "dir": "/mnt/ai/llm/llm-router-new/engine/llama.cpp",
        "repo": {"version": "b7220"},
        "variants": {"rocm": {"binaries": [{"name": "llama-server", "exists": True}]}},
        "deps": [], "notes": [],
    },
    {
        "engine": "strata", "label": "Strata", "state": "built",
        "source": "github.com/ornith-ai/strata",
        "dir": "/mnt/ai/llm/llm-router-new/engine/Strata",
        "repo": {"version": "0.1.57"},
        "variants": {"rocm": {"binaries": [{"name": "strata", "exists": True}]}},
        "deps": [], "notes": [],
    },
    {
        "engine": "kvmem", "label": "KVMem", "state": "built",
        "source": "github.com/smashdog/kvmem",
        "dir": "/mnt/ai/llm/llm-router-new/engine/kvmem",
        "repo": {"version": "v0.4.6"},
        "variants": {"rocm": {"binaries": [
            {"name": "llama-kvmem-server", "exists": True},
            {"name": "strata-poly", "exists": True},
            {"name": "ninfer-serve", "exists": True},
        ]}},
        "deps": [], "notes": [],
    },
]

CONFIG = {"proxy_configured": True, "install_fails": False}
SETUP_PROXY: dict[str, bool] = {}
LAST = {"proxy": None, "install": None, "install_op": None, "toolchain": None}
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
JOB_SEQ = [0]


def job_view(job: dict) -> dict:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def setup_payload() -> dict:
    return {
        "show_cuda": False,
        "show_rocm": True,
        "cuda": {"installed": False, "ok": False, "version": None,
                 "reason": "未检测到 NVIDIA 显卡"},
        "rocm": {"installed": True, "ok": False, "version": "6.2.0",
                 "resolved_path": "/opt/rocm-6.2", "reason": "低于最低要求 6.4"},
        "cuda_catalog": [],
        "rocm_catalog": [{"version": "10.1.0"}],
        "minimum": {"cuda": "12.4", "rocm": "6.4"},
        "rocm_family": "RDNA3",
        "engines": SETUP_ENGINES,
        "dirs": {
            "engine": "/mnt/ai/llm/llm-router-new/engine",
            "bin": "/mnt/ai/llm/llm-router-new/bin",
            "config": "/mnt/ai/llm/llm-router-new/config",
            "log": "/mnt/ai/llm/llm-router-new/log",
        },
        "jobs": [job_view(job) for job in JOBS.values()],
        "proxy_configured": bool(CONFIG["proxy_configured"]),
        "setup_proxy": dict(SETUP_PROXY),
    }


def start_job(payload: dict, action: str) -> dict:
    with LOCK:
        JOB_SEQ[0] += 1
        job_id = f"job-{JOB_SEQ[0]}"
        if action == "toolchain":
            label = f"安装 {payload.get('kind')} {payload.get('version') or ''}".strip()
        else:
            label = f"编译 {payload.get('engine')}"
        job = {
            "id": job_id,
            "action": action,
            "status": "running",
            "percent": 10,
            "detail": label,
            "steps": [
                {"status": "ok", "label": "准备源码", "detail": ""},
                {"status": "running", "label": "编译", "detail": ""},
            ],
            "_polls": 0,
        }
        if action == "toolchain":
            job["kind"] = payload.get("kind")
        else:
            job["engine"] = payload.get("engine")
        JOBS[job_id] = job
        return job_view(job)


def poll_job(job_id: str) -> dict | None:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        if job["status"] == "running":
            job["_polls"] += 1
            if job["_polls"] >= 2:
                if CONFIG["install_fails"]:
                    job["status"] = "error"
                    job["error"] = ("编译失败（退出码 1）：main.cpp:1: error: broken\n"
                                    "已恢复原二进制\n已回滚源码到更新前提交")
                    job["steps"][-1]["status"] = "error"
                    job["percent"] = 100
                else:
                    job["status"] = "ok"
                    job["percent"] = 100
                    job["detail"] = "编译完成"
                    job["steps"][-1]["status"] = "ok"
            else:
                job["percent"] = 60
        return job_view(job)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self):
        split = urlsplit(self.path)
        path = split.path
        if path in ("/", "/index.html"):
            data = (ROOT / "console" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/router/status":
            return self.send_json(STATUS)
        if path == "/models":
            return self.send_json({"object": "list", "data": []})
        if path == "/router/models/manage":
            return self.send_json({
                "engines": MANAGE_ENGINES,
                "param_schema": {},
                "kvmem_engine_schema": [],
                "kvmem_poly_engine_schema": [],
                "ninfer_engine_schema": [],
                "smoke_image": None,
                "engine_defaults": {"kv": {}, "vision_gpu": False},
                "jobs": [],
            })
        if path == "/router/auth":
            return self.send_json({"configured": False, "source": None, "hint": None,
                                   "key_file": "api-key", "protected": ["/v1/*"]})
        if path == "/router/setup":
            return self.send_json(setup_payload())
        if path == "/router/setup/job":
            job_id = (parse_qs(split.query).get("id") or [""])[0]
            job = poll_job(job_id)
            if job is None:
                return self.send_json({"error": {"message": "setup job not found"}}, 404)
            return self.send_json(job)
        if path == "/mock/state":
            with LOCK:
                return self.send_json({
                    "config": CONFIG,
                    "last": LAST,
                    "setup_proxy": SETUP_PROXY,
                    "jobs": {key: job_view(job) for key, job in JOBS.items()},
                })
        return self.send_json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        path = urlsplit(self.path).path
        payload = self.read_json()
        if path == "/mock/config":
            with LOCK:
                if "proxy_configured" in payload:
                    CONFIG["proxy_configured"] = bool(payload["proxy_configured"])
                if "install_fails" in payload:
                    CONFIG["install_fails"] = bool(payload["install_fails"])
                if payload.get("reset"):
                    JOBS.clear()
                    LAST.update({"proxy": None, "install": None,
                                 "install_op": None, "toolchain": None})
            return self.send_json({"status": "ok", "config": CONFIG})
        if path == "/router/setup/proxy":
            key = payload.get("key")
            if not isinstance(key, str) or not (
                key.startswith("engine:") or key in ("toolchain:cuda", "toolchain:rocm")
            ):
                return self.send_json({"error": {"message": "bad key"}}, 400)
            with LOCK:
                SETUP_PROXY[key] = payload.get("enabled") is True
                LAST["proxy"] = {"key": key, "enabled": SETUP_PROXY[key]}
                return self.send_json({"setup_proxy": dict(SETUP_PROXY)})
        if path in ("/router/setup/engine/install", "/router/setup/engine/update"):
            op = "update" if path.endswith("/update") else "install"
            with LOCK:
                LAST["install"] = payload
                LAST["install_op"] = op
            return self.send_json(start_job(payload, "engine"), 202)
        if path == "/router/setup/toolchain/install":
            with LOCK:
                LAST["toolchain"] = payload
            return self.send_json(start_job(payload, "toolchain"), 202)
        if path == "/router/setup/job/cancel":
            with LOCK:
                job = JOBS.get(str(payload.get("id")))
                if job is None:
                    return self.send_json({"error": {"message": "setup job not found"}}, 404)
                if job["status"] == "running":
                    job["status"] = "cancelled"
                return self.send_json(job_view(job))
        return self.send_json({"error": {"message": "not found"}}, 404)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print(f"mock listening on {PORT}", flush=True)
    server.serve_forever()
