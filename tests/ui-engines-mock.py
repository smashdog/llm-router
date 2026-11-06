#!/usr/bin/env python3
"""Mock gateway for the engine management / add-model UI test."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
PORT = 18097

FIELDS = {
    "llama": {
        "binary": "/mnt/ai/llm/llama.cpp/build/bin/llama-server",
        "cwd": "/mnt/ai/llm/llama.cpp",
        "models_dir": "/mnt/ai/llm/llm-router/models",
        "preset": "/mnt/ai/llm/llm-router/models.ini",
    },
    "kvmem_llama": {
        "binary": "/mnt/ai/llm/kvmem-llamacpp090/build-hip/bin/llama-kvmem-server",
        "cwd": "/mnt/ai/llm/kvmem-llamacpp090",
        "models_dir": "/mnt/ai/llm/llm-router/models-kvmem-llama",
        "template": "/mnt/ai/llm/llm-router/models-kvmem-llama/kvmem-llama.template.json",
    },
    "kvmem": {
        "binary": "/mnt/ai/llm/kvmem/.venv/bin/python",
        "cwd": "/mnt/ai/llm/kvmem/backends/polystrata",
        "models_dir": "/mnt/ai/llm/kvmem/local_config",
        "meta": "/mnt/ai/llm/kvmem/local_config/models.json",
    },
    "ninfer": {
        "binary": "/mnt/ai/llm/build-ninfer/apps/ninfer-serve",
        "cwd": "/mnt/ai/llm/kvmem",
        "models_dir": "/mnt/ai/llm/kvmem/local_config",
        "template": "/mnt/ai/llm/kvmem/local_config/ninfer.template.json",
    },
}

DEFAULTS = {
    "llama": dict(FIELDS["llama"]),
    "kvmem_llama": dict(FIELDS["kvmem_llama"]),
    "kvmem": dict(FIELDS["kvmem"]),
    "ninfer": dict(FIELDS["ninfer"]),
}


def path_items() -> list[dict]:
    def item(engine, category, name, label, configured, running, restart, models, warning=None):
        return {
            "engine": engine,
            "category": category,
            "name": name,
            "label": label,
            "kind": engine,
            "configured": configured,
            "running": running,
            "needs_restart": restart,
            "fields": dict(FIELDS[engine]) if configured else {k: "" for k in FIELDS[engine]},
            "defaults": DEFAULTS[engine],
            "models": models,
            "warning": warning,
            "address": {
                "llama": "127.0.0.1:18080",
                "kvmem_llama": "127.0.0.1:18083",
                "kvmem": "127.0.0.1:18081",
                "ninfer": "127.0.0.1:18082",
            }[engine],
        }

    return [
        item("llama", "llama.cpp", "llama.cpp", "llama.cpp", True, True, False, 5),
        item("kvmem_llama", "kvmem", "llama.cpp", "KVMem + llama.cpp", True, True, True, 2),
        item("kvmem", "kvmem", "polystrata", "KVMem（polystrata）", True, False, False, 3),
        item("ninfer", "kvmem", "ninfer", "NInfer（KVMem）", False, False, False, 0),
    ]


MANAGE_ENGINES = {
    "llama": {
        "engine": "llama", "kind": "llama", "label": "llama.cpp",
        "category": "llama.cpp", "name": "llama.cpp",
        "models_dir": FIELDS["llama"]["models_dir"], "preset": FIELDS["llama"]["preset"],
        "cache_types": ["q8_0", "f16", "f32"], "defaults": {},
        "default_params": {"temperature": 0.7},
    },
    "kvmem_llama": {
        "engine": "kvmem_llama", "kind": "kvmem_llama", "label": "KVMem + llama.cpp",
        "category": "kvmem", "name": "llama.cpp",
        "models_dir": FIELDS["kvmem_llama"]["models_dir"],
        "template": FIELDS["kvmem_llama"]["template"],
        "cache_types": ["q8_0", "f16"], "defaults": {"vision_gpu": False},
        "default_params": {},
    },
    "kvmem": {
        "engine": "kvmem", "kind": "kvmem", "label": "KVMem（polystrata）",
        "category": "kvmem", "name": "polystrata",
        "models_dir": FIELDS["kvmem"]["models_dir"], "template": None,
        "defaults": {"vision_gpu": False}, "default_params": {},
    },
    "ninfer": {
        "engine": "ninfer", "kind": "ninfer", "label": "NInfer（KVMem）",
        "category": "kvmem", "name": "ninfer",
        "models_dir": FIELDS["ninfer"]["models_dir"], "template": FIELDS["ninfer"]["template"],
        "defaults": {"vision": False, "vision_gpu": False}, "default_params": {},
    },
}

STATUS = {
    "engines": {
        "llama": {"kind": "llama_router", "alive": True, "host": "127.0.0.1", "port": 18080,
                  "pid": 41001, "model": "Qwen3-8B", "started_at": "2026-11-06T10:00:00+08:00"},
        "kvmem_llama": {"kind": "kvmem_llama", "alive": True, "host": "127.0.0.1", "port": 18083,
                        "pid": 41002, "model": "Gemma-3-27B", "started_at": "2026-11-06T10:02:00+08:00"},
        "kvmem": {"kind": "kvmem", "alive": False, "host": "127.0.0.1", "port": 18081,
                  "pid": None, "model": None, "started_at": None},
        "ninfer": {"kind": "ninfer", "alive": False, "host": "127.0.0.1", "port": 18082,
                   "pid": None, "model": None, "started_at": None},
    },
    "active_engine": "llama",
    "active_model": "Qwen3-8B",
    "switching": False,
    "async_operation": None,
}


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

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            data = (ROOT / "webui" / "index.html").read_bytes()
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
        if path == "/router/engines/paths":
            return self.send_json({
                "object": "list",
                "config_file": "config.json",
                "engines": path_items(),
            })
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
                                   "key_file": "api-key", "protected": ["/v1/*", "/props"]})
        return self.send_json({"error": {"message": "not found"}}, 404)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print(f"mock listening on {PORT}", flush=True)
    server.serve_forever()
