#!/usr/bin/env python3
"""Mock gateway for the proxy / download management page UI test."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent
PORT = 18098

STATE = {
    "proxy": {
        "enabled": False, "type": "http", "host": "", "port": 0,
        "username": "", "password": "",
    },
    "dir": "",
    "sources": {
        "huggingface": False, "hf-mirror": False, "modelscope": False,
    },
    "job": None,
    "test_calls": [],
    "settings_posts": [],
    "search_q": None,
    "search_sources": None,
    "start": None,
    "cancel": False,
}

SOURCES = [
    ("huggingface", "Hugging Face", "hf"),
    ("hf-mirror", "HF-Mirror", "hf"),
    ("modelscope", "ModelScope", "modelscope"),
]

ITEMS = [
    {
        "source": "huggingface", "source_label": "Hugging Face",
        "repo_id": "Qwen/Qwen3-8B-GGUF", "pipeline_tag": "text-generation",
        "library": "gguf", "downloads": 1284000, "likes": 312,
        "license": "apache-2.0", "params": 8200000000,
        "updated_at": "2026-10-02T08:00:00.000Z", "tags": ["gguf", "qwen3", "text-generation"],
    },
    {
        "source": "huggingface", "source_label": "Hugging Face",
        "repo_id": "hf-internal-testing/tiny-random-bert", "pipeline_tag": "fill-mask",
        "library": "transformers", "downloads": 98000, "likes": 12,
        "license": "apache-2.0", "params": 4400000,
        "updated_at": "2025-11-16T03:16:16.000Z", "tags": ["bert", "tiny"],
    },
    {
        "source": "hf-mirror", "source_label": "HF-Mirror",
        "repo_id": "Qwen/Qwen3-8B-GGUF", "pipeline_tag": "text-generation",
        "library": "gguf", "downloads": 764000, "likes": 190,
        "license": "apache-2.0", "params": 8200000000,
        "updated_at": "2026-10-02T08:00:00.000Z", "tags": ["gguf", "qwen3"],
    },
    {
        "source": "hf-mirror", "source_label": "HF-Mirror",
        "repo_id": "unsloth/Qwen3-8B-GGUF", "pipeline_tag": "text-generation",
        "library": "gguf", "downloads": 233000, "likes": 87,
        "license": "apache-2.0", "params": 8200000000,
        "updated_at": "2026-09-20T10:00:00.000Z", "tags": ["gguf", "unsloth"],
    },
    {
        "source": "modelscope", "source_label": "ModelScope",
        "repo_id": "Qwen/Qwen3-8B", "pipeline_tag": None,
        "library": None, "downloads": 34210, "likes": 45,
        "license": "Apache License 2.0", "params": None,
        "updated_at": None, "tags": [], "visibility": "public",
    },
    {
        "source": "modelscope", "source_label": "ModelScope",
        "repo_id": "inclusionAI/Ling-3.0-tiny", "pipeline_tag": None,
        "library": None, "downloads": 4364, "likes": 118,
        "license": "mit", "params": None,
        "updated_at": None, "tags": [], "visibility": "public",
    },
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def proxy_view() -> dict:
    proxy = STATE["proxy"]
    configured = bool(
        proxy["enabled"] and proxy["host"] and 1 <= int(proxy["port"] or 0) <= 65535
    )
    return {
        "enabled": bool(proxy["enabled"]),
        "type": proxy["type"],
        "host": proxy["host"],
        "port": int(proxy["port"] or 0) or None,
        "username": proxy["username"],
        "has_password": bool(proxy["password"]),
        "configured": configured,
        "sources": [
            {
                "id": source_id,
                "label": label,
                "use_proxy": bool(configured and STATE["sources"][source_id]),
            }
            for source_id, label, _tool in SOURCES
        ],
        "config_file": "config.json",
    }


def downloads_view() -> dict:
    proxy = proxy_view()
    job = STATE["job"]
    return {
        "dir": STATE["dir"],
        "dir_configured": bool(STATE["dir"]),
        "dir_exists": bool(STATE["dir"]),
        "proxy_configured": proxy["configured"],
        "sources": [
            {
                "id": source_id, "label": label, "tool": tool,
                "available": True,
                "use_proxy": bool(proxy["configured"] and STATE["sources"][source_id]),
            }
            for source_id, label, tool in SOURCES
        ],
        "tools": {
            "hf": {"command": "hf", "path": "/srv/llm-router/.venv/bin/hf", "available": True},
            "modelscope": {
                "command": "modelscope",
                "path": "/srv/llm-router/.venv/bin/modelscope",
                "available": True,
            },
        },
        "jobs": [job] if job else [],
        "last_job_id": job["id"] if job else None,
        "config_file": "config.json",
    }


def make_job(body: dict) -> dict:
    repo = body.get("repo_id") or "Qwen/Qwen3-8B-GGUF"
    source_id = body.get("source", "huggingface")
    label = next((item[1] for item in SOURCES if item[0] == source_id), source_id)
    return {
        "id": "ui-job-1",
        "kind": "download",
        "source": source_id,
        "source_label": label,
        "repo_id": repo,
        "dir": STATE["dir"],
        "target_dir": f"{STATE['dir']}/{repo}",
        "files": body.get("files") or [],
        "files_count": len(body.get("files") or []),
        "total_bytes": 1049052,
        "status": "running",
        "started_at": now_iso(),
        "finished_at": None,
        "error": None,
        "cancel_requested": False,
        "exit_code": None,
        "log_file": "logs/downloads/ui-job-1.log",
        "log_tail": (
            "Fetching 2 files: 100%|##########| 2/2\n"
            "config.json: 100%|##########| 123/123\n"
            "sub/tokenizer.json: 100%|##########| 456/456\n"
            "saved to " + f"{STATE['dir']}/{repo}"
        ),
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

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except ValueError:
            return {}

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
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
            return self.send_json({"engines": {}})
        if path == "/models":
            return self.send_json({"object": "list", "data": []})
        if path == "/router/engines/paths":
            return self.send_json({"object": "list", "config_file": "config.json", "engines": []})
        if path == "/router/auth":
            return self.send_json({"configured": False, "source": None, "hint": None,
                                   "key_file": "api-key", "protected": ["/v1/*", "/props"]})
        if path == "/router/proxy":
            return self.send_json(proxy_view())
        if path == "/router/downloads":
            return self.send_json(downloads_view())
        if path == "/router/downloads/search":
            params = parse_qs(parsed.query)
            query = (params.get("q") or [""])[0]
            raw_sources = (params.get("sources") or [None])[0]
            selected = (
                [part for part in raw_sources.split(",") if part]
                if raw_sources
                else [source_id for source_id, _label, _tool in SOURCES]
            )
            STATE["search_q"] = query
            STATE["search_sources"] = selected
            items = [item for item in ITEMS if item["source"] in selected]
            if query == "boom":
                items = []
            sources = []
            for source_id, label, _tool in SOURCES:
                if source_id not in selected:
                    continue
                count = sum(1 for item in items if item["source"] == source_id)
                sources.append({
                    "id": source_id, "label": label, "ok": query != "boom",
                    "count": count, "total": None,
                    "error": "模拟搜索失败" if query == "boom" else None,
                    "elapsed_ms": 240,
                })
            return self.send_json({"query": query, "limit": 30, "sources": sources, "items": items})
        if path == "/router/downloads/job":
            job = STATE["job"]
            if not job:
                return self.send_json({"error": {"message": "not found"}}, 404)
            return self.send_json(job)
        if path == "/state":
            return self.send_json(STATE)
        return self.send_json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        path = urlsplit(self.path).path
        body = self.read_json()
        if path == "/router/proxy":
            proxy = STATE["proxy"]
            proxy["enabled"] = bool(body.get("enabled"))
            proxy["type"] = body.get("type", proxy["type"])
            proxy["host"] = str(body.get("host") or "")
            proxy["port"] = int(body.get("port") or 0)
            proxy["username"] = str(body.get("username") or "")
            if body.get("password"):
                proxy["password"] = str(body["password"])
            if body.get("clear_password") is True:
                proxy["password"] = ""
            if isinstance(body.get("sources"), dict):
                for source_id, use_proxy in body["sources"].items():
                    if source_id in STATE["sources"]:
                        STATE["sources"][source_id] = bool(use_proxy)
            if not proxy_view()["configured"]:
                for key in STATE["sources"]:
                    STATE["sources"][key] = False
            return self.send_json(proxy_view())
        if path == "/router/proxy/test":
            STATE["test_calls"].append(body)
            if not body.get("host") or not body.get("port"):
                return self.send_json({"error": {"message": "代理地址和端口不完整"}}, 400)
            return self.send_json({
                "ok": True, "reachable": True, "type": body.get("type", "http"),
                "address": f"{body.get('host')}:{body.get('port')}",
                "tests": [
                    {"target": "HF-Mirror", "ok": True, "status": 200, "elapsed_ms": 182},
                    {"target": "ModelScope", "ok": True, "status": 200, "elapsed_ms": 264},
                ],
                "error": None,
            })
        if path == "/router/downloads/settings":
            STATE["settings_posts"].append(body)
            if "dir" in body:
                raw = str(body["dir"])
                if not raw:
                    return self.send_json({"error": {"message": "下载目录不能为空"}}, 400)
                STATE["dir"] = raw if raw.startswith("/") else "/srv/" + raw
            if "sources" in body:
                if not proxy_view()["configured"]:
                    return self.send_json(
                        {"error": {"message": "代理尚未启用或配置不完整，不能为下载源启用代理"}},
                        400,
                    )
                for source_id, entry in (body["sources"] or {}).items():
                    if source_id in STATE["sources"]:
                        STATE["sources"][source_id] = bool(entry.get("use_proxy"))
            return self.send_json(downloads_view())
        if path == "/router/downloads/plan":
            source_id = body.get("source")
            repo = body.get("repo_id")
            files = [
                {"path": "config.json", "size": 123, "exists": False},
                {"path": "model.gguf", "size": 1048576, "exists": True},
                {"path": "sub/tokenizer.json", "size": 456, "exists": False},
            ]
            return self.send_json({
                "source": source_id,
                "source_label": next((label for sid, label, _ in SOURCES if sid == source_id), source_id),
                "repo_id": repo,
                "dir": STATE["dir"],
                "target_dir": f"{STATE['dir']}/{repo}",
                "files": files,
                "file_count": 3,
                "total_bytes": 1049155,
                "missing_bytes": 579,
                "existing_count": 1,
                "use_proxy": bool(STATE["sources"].get(source_id)),
            })
        if path == "/router/downloads/start":
            STATE["start"] = body
            STATE["job"] = make_job(body)
            STATE["cancel"] = False
            return self.send_json(STATE["job"], 202)
        if path == "/router/downloads/cancel":
            STATE["cancel"] = True
            job = STATE["job"]
            if job:
                job["status"] = "cancelled"
                job["cancel_requested"] = True
                job["finished_at"] = now_iso()
                job["error"] = "下载已取消"
            return self.send_json(job or {"error": {"message": "not found"}}, 200 if job else 404)
        if path == "/router/reload":
            return self.send_json({"status": "ok", "models": 0})
        return self.send_json({"error": {"message": "not found"}}, 404)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print(f"mock listening on {PORT}", flush=True)
    server.serve_forever()
