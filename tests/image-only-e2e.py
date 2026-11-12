#!/usr/bin/env python3
"""End-to-end test for image-only chat turns on KVMem-retrieval engines.

PolyStrata serve rejects a last user message that carries images but no text
("KVMem query needs text beside the image").  The gateway must append a default
instruction for engines with KVMem retrieval enabled (--kvmem-budget > 0 or
info.kvmem=1) and leave every other payload untouched.

Builds a throwaway gateway with fake engines that record the payloads they
receive; no real model weights are read and no browser is needed.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tmp" / "image-only-e2e"
if BASE.exists():
    shutil.rmtree(BASE)
BASE.mkdir(parents=True)
shutil.copy2(ROOT / "router.py", BASE / "router.py")
shutil.copy2(ROOT / "engine_setup.py", BASE / "engine_setup.py")
shutil.copy2(ROOT / "smoke.png", BASE / "smoke.png")

DEFAULT_TEXT = "请描述这张图片。"
TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def model_gguf(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [("general.architecture", 8, "qwen35"), ("general.name", 8, name)]
    parts = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", 0), struct.pack("<Q", len(values))]
    for key, value_type, value in values:
        key_bytes = key.encode("utf-8")
        parts.append(struct.pack("<Q", len(key_bytes)) + key_bytes)
        parts.append(struct.pack("<I", value_type))
        if value_type == 8:
            text = str(value).encode("utf-8")
            parts.append(struct.pack("<Q", len(text)) + text)
    path.write_bytes(b"".join(parts))


FAKE_RECORD = r'''#!/usr/bin/env python3
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def arg_value(*flags):
    for flag in flags:
        if flag in sys.argv[:-1]:
            return sys.argv[sys.argv.index(flag) + 1]
    return None


alias = "fake"
config = arg_value("--config")
models = []
if config:
    try:
        with open(config, encoding="utf-8") as fh:
            alias = json.load(fh).get("model_name") or alias
    except (OSError, ValueError):
        pass
    import glob
    for path in sorted(glob.glob(os.path.join(os.path.dirname(config), "strata-*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                name = json.load(fh).get("model_name")
        except (OSError, ValueError):
            continue
        if name and name not in models:
            models.append(name)
if alias not in models:
    models.append(alias)
state = {"current": alias}
host = arg_value("--host") or "127.0.0.1"
port = int(arg_value("--port") or "0")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send_json(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        elif self.path.rstrip("/") == "/v1/models":
            self.send_json(200, {"object": "list", "data": [
                {"id": alias, "object": "model", "status": "loaded"},
            ]})
        elif self.path.rstrip("/") == "/api/models":
            self.send_json(200, {"current": state["current"], "models": [
                {"id": name, "status": "loaded"} for name in models
            ]})
        else:
            self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            payload = {}
        if self.path.rstrip("/") == "/v1/chat/completions":
            with open(os.path.join(os.getcwd(), "received.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.send_json(200, {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": "fake",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        elif self.path.rstrip("/") == "/api/models/load":
            name = payload.get("model")
            if name in models:
                state["current"] = name
                self.send_json(200, {"status": "ok", "current": name})
            else:
                self.send_json(404, {"error": {"message": f"unknown model {name}"}})
        else:
            self.send_json(404, {"error": {"message": "not found"}})


server = ThreadingHTTPServer((host, port), Handler)


def stop(_signum, _frame):
    threading.Thread(target=server.shutdown, daemon=True).start()


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
server.serve_forever()
server.server_close()
'''


def fake_binary(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(FAKE_RECORD, encoding="utf-8")
    path.chmod(0o755)
    return path


# --- engines and model configs --------------------------------------------
kv_dir = BASE / "kvmem"
kv_binary = fake_binary(kv_dir, "strata-poly")
st_dir = BASE / "strata"
st_binary = fake_binary(st_dir, "strata")

kv_models = BASE / "kv-models"
kv_models.mkdir()
model_gguf(kv_models / "kv-on.gguf", "KV On")
model_gguf(kv_models / "kv-off.gguf", "KV Off")
kv_tokenizer = kv_models / "tokenizer"
kv_tokenizer.mkdir()
(kv_tokenizer / "vocab.json").write_text("{}", encoding="utf-8")
(kv_models / "strata-kv-on.json").write_text(json.dumps({
    "model_name": "kv-on",
    "exe": str(kv_binary),
    "args": ["--model", str(kv_models / "kv-on.gguf"), "--kvmem-budget", "32768"],
    "tokenizer": str(kv_tokenizer),
}, ensure_ascii=False), encoding="utf-8")
(kv_models / "strata-kv-off.json").write_text(json.dumps({
    "model_name": "kv-off",
    "exe": str(kv_binary),
    "args": ["--model", str(kv_models / "kv-off.gguf"), "--kvmem-budget", "0"],
    "tokenizer": str(kv_tokenizer),
}, ensure_ascii=False), encoding="utf-8")

st_models = BASE / "st-models"
st_models.mkdir()
model_gguf(st_models / "st-plain.gguf", "Strata Plain")
tokenizer = st_models / "tokenizer"
tokenizer.mkdir()
(tokenizer / "vocab.json").write_text("{}", encoding="utf-8")
(st_models / "strata-st-plain.json").write_text(json.dumps({
    "model_name": "st-plain",
    "exe": str(st_binary),
    "args": ["--native", str(st_models / "st-plain.gguf"), "--max-context", "4096"],
    "tokenizer": str(tokenizer),
}, ensure_ascii=False), encoding="utf-8")

# The console hides engines with no managed binaries; pretend installed.
setup_bin = BASE / "bin"
setup_bin.mkdir()
for name in (
    "llama-server", "llama-mtmd-cli", "strata", "strata-vision",
    "kvmem-llama-server", "kvmem-strata-poly", "kvmem-ninfer-serve",
):
    for candidate in (name, f"{name}-cuda", f"{name}-rocm"):
        path = setup_bin / candidate
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

port = free_port()
config = {
    "listen": {"host": "127.0.0.1", "port": port},
    "state_file": "state.json",
    "pid_file": "router.pid",
    "log_dir": "logs",
    "log_file": "logs/router.log",
    "model_params_file": "model-params.json",
    "smoke_image": "smoke.png",
    "console": {"enabled": True, "root": str(ROOT / "console"), "file": "index.html"},
    "engines": {
        "kvmem": {
            "kind": "kvmem",
            "host": "127.0.0.1",
            "port": free_port(),
            "cwd": str(kv_dir),
            "command": [str(kv_binary), "--config", "{config}", "--host", "{host}", "--port", "{port}"],
            "start_timeout_s": 8,
            "log_path": "logs/kvmem.log",
            "models": {"kind": "kvmem_dir", "dir": str(kv_models)},
        },
        "strata": {
            "kind": "strata",
            "host": "127.0.0.1",
            "port": free_port(),
            "cwd": str(st_dir),
            "command": [str(st_binary), "--config", "{config}", "--host", "{host}", "--port", "{port}"],
            "start_timeout_s": 8,
            "log_path": "logs/strata.log",
            "models": {"kind": "strata_dir", "dir": str(st_models)},
        },
    },
}
(BASE / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
url = f"http://127.0.0.1:{port}"
log = (BASE / "router.out").open("w", encoding="utf-8")
router = subprocess.Popen(
    [sys.executable, "router.py", "--host", "127.0.0.1", "--port", str(port)],
    cwd=BASE,
    stdout=log,
    stderr=subprocess.STDOUT,
    text=True,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)


def stop_router() -> None:
    if router.poll() is None:
        router.send_signal(signal.SIGTERM)
        try:
            router.wait(timeout=20)
        except subprocess.TimeoutExpired:
            router.kill()
            router.wait(timeout=5)


def wait_api() -> None:
    deadline = time.time() + 25
    while time.time() < deadline:
        if router.poll() is not None:
            raise SystemExit("router exited:\n" + (BASE / "router.out").read_text())
        try:
            requests.get(url + "/router/status", timeout=1)
            return
        except requests.RequestException:
            time.sleep(0.15)
    raise SystemExit("router did not start")


def status() -> dict:
    return requests.get(url + "/router/status", timeout=5).json()


def load(model: str) -> None:
    current = status().get("active_model")
    if current and current != model:
        requests.post(url + "/models/unload", json={"model": current}, timeout=20)
        deadline = time.time() + 20
        while status().get("active_model") and time.time() < deadline:
            time.sleep(0.2)
    response = requests.post(url + "/models/load", json={"model": model}, timeout=20)
    assert response.status_code in (200, 202), response.text
    deadline = time.time() + 30
    while time.time() < deadline:
        snap = status()
        if snap.get("active_model") == model and snap.get("phase") == "ready":
            return
        if router.poll() is not None:
            raise SystemExit("router exited:\n" + (BASE / "router.out").read_text())
        time.sleep(0.2)
    raise SystemExit(f"model {model} did not load: {status()}")


def chat(model: str, content) -> dict:
    response = requests.post(
        url + "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}]},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response.json()


def last_received(engine_dir: Path) -> dict:
    lines = (engine_dir / "received.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines, f"no payload recorded in {engine_dir}"
    return json.loads(lines[-1])


def image_part() -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}}


def forwarded_content(engine_dir: Path) -> list:
    return last_received(engine_dir)["messages"][-1]["content"]


try:
    wait_api()

    # KVMem retrieval on: bare image gets the default instruction appended.
    load("kv-on")
    chat("kv-on", [image_part()])
    content = forwarded_content(kv_dir)
    assert len(content) == 2 and content[-1] == {"type": "text", "text": DEFAULT_TEXT}, content

    # Text beside the image: payload passes through untouched.
    chat("kv-on", [image_part(), {"type": "text", "text": "图中是什么？"}])
    content = forwarded_content(kv_dir)
    assert len(content) == 2 and content[-1]["text"] == "图中是什么？", content

    # Retrieval disabled (budget 0): no injection even on the kvmem engine.
    load("kv-off")
    chat("kv-off", [image_part()])
    content = forwarded_content(kv_dir)
    assert len(content) == 1 and content[0]["type"] == "image_url", content

    # Strata without KVMem retrieval: untouched as well.
    load("st-plain")
    chat("st-plain", [image_part()])
    content = forwarded_content(st_dir)
    assert len(content) == 1 and content[0]["type"] == "image_url", content

    print("image-only e2e: PASS")
finally:
    stop_router()