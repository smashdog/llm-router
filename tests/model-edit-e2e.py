#!/usr/bin/env python3
"""End-to-end test for editing a model (engine / paths / parameters).

Builds a throwaway gateway with fake engines and checks the /router/models/edit
contract:

* unchanged engine+paths -> lightweight save, no validation job;
* changed model path -> full validation job, new id from the new file, old
  config removed and sampling/display name migrated;
* changed engine, same path -> validation keeps the model id and replaces the
  old engine config (no duplicate registry entries);
* failed validation -> old config bytes, sampling and display name restored;
* directory-scanned models: edits create the missing preset; a path/engine
  change adds a new id beside the scanned entry instead of replacing it.

Runs a real gateway, fake engine binaries and the real console in headless
Firefox.  No real model weights are read.
"""
from __future__ import annotations

import base64
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
BASE = ROOT / "tmp" / "model-edit-e2e"
OUT = BASE / "shots"
if BASE.exists():
    shutil.rmtree(BASE)
BASE.mkdir(parents=True)
OUT.mkdir()
shutil.copy2(ROOT / "router.py", BASE / "router.py")
shutil.copy2(ROOT / "engine_setup.py", BASE / "engine_setup.py")
shutil.copy2(ROOT / "smoke.png", BASE / "smoke.png")


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def write_gguf(path: Path, values: list[tuple[str, int, object]]) -> None:
    parts = [
        b"GGUF",
        struct.pack("<I", 3),
        struct.pack("<Q", 0),
        struct.pack("<Q", len(values)),
    ]
    for key, value_type, value in values:
        key_bytes = key.encode("utf-8")
        parts.append(struct.pack("<Q", len(key_bytes)) + key_bytes)
        parts.append(struct.pack("<I", value_type))
        if value_type == 8:
            text = str(value).encode("utf-8")
            parts.append(struct.pack("<Q", len(text)) + text)
        elif value_type == 4:
            parts.append(struct.pack("<I", int(value)))
        elif value_type == 7:
            parts.append(b"\x01" if value else b"\x00")
        else:
            raise AssertionError(value_type)
    path.write_bytes(b"".join(parts))


def model_gguf(path: Path, name: str, context: int = 8192) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(path, [
        ("general.architecture", 8, "qwen35"),
        ("general.name", 8, name),
        ("qwen35.context_length", 4, context),
    ])


def mmproj_gguf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(path, [
        ("clip.has_vision_encoder", 7, True),
        ("clip.has_audio_encoder", 7, False),
    ])


FAKE_SINGLE = r'''#!/usr/bin/env python3
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
        for token in sys.argv:
            if token.startswith(flag + "="):
                return token.split("=", 1)[1]
    return None


record = {"argv": sys.argv[1:]}
with open(os.path.join(os.getcwd(), "starts.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\n")

alias = arg_value("-a", "--alias")
config = arg_value("--config")
if config:
    try:
        with open(config, encoding="utf-8") as fh:
            cfg = json.load(fh)
        alias = cfg.get("model_name") or cfg.get("model_id") or alias
    except (OSError, ValueError):
        pass
alias = alias or "fake"
host = arg_value("--host") or "127.0.0.1"
port = int(arg_value("--port") or "0")


def has_image(payload):
    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


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
            image = has_image(payload)
            self.send_json(200, {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": alias,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong (image)" if image else "pong"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 30 if image else 12,
                    "completion_tokens": 2,
                    "total_tokens": 32 if image else 14,
                },
            })
        elif self.path.rstrip("/") == "/api/models":
            self.send_json(200, {"current": alias, "models": [
                {"id": alias, "status": "loaded"},
            ]})
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

FAKE_LLAMA = r'''#!/usr/bin/env python3
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--help" in sys.argv:
    print("usage: llama-server [options]")
    print("--cache-type-k TYPE  KV cache data type for K")
    print("                     allowed values: f32, f16, bf16, q8_0, q4_0, q5_0")
    print("--cache-type-v TYPE  KV cache data type for V")
    print("                     allowed values: f32, f16, bf16, q8_0, q4_0, q5_0")
    raise SystemExit(0)


def arg_value(*flags):
    for flag in flags:
        if flag in sys.argv[:-1]:
            return sys.argv[sys.argv.index(flag) + 1]
        for token in sys.argv:
            if token.startswith(flag + "="):
                return token.split("=", 1)[1]
    return None


def preset_models(path):
    names = []
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return names
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and stripped != "[*]":
            names.append(stripped[1:-1].strip())
    return names


preset = arg_value("--models-preset")
host = arg_value("--host") or "127.0.0.1"
port = int(arg_value("--port") or "0")
loaded = {name: "unloaded" for name in preset_models(preset)}
with open(os.path.join(os.getcwd(), "llama-starts.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": sys.argv[1:], "models": list(loaded)}) + "\n")


def has_image(payload):
    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


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
        elif self.path.rstrip("/") == "/models":
            self.send_json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "status": loaded.get(name, "unloaded")}
                for name in loaded
            ]})
        elif self.path.rstrip("/") == "/v1/models":
            self.send_json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "status": loaded.get(name, "unloaded")}
                for name in loaded
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
        path = self.path.rstrip("/")
        if path == "/models/load":
            loaded[str(payload.get("model"))] = "loaded"
            self.send_json(200, {"success": True})
        elif path == "/models/unload":
            loaded[str(payload.get("model"))] = "unloaded"
            self.send_json(200, {"success": True})
        elif path == "/v1/chat/completions":
            image = has_image(payload)
            self.send_json(200, {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": str(payload.get("model") or ""),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong (image)" if image else "pong"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 30 if image else 12,
                    "completion_tokens": 2,
                    "total_tokens": 32 if image else 14,
                },
            })
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


def fake_binary(directory: Path, name: str, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_server.py"
    script.write_text(source, encoding="utf-8")
    binary = directory / name
    binary.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


# --- fake engines ---------------------------------------------------------
llama_dir = BASE / "llama.cpp"
llama_binary = fake_binary(llama_dir, "llama-server", FAKE_LLAMA)
kv_a_dir = BASE / "kvmem-a"
kv_a_binary = fake_binary(kv_a_dir, "llama-kvmem-server", FAKE_SINGLE)
kv_b_dir = BASE / "kvmem-b"
kv_b_binary = fake_binary(kv_b_dir, "llama-kvmem-server", FAKE_SINGLE)
kv_bad_dir = BASE / "kvmem-bad"
kv_bad_binary = kv_bad_dir / "llama-kvmem-server"
kv_bad_dir.mkdir()
kv_bad_binary.write_text("#!/bin/sh\necho 'broken engine' >&2\nexit 3\n", encoding="utf-8")
kv_bad_binary.chmod(0o755)
strata_dir = BASE / "strata"
strata_binary = fake_binary(strata_dir, "strata", FAKE_SINGLE)
(strata_dir / "tools").mkdir(exist_ok=True)
(strata_dir / "tools" / "strata_tokenizer.py").write_text(
    "import sys\nfrom pathlib import Path\n"
    "out = Path(sys.argv[sys.argv.index('--out') + 1]) / 'tokenizer'\n"
    "out.mkdir(parents=True, exist_ok=True)\n"
    "(out / 'vocab.json').write_text('{}')\n"
    "print(out)\n",
    encoding="utf-8",
)

# --- model files ----------------------------------------------------------
llama_models = BASE / "models"
model_gguf(llama_models / "llama-alpha" / "llama-alpha.gguf", "Llama Alpha")
model_gguf(llama_models / "llama-beta" / "llama-beta.gguf", "Llama Beta")
model_gguf(llama_models / "llama-scan" / "llama-scan.gguf", "Llama Scan")
model_gguf(BASE / "extra" / "llama-delta.gguf", "Llama Delta")
mmproj_a = llama_models / "vision-a" / "mmproj-a.gguf"
mmproj_gguf(mmproj_a)
mmproj_b = llama_models / "vision-b" / "mmproj-b.gguf"
mmproj_gguf(mmproj_b)
preset = BASE / "models.ini"
preset.write_text(
    "version = 1\n\n"
    "[*]\n"
    "cache-type-k = q8_0\n"
    "cache-type-v = q8_0\n\n"
    "[llama-alpha]\n"
    "c = 4096\n"
    "model = llama-alpha/llama-alpha.gguf\n",
    encoding="utf-8",
)

kv_models_a = BASE / "kv-a-models"
kv_models_a.mkdir()
kv_model_a = kv_models_a / "kvl-alpha.gguf"
model_gguf(kv_model_a, "KVL Alpha")
kv_model_b = kv_models_a / "kvl-beta.gguf"
model_gguf(kv_model_b, "KVL Beta", 12288)
kv_config_a = kv_models_a / "kvmem-llama-kvl-alpha.json"
kv_config_a.write_text(json.dumps({
    "model_name": "kvl-alpha",
    "model_id": "kvl-alpha",
    "model": str(kv_model_a),
    "display_name": "KVL Alpha",
    "aliases": [],
    "exe": str(kv_a_binary),
    "args": ["--kvmem", "-c", "8192", "--kvmem-budget", "16384", "--kv-dtype", "q8_0"],
    "vision": {"mmproj": str(mmproj_a), "model": str(kv_model_a), "gpu": False},
}, ensure_ascii=False, indent=2), encoding="utf-8")
kv_models_b = BASE / "kv-b-models"
kv_models_b.mkdir()
kv_models_bad = BASE / "kv-bad-models"
kv_models_bad.mkdir()

strata_models = BASE / "strata-models"
strata_models.mkdir()
strata_model = strata_models / "strata-alpha.gguf"
model_gguf(strata_model, "Strata Alpha", 4096)
strata_tokenizer = strata_models / "tokenizer-a"
strata_tokenizer.mkdir()
(strata_tokenizer / "vocab.json").write_text("{}", encoding="utf-8")
strata_config = strata_models / "strata-strata-alpha.json"
strata_config.write_text(json.dumps({
    "model_name": "strata-alpha",
    "display_name": "Strata Alpha",
    "exe": str(strata_binary),
    "args": ["--native", str(strata_model), "--max-context", "4096"],
    "tokenizer": str(strata_tokenizer),
}, ensure_ascii=False, indent=2), encoding="utf-8")

port = free_port()
# The console hides engines with no managed binaries; pretend all products are
# installed.  Only bins exist, so reconcile_managed_engines() adds no slots.
setup_bin = BASE / "bin"
setup_bin.mkdir()
for name in (
    "llama-server", "llama-mtmd-cli", "strata", "strata-vision",
    "kvmem-llama-server", "kvmem-strata-poly", "kvmem-ninfer-serve",
    "kvmem-hipfire-server", "kvmem-hipfire-daemon",
):
    for candidate in (name, f"{name}-cuda", f"{name}-rocm"):
        path = setup_bin / candidate
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)


def engine_slot(kind: str, cwd: Path, binary: Path, models: dict, port_: int) -> dict:
    return {
        "kind": kind,
        "host": "127.0.0.1",
        "port": port_,
        "cwd": str(cwd),
        "command": [str(binary), "--host", "{host}", "--port", "{port}"],
        "start_timeout_s": 8,
        "log_path": f"logs/{kind}.log",
        "models": models,
    }


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
        "llama": {
            **engine_slot("llama_router", llama_dir, llama_binary, {
                "kind": "llama_dir",
                "dir": str(llama_models),
                "preset": str(preset),
            }, free_port()),
            "command": [
                str(llama_binary), "--models-preset", str(preset),
                "--models-dir", str(llama_models),
                "--host", "{host}", "--port", "{port}",
            ],
        },
        "kvmem_llama": engine_slot("kvmem_llama", kv_a_dir, kv_a_binary, {
            "kind": "kvmem_llama_dir", "dir": str(kv_models_a),
        }, free_port()),
        "kvmem_llama2": engine_slot("kvmem_llama", kv_b_dir, kv_b_binary, {
            "kind": "kvmem_llama_dir", "dir": str(kv_models_b),
        }, free_port()),
        "kvmem_llama_broken": engine_slot("kvmem_llama", kv_bad_dir, kv_bad_binary, {
            "kind": "kvmem_llama_dir", "dir": str(kv_models_bad),
        }, free_port()),
        "strata": {
            **engine_slot("strata", strata_dir, strata_binary, {
                "kind": "strata_dir", "dir": str(strata_models),
            }, free_port()),
            "command": [
                str(strata_binary), "--config", "{config}",
                "--host", "{host}", "--port", "{port}",
            ],
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
            log.flush()
            raise SystemExit("router exited:\n" + (BASE / "router.out").read_text())
        try:
            requests.get(url + "/router/status", timeout=1)
            return
        except requests.RequestException:
            time.sleep(0.15)
    raise SystemExit("router did not start")


def models() -> dict[str, dict]:
    data = requests.get(url + "/models", timeout=10).json()["data"]
    return {item["id"]: item for item in data}


def wait_job(job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = requests.get(url + "/router/models/job", params={"id": job_id}, timeout=10).json()
        if last.get("status") != "running":
            return last
        time.sleep(0.2)
    raise SystemExit(f"edit job did not finish: {last}")


def edit(payload: dict) -> tuple[int, dict]:
    response = requests.post(url + "/router/models/edit", json=payload, timeout=20)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"text": response.text}


def edit_and_wait(payload: dict) -> dict:
    status, body = edit(payload)
    assert status == 202, (status, body)
    assert body.get("mode") == "validate", body
    job = wait_job(body["job"]["id"])
    return job


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preset_sections() -> list[str]:
    names = []
    for line in preset.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            names.append(stripped[1:-1].strip())
    return names


checks = 0
wait_api()
try:
    initial = models()
    assert set(initial) >= {"llama-alpha", "llama-scan", "kvl-alpha", "strata-alpha"}, initial.keys()
    assert initial["llama-scan"]["source"] == "models_dir", initial["llama-scan"]
    assert initial["llama-alpha"]["source"] == "preset", initial["llama-alpha"]
    checks += 1

    # 1. Unchanged engine and paths: lightweight save, no job.
    saved_status, saved = edit({
        "edit_model": "kvl-alpha",
        "engine": "kvmem_llama",
        "model": str(kv_model_a),
        "vision": str(mmproj_a),
        "params": {"temperature": 0.5, "top_p": 0.9},
        "display_name": "KVL Alpha Custom",
        "engine_params": {"kvmem_block_tokens": 64},
    })
    assert saved_status == 200 and saved.get("mode") == "save", (saved_status, saved)
    item = models()["kvl-alpha"]
    assert item["params"]["temperature"] == 0.5, item
    assert item["display_name"] == "KVL Alpha Custom", item
    cfg = read_json(kv_config_a)
    assert cfg["args"][cfg["args"].index("--kvmem-block-tokens") + 1] == "64", cfg["args"]
    checks += 1

    # 2. Model path change: validation job, new id from the new file, old
    #    config/params cleaned up and the display name migrated.
    job = edit_and_wait({
        "edit_model": "kvl-alpha",
        "engine": "kvmem_llama",
        "model": str(kv_model_b),
        "vision": str(mmproj_a),
    })
    assert job["status"] == "ok", job
    assert job["name"] == "kvl-beta", job
    assert job["previous_model"] == "kvl-alpha", job
    steps = {step["key"]: step for step in job["steps"]}
    assert steps["backup"]["status"] == "ok", steps["backup"]
    assert steps["smoke"]["status"] == "ok", steps["smoke"]
    new_config = kv_models_a / "kvmem-llama-kvl-beta.json"
    assert new_config.is_file(), list(kv_models_a.iterdir())
    assert not kv_config_a.exists(), list(kv_models_a.iterdir())
    fresh = models()
    assert "kvl-alpha" not in fresh and "kvl-beta" in fresh, fresh.keys()
    assert fresh["kvl-beta"]["params"]["temperature"] == 0.5, fresh["kvl-beta"]
    assert fresh["kvl-beta"]["display_name"] == "KVL Alpha Custom", fresh["kvl-beta"]
    cfg = read_json(new_config)
    assert cfg["model"] == str(kv_model_b), cfg
    assert cfg["vision"]["mmproj"] == str(mmproj_a), cfg["vision"]
    checks += 1

    # 3. Engine change with the same path: the id stays, the config moves to
    #    the new engine and no duplicate id remains in the registry.
    job = edit_and_wait({
        "edit_model": "kvl-beta",
        "engine": "kvmem_llama2",
        "model": str(kv_model_b),
        "vision": str(mmproj_a),
        "params": {"temperature": 0.6},
    })
    assert job["status"] == "ok", job
    assert job["name"] == "kvl-beta", job
    assert job.get("previous_model") is None, job
    assert not list(kv_models_a.glob("kvmem-llama-*.json")), list(kv_models_a.iterdir())
    moved = kv_models_b / "kvmem-llama-kvl-beta.json"
    assert moved.is_file(), list(kv_models_b.iterdir())
    fresh = models()
    assert "kvl-beta" in fresh, fresh.keys()
    assert fresh["kvl-beta"]["engine_slot"] == "kvmem_llama2", fresh["kvl-beta"]
    assert fresh["kvl-beta"]["params"]["temperature"] == 0.6, fresh["kvl-beta"]
    assert not any(key.startswith("kvmem_llama/") for key in fresh), fresh.keys()
    checks += 1

    # 4. Failed validation: the old config, params and display name come back.
    old_bytes = moved.read_text(encoding="utf-8")
    job = edit_and_wait({
        "edit_model": "kvl-beta",
        "engine": "kvmem_llama_broken",
        "model": str(kv_model_b),
        "vision": str(mmproj_a),
        "display_name": "Should Roll Back",
    })
    assert job["status"] == "failed", job
    assert job["rollback"] is True, job
    assert str(moved) in (job.get("restored") or []), job
    assert moved.read_text(encoding="utf-8") == old_bytes, "old config bytes changed"
    assert not list(kv_models_bad.glob("kvmem-llama-*.json")), list(kv_models_bad.iterdir())
    fresh = models()
    assert fresh["kvl-beta"]["engine_slot"] == "kvmem_llama2", fresh["kvl-beta"]
    assert fresh["kvl-beta"]["params"]["temperature"] == 0.6, fresh["kvl-beta"]
    assert fresh["kvl-beta"]["display_name"] != "Should Roll Back", fresh["kvl-beta"]
    checks += 1

    # 5. Directory-scanned model: an engine change adds a suffixed id beside
    #    the scanned entry and leaves the original alone.
    before = models()["llama-scan"]
    assert before["source"] == "models_dir" and before["config_path"] is None, before
    job = edit_and_wait({
        "edit_model": "llama-scan",
        "engine": "kvmem_llama2",
        "model": str(llama_models / "llama-scan" / "llama-scan.gguf"),
        "vision": str(mmproj_b),
    })
    assert job["status"] == "ok", job
    assert job["name"] == "llama-scan-kvmem_llama", job
    assert job.get("previous_model") is None, job
    fresh = models()
    assert "llama-scan" in fresh and fresh["llama-scan"]["source"] == "models_dir", fresh.get("llama-scan")
    assert "llama-scan-kvmem_llama" in fresh, fresh.keys()
    assert fresh["llama-scan-kvmem_llama"]["engine_slot"] == "kvmem_llama2", fresh
    checks += 1

    # 6. Scanned llama model, same engine: a vision change re-validates and
    #    creates the missing models.ini section.
    job = edit_and_wait({
        "edit_model": "llama-scan",
        "engine": "llama",
        "model": str(llama_models / "llama-scan" / "llama-scan.gguf"),
        "vision": str(mmproj_b),
        "vision_device": "cpu",
        "params": {"temperature": 0.4},
    })
    assert job["status"] == "ok", job
    assert "llama-scan" in preset_sections(), preset_sections()
    fresh = models()["llama-scan"]
    assert fresh["source"] == "preset", fresh
    assert fresh["vision_path"] == str(mmproj_b), fresh
    cfg_lines = preset.read_text(encoding="utf-8")
    section = cfg_lines.split("[llama-scan]", 1)[1]
    assert "mmproj = " + str(mmproj_b) in section, section
    assert "mmproj-offload = off" in section, section
    assert fresh["params"]["temperature"] == 0.4, fresh
    checks += 1

    # 7. Preset model path change: new id, old section removed, the same
    #    models.ini file stays in one piece.
    job = edit_and_wait({
        "edit_model": "llama-alpha",
        "engine": "llama",
        "model": str(BASE / "extra" / "llama-delta.gguf"),
        "params": {"temperature": 0.2},
    })
    assert job["status"] == "ok", job
    assert job["name"] == "llama-delta", job
    assert job["previous_model"] == "llama-alpha", job
    sections = preset_sections()
    assert "llama-alpha" not in sections and "llama-delta" in sections, sections
    fresh = models()
    # The old weights still sit in the scanned models dir, so the old id
    # survives as a directory-scanned entry; only the preset went away.
    assert fresh["llama-alpha"]["source"] == "models_dir", fresh["llama-alpha"]
    assert fresh["llama-alpha"]["config_path"] is None, fresh["llama-alpha"]
    assert fresh["llama-delta"]["source"] == "preset", fresh["llama-delta"]
    assert fresh["llama-delta"]["params"]["temperature"] == 0.2, fresh["llama-delta"]
    checks += 1

    # 8. Preset model engine change: id is kept, the old section is detached,
    #    a foreign-engine config is written instead.
    job = edit_and_wait({
        "edit_model": "llama-delta",
        "engine": "kvmem_llama2",
        "model": str(BASE / "extra" / "llama-delta.gguf"),
        "params": {"temperature": 0.2},
    })
    assert job["status"] == "ok", job
    assert job["name"] == "llama-delta", job
    sections = preset_sections()
    assert "llama-delta" not in sections, sections
    fresh = models()
    assert fresh["llama-delta"]["engine_slot"] == "kvmem_llama2", fresh["llama-delta"]
    assert not any(key.startswith("llama/") for key in fresh), fresh.keys()
    checks += 1

    # 9. Strata draft path change: validation is triggered (the draft runtime
    #    is a directory, so no conversion is needed) and --mtp follows.
    draft_runtime = BASE / "draft-rt"
    draft_runtime.mkdir()
    for name in ("dense.txt", "dense.bin", "experts.bin"):
        (draft_runtime / name).write_bytes(b"x")
    job = edit_and_wait({
        "edit_model": "strata-alpha",
        "engine": "strata",
        "model": str(strata_model),
        "draft": str(draft_runtime),
        "params": {"temperature": 0.3},
    })
    assert job["status"] == "ok", job
    cfg = read_json(strata_config)
    args = [str(value) for value in cfg["args"]]
    assert args[args.index("--mtp") + 1] == str(draft_runtime), args
    assert cfg["drafting"] == "mtp" and cfg["mtp"] is True, cfg
    fresh = models()["strata-alpha"]
    assert fresh["draft_path"] == str(draft_runtime), fresh
    checks += 1
    print(f"PASS {checks} checks (api)")
except BaseException:
    log.flush()
    stop_router()
    raise

# --- browser UI: params dialog as an edit form ----------------------------
driver = None
try:
    driver_port = free_port()
    driver = subprocess.Popen(
        ["/snap/bin/geckodriver", "--port", str(driver_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    driver_base = f"http://127.0.0.1:{driver_port}"
    for _ in range(80):
        try:
            requests.get(driver_base + "/status", timeout=1)
            break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        raise SystemExit("geckodriver did not start")
    session = requests.post(driver_base + "/session", json={"capabilities": {"alwaysMatch": {
        "browserName": "firefox",
        "moz:firefoxOptions": {"args": ["-headless"]},
    }}}, timeout=60).json()["value"]["sessionId"]

    def wd(method: str, path: str, payload=None, timeout: int = 60):
        response = requests.request(method, f"{driver_base}/session/{session}{path}",
                                    json=payload, timeout=timeout)
        data = response.json()
        value = data.get("value")
        if isinstance(value, dict) and value.get("error"):
            raise RuntimeError(value)
        return value

    def js(script: str):
        return wd("POST", "/execute/sync", {"script": script, "args": []})

    def wait_ui(script: str, timeout: float = 25, message: str = ""):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                last = js(script)
                if last:
                    return last
            except Exception as exc:  # noqa: BLE001
                last = exc
            time.sleep(0.15)
        raise SystemExit(f"wait failed: {message or script}: {last}")

    def open_model_ui(model_id: str) -> None:
        js(f"""
          const card = [...document.querySelectorAll('#models .model')]
            .find(c => c.querySelector('.id').textContent.trim() === {json.dumps(model_id)});
          if (!card) throw new Error('missing model ' + {json.dumps(model_id)});
          card.querySelector('button[data-action="params"]').click();
          return true;
        """)
        wait_ui("return document.getElementById('params-dialog').open", message="params dialog")

    def toast_text() -> str:
        return js("""return [...document.querySelectorAll('#toasts .toast')]
          .map(node => node.textContent.trim()).join(' | ')""")

    wd("POST", "/window/rect", {"width": 1440, "height": 1200})
    wd("POST", "/url", {"url": url + "/"})
    wait_ui("return document.querySelectorAll('#models .model').length >= 9", message="model cards")
    js("""
      window.__uiErrors = [];
      window.addEventListener('error', e => window.__uiErrors.push(String(e.message)));
      window.addEventListener('unhandledrejection',
        e => window.__uiErrors.push(String(e.reason && e.reason.message || e.reason)));
      return true;
    """)

    # The dialog is an edit form: engine selector, model/vision paths, and a
    # save button that announces the re-validation before it is requested.
    open_model_ui("kvl-beta")
    state = js("""
      const slot = document.getElementById('params-engine-slot');
      return {
        engine: slot.value,
        options: [...slot.options].map(option => option.value),
        modelPath: document.getElementById('params-model-path').value,
        visionPath: document.getElementById('params-vision-path').value,
        contextHidden: document.getElementById('params-engine').hidden,
        save: document.getElementById('params-save').textContent,
        hint: document.getElementById('params-edit-hint').textContent,
      };
    """)
    assert state["engine"] == "kvmem_llama2", state
    assert {"llama", "kvmem_llama", "kvmem_llama2", "strata"} <= set(state["options"]), state
    assert state["modelPath"] == str(kv_model_b), state
    assert state["visionPath"] == str(mmproj_a), state
    assert state["contextHidden"] is False, state
    assert state["save"] == "保存", state
    assert "未改动引擎与文件" in state["hint"], state
    checks += 1

    def set_path(value: str) -> str:
        return js(f"""
          const el = document.getElementById('params-model-path');
          el.value = {json.dumps(value)};
          el.dispatchEvent(new Event('input', {{bubbles: true}}));
          return document.getElementById('params-save').textContent;
        """)

    start_budget = js("""
      const el = document.getElementById('params-dialog')
        .querySelector('[data-param-key="kvmem_budget"]');
      return el ? el.value : null;
    """)
    assert start_budget == "11264", start_budget
    assert set_path(str(kv_model_a)) == "保存并重新验证"
    # The old model's explicit 12288 context (and its derived 11264 budget)
    # must not follow the new 8192-cap file; the context comes back when the
    # path is reverted.
    assert js("return document.getElementById('params-context').value") == ""
    budget_before = js("""
      const el = document.getElementById('params-dialog')
        .querySelector('[data-param-key="kvmem_budget"]');
      return el ? el.value : null;
    """)
    assert budget_before == "", budget_before
    assert set_path(str(kv_model_b)) == "保存"
    assert js("return document.getElementById('params-context').value") == "12288"
    switched = js("""
      const slot = document.getElementById('params-engine-slot');
      slot.value = 'kvmem_llama';
      slot.dispatchEvent(new Event('change', {bubbles: true}));
      return {
        save: document.getElementById('params-save').textContent,
        hint: document.getElementById('params-edit-hint').textContent,
        modelPath: document.getElementById('params-model-path').value,
      };
    """)
    assert switched["save"] == "保存并重新验证", switched
    assert "重新验证" in switched["hint"], switched
    assert switched["modelPath"] == str(kv_model_b), switched
    js("""
      const slot = document.getElementById('params-engine-slot');
      slot.value = 'kvmem_llama2';
      slot.dispatchEvent(new Event('change', {bubbles: true}));
      return true;
    """)
    assert js("return document.getElementById('params-save').textContent") == "保存"
    checks += 1

    # Save with a new model path: the dialog runs the validation job and
    # closes only after the old config is gone and the id has migrated.
    set_path(str(kv_model_a))
    js("""
      document.getElementById('params-save').click();
      return true;
    """)
    wait_ui("return !document.getElementById('params-dialog').open",
            timeout=60, message="dialog closes after a successful edit")
    toasts = toast_text()
    assert "kvl-alpha" in toasts and "kvl-beta" in toasts, toasts
    wait_ui("""return [...document.querySelectorAll('#models .model .id')]
      .some(node => node.textContent.trim() === 'kvl-alpha')""", message="kvl-alpha card")
    fresh = models()
    assert "kvl-beta" not in fresh and "kvl-alpha" in fresh, fresh.keys()
    assert fresh["kvl-alpha"]["engine_slot"] == "kvmem_llama2", fresh["kvl-alpha"]
    assert (kv_models_b / "kvmem-llama-kvl-alpha.json").is_file(), list(kv_models_b.iterdir())
    assert not (kv_models_b / "kvmem-llama-kvl-beta.json").exists(), list(kv_models_b.iterdir())
    checks += 1

    # A failed validation keeps the dialog open with the rollback note and
    # restores the old entry.
    open_model_ui("kvl-alpha")
    js("""
      const slot = document.getElementById('params-engine-slot');
      slot.value = 'kvmem_llama_broken';
      slot.dispatchEvent(new Event('change', {bubbles: true}));
      document.getElementById('params-save').click();
      return true;
    """)
    wait_ui("return document.getElementById('params-save').textContent === '重试保存'",
            timeout=90, message="failed validation returns the button")
    failure = js("""
      return {
        message: document.getElementById('params-error').textContent,
        progress: document.getElementById('params-progress').textContent,
        open: document.getElementById('params-dialog').open,
        toasts: [...document.querySelectorAll('#toasts .toast')]
          .map(node => node.textContent.trim()).join(' | '),
      };
    """)
    assert failure["open"] is True, failure
    assert "旧模型已恢复" in failure["progress"], failure
    assert failure["message"].strip(), failure
    assert "保存失败" in failure["toasts"], failure
    js("document.getElementById('params-cancel').click(); return true;")
    wait_ui("return !document.getElementById('params-dialog').open", message="dialog closed")
    fresh = models()
    assert fresh["kvl-alpha"]["engine_slot"] == "kvmem_llama2", fresh["kvl-alpha"]
    assert (kv_models_b / "kvmem-llama-kvl-alpha.json").is_file(), list(kv_models_b.iterdir())
    assert not list(kv_models_bad.glob("kvmem-llama-*.json")), list(kv_models_bad.iterdir())
    checks += 1

    errors = js("return window.__uiErrors")
    assert errors == [], errors
    checks += 1
    print(f"PASS {checks} checks (api + ui)")
finally:
    if driver is not None:
        try:
            requests.delete(f"{driver_base}/session/{session}", timeout=10)
        except Exception:
            pass
        try:
            driver.terminate()
            driver.wait(5)
        except Exception:
            try:
                driver.kill()
            except Exception:
                pass
    log.flush()
    stop_router()
    log.close()
