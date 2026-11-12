#!/usr/bin/env python3
"""End-to-end test for the KVMem + hipfire engine.

Builds a throwaway gateway with a fake hipfire CLI that answers the version
probe, serves one HFQM model per process (health loading window, /v1/models
without status fields, SSE streaming) and reaps its fake daemon child on
SIGTERM. Covers engine creation, adding an HFQM artifact, the trained-context
--max-seq fallback, KVMem env defaults, engine_params write/clear, validation
failures, restart-on-switch, daemon teardown and the manage-view schemas.
Console dialog coverage lands with the params-form task.
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
BASE = ROOT / "tmp" / "kvmem-hipfire-e2e"
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


def write_hfqm(path: Path, context: int) -> None:
    """Minimal HFQM artifact: 32-byte header + metadata JSON + fake weights.

    Header layout (little endian): magic "HFQM", version u32, arch u32,
    tensor count u32, metadata offset u64, payload offset u64. The router
    reads config.text_config.max_position_embeddings from the metadata.
    """
    meta = json.dumps({
        "architecture": "qwen3_5",
        "config": {"text_config": {"max_position_embeddings": context}},
    }).encode("utf-8")
    data_off = (32 + len(meta) + 4095) & ~4095
    header = b"HFQM" + struct.pack("<III", 1, 5, 2) + struct.pack("<QQ", 32, data_off)
    path.write_bytes(header + meta + b"\x00" * (data_off - 32 - len(meta)) + b"\x00" * 16)


# --- fake hipfire CLI + daemon ---------------------------------------------
engine_dir = BASE / "hipfire-kvmem-v046"
engine_dir.mkdir()
fake_binary = engine_dir / "kvmem-hipfire-server"
fake_daemon_binary = engine_dir / "kvmem-hipfire-daemon"
fake_server = engine_dir / "fake_server.py"
fake_daemon = engine_dir / "fake_daemon.py"
fake_daemon.write_text(
    '''#!/usr/bin/env python3
import time
while True:
    time.sleep(3600)
''',
    encoding="utf-8",
)
fake_server.write_text(
    '''#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--version" in sys.argv:
    print("hipfire 0.4.6 (fake00000000; kvmem-v046)")
    raise SystemExit(0)

VALUE_FLAGS = {
    "--model", "--max-seq", "--spec", "--speculation", "--idle-timeout",
    "--kv-mode", "--kv-k", "--kv-v", "--kv-backend", "--vision", "--tp",
    "--swap-host-bytes", "--swap-disk-bytes", "--swap-dir",
    "--continuous-batch-size", "--api-key-file",
}
argv = sys.argv[1:]
model = None
max_seq = None
spec = None
bind = None
i = 0
positional = []
while i < len(argv):
    token = argv[i]
    if token in VALUE_FLAGS and i + 1 < len(argv):
        if token == "--model":
            model = argv[i + 1]
        elif token == "--max-seq":
            max_seq = argv[i + 1]
        elif token in ("--spec", "--speculation"):
            spec = argv[i + 1]
        i += 2
        continue
    if token.startswith("--") and "=" in token:
        flag, value = token.split("=", 1)
        if flag == "--model":
            model = value
        elif flag == "--max-seq":
            max_seq = value
        elif flag in ("--spec", "--speculation"):
            spec = value
        i += 1
        continue
    if not token.startswith("-") and token != "serve":
        positional.append(token)
    i += 1
for token in positional:
    if ":" in token:
        bind = token
        break
host, _, port_text = (bind or "127.0.0.1:0").rpartition(":")
port = int(port_text or "0")

record = {
    "argv": argv,
    "model": model,
    "max_seq": max_seq,
    "spec": spec,
    "bind": bind,
    "env": {key: os.environ.get(key) for key in (
        "HIPFIRE_KVMEM", "HIPFIRE_SERVE_MULTI_SLOT", "HIPFIRE_KV_MODE",
        "HIPFIRE_KVMEM_BUDGET_TOKENS", "HIPFIRE_SERVE_MULTI_SLOT_SLOTS",
        "HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK", "HIPFIRE_DAEMON_BIN",
        "HIPFIRE_SPILL_DIR", "HIPFIRE_DEVICES",
    )},
}
with open(os.path.join(os.getcwd(), "starts.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, ensure_ascii=False) + "\\n")

# The real CLI spawns its daemon as a piped child in the same process group
# and reaps it in Drop on graceful shutdown; mirror that lifecycle.
daemon_bin = os.environ.get("HIPFIRE_DAEMON_BIN") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kvmem-hipfire-daemon")
daemon = subprocess.Popen([daemon_bin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

started = time.time()


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
            # Up only after a short load window; before that 503 + loading.
            if time.time() - started < 0.6:
                self.send_json(503, {"status": "loading", "loading_model": model})
            else:
                self.send_json(200, {
                    "status": "ok", "model": model,
                    "n_ctx": int(max_seq or 0), "capabilities": {},
                })
        elif self.path.rstrip("/") == "/v1/models":
            # hipfire entries carry no "status" field.
            self.send_json(200, {"object": "list", "data": [
                {"id": model, "object": "model", "owned_by": "hipfire",
                 "meta": {"n_ctx": int(max_seq or 0)}},
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
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                for piece in ("po", "ng"):
                    chunk = {
                        "id": "chatcmpl-fake", "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": piece},
                                     "finish_reason": None}],
                    }
                    self.wfile.write(("data: " + json.dumps(chunk) + "\\n\\n").encode())
                    self.wfile.flush()
                done = {
                    "id": "chatcmpl-fake", "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2,
                              "total_tokens": 14},
                }
                self.wfile.write(("data: " + json.dumps(done) + "\\n\\n").encode())
                self.wfile.write(b"data: [DONE]\\n\\n")
                self.wfile.flush()
                return
            self.send_json(200, {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2,
                          "total_tokens": 14},
            })
        else:
            self.send_json(404, {"error": {"message": "not found"}})


server = ThreadingHTTPServer((host or "127.0.0.1", port), Handler)


def stop(_signum, _frame):
    threading.Thread(target=server.shutdown, daemon=True).start()


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
server.serve_forever()
server.server_close()
# Graceful shutdown reaps the daemon child (EngineInner::Drop semantics).
daemon.terminate()
try:
    daemon.wait(timeout=5)
except subprocess.TimeoutExpired:
    daemon.kill()
    daemon.wait(timeout=5)
''',
    encoding="utf-8",
)
fake_binary.write_text(
    f"#!/bin/sh\nexec {sys.executable} {fake_server} \"$@\"\n", encoding="utf-8"
)
fake_binary.chmod(0o755)
fake_daemon_binary.write_text(
    f"#!/bin/sh\nexec {sys.executable} {fake_daemon} \"$@\"\n", encoding="utf-8"
)
fake_daemon_binary.chmod(0o755)
# The console lists only engines whose managed products are installed;
# pretend every product is, so the manage view can offer kvmem_hipfire.
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

# --- model files ----------------------------------------------------------
models_root = BASE / "models"
models_root.mkdir()
main_model = models_root / "Qwen3.8-27B.mq4-xts"
write_hfqm(main_model, 262144)
second_model = models_root / "Qwen3.8-27B-2.mq4-xts"
write_hfqm(second_model, 131072)
bad_model = models_root / "not-hfqm.bin"
bad_model.write_bytes(b"NOPE" + b"\x00" * 64)

# --- gateway --------------------------------------------------------------
port = free_port()
config = {
    "listen": {"host": "127.0.0.1", "port": port},
    "state_file": "state.json",
    "pid_file": "router.pid",
    "log_dir": "logs",
    "log_file": "logs/router.log",
    "model_params_file": "model-params.json",
    "smoke_image": "smoke.png",
    "console": {
        "enabled": True,
        "root": str(ROOT / "console"),
        "file": "index.html",
    },
    "engines": {},
}
(BASE / "config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
)
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


def kill_fakes() -> None:
    for script in (fake_server, fake_daemon):
        try:
            found = subprocess.run(
                ["pgrep", "-f", str(script)], capture_output=True, text=True, check=False
            ).stdout
        except OSError:
            continue
        for token in found.split():
            try:
                os.kill(int(token), signal.SIGTERM)
            except (OSError, ValueError):
                pass


def stop_router() -> None:
    if router.poll() is None:
        router.send_signal(signal.SIGTERM)
        try:
            router.wait(timeout=20)
        except subprocess.TimeoutExpired:
            router.kill()
            router.wait(timeout=5)
    kill_fakes()


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


def status() -> dict:
    return requests.get(url + "/router/status", timeout=10).json()


def models() -> dict[str, dict]:
    data = requests.get(url + "/models", timeout=10).json()["data"]
    return {item["id"]: item for item in data}


def wait_idle(timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = status()
        if last.get("async_operation") is None and not last.get("switching"):
            return last
        time.sleep(0.2)
    raise SystemExit(f"gateway did not settle: {last}")


def wait_job(job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = requests.get(url + "/router/models/job", params={"id": job_id}, timeout=10).json()
        if last.get("status") != "running":
            return last
        time.sleep(0.2)
    raise SystemExit(f"model job did not finish: {last}")


def add_model(**payload) -> dict:
    response = requests.post(url + "/router/models/add", json=payload, timeout=20)
    assert response.status_code in (200, 202), response.text
    return wait_job(response.json()["id"])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def starts() -> list[dict]:
    path = engine_dir / "starts.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def daemon_pids() -> list[int]:
    try:
        found = subprocess.run(
            ["pgrep", "-f", str(fake_daemon)], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return []
    return [int(token) for token in found.split() if token.isdigit()]


checks = 0
wait_api()
try:
    # 1. Create the fixed kvmem_hipfire slot through the management API.
    config_dir = BASE / "models-kvmem-hipfire"
    template = config_dir / "kvmem-hipfire.template.json"
    created = requests.post(url + "/router/engines/paths", json={
        "engine": "kvmem_hipfire",
        "create": True,
        "paths": {
            "binary": str(fake_binary),
            "cwd": str(engine_dir),
            "models_dir": str(config_dir),
            "template": str(template),
        },
    }, timeout=30)
    body = created.json()
    assert created.status_code == 200, body
    assert body.get("created") is True, body
    assert template.is_file(), "template was not created"
    template_doc = read_json(template)
    assert template_doc["args"] == ["--spec", "auto"], template_doc
    paths = requests.get(url + "/router/engines/paths", timeout=10).json()
    slot = next((item for item in paths.get("engines") or [] if item.get("engine") == "kvmem_hipfire"), None)
    assert slot is not None and slot.get("configured") is True, paths
    assert status()["engines"]["kvmem_hipfire"]["kind"] == "kvmem_hipfire"
    checks += 1

    # 2. Add the HFQM artifact: trained ceiling, KVMem budget, q8, smoke.
    job = add_model(
        engine="kvmem_hipfire",
        model=str(main_model),
        kvmem_budget=32768,
        smoke=True,
    )
    assert job["status"] == "ok", job
    steps = {step["key"]: step for step in job["steps"]}
    assert all(step["status"] == "ok" for step in job["steps"]), job["steps"]
    assert "262,144" in steps["metadata"]["detail"], steps["metadata"]
    assert "预算 32,768" in steps["kv"]["detail"], steps["kv"]
    assert "q8" in steps["kv"]["detail"], steps["kv"]
    checks += 1

    model_id = "Qwen3.8-27B"
    item = models()[model_id]
    assert item["engine"] == "kvmem_hipfire", item
    assert item["source"] == "kvmem_hipfire", item
    assert item["context_length"] == 262144, item
    assert item["cache_type_k"] == "q8" and item["cache_type_v"] == "q8", item
    assert item["engine_params"]["kvmem_budget"] == 32768, item
    assert item["engine_params"]["spec"] == "auto", item
    checks += 1

    model_cfg_path = config_dir / "kvmem-hipfire-qwen3.8-27b.json"
    assert model_cfg_path.is_file(), list(config_dir.iterdir())
    model_cfg = read_json(model_cfg_path)
    args = [str(value) for value in model_cfg["args"]]
    assert model_cfg["model"] == str(main_model), model_cfg
    assert "--model" not in args and "--max-seq" not in args, args
    assert "--host" not in args and "--port" not in args and "--detach" not in args, args
    assert args[args.index("--spec") + 1] == "auto", args
    assert model_cfg["env"]["HIPFIRE_KVMEM_BUDGET_TOKENS"] == "32768", model_cfg
    assert "max_seq" not in model_cfg, model_cfg
    checks += 1

    # 3. The start command: absolute --model, ceiling --max-seq, host:port
    # positional, KVMem env defaults and the sibling daemon binary.
    first = starts()[0]
    assert first["model"] == str(main_model), first
    assert first["max_seq"] == "262144", first
    assert first["spec"] == "auto", first
    assert first["bind"] == f"127.0.0.1:{status()['engines']['kvmem_hipfire']['port']}", first
    assert first["argv"][0] == "serve", first
    env = first["env"]
    assert env["HIPFIRE_KVMEM"] == "1", env
    assert env["HIPFIRE_SERVE_MULTI_SLOT"] == "true", env
    assert env["HIPFIRE_KV_MODE"] == "q8", env
    assert env["HIPFIRE_KVMEM_BUDGET_TOKENS"] == "32768", env
    assert env["HIPFIRE_SERVE_MULTI_SLOT_SLOTS"] == "4", env
    assert env["HIPFIRE_DAEMON_BIN"] == str(fake_daemon_binary), env
    assert status()["active_model"] == model_id, status()
    assert daemon_pids(), "fake daemon child was not spawned"
    checks += 1

    # 4. engine_params: write args/env/config targets, restart, read back.
    saved = requests.post(url + "/router/params", json={
        "model": model_id, "params": {},
        "engine_params": {
            "spec": "ngram",
            "idle_timeout": 300,
            "slots": 2,
            "prefill_chunk": 2048,
            "kvmem_budget": 16384,
            "kvmem_enable": True,
            "sink_blocks": 2,
            "recent_blocks": 3,
            "spill_dir": str(BASE / "spill"),
            "devices": "0",
            "max_seq": 65536,
        },
    }, timeout=20)
    assert saved.status_code == 200, saved.text
    saved_values = saved.json()["engine_params"]
    assert saved_values["spec"] == "ngram", saved_values
    assert saved_values["max_seq"] == 65536, saved_values
    assert saved_values["kvmem_budget"] == 16384, saved_values
    wait_idle()
    updated = read_json(model_cfg_path)
    updated_args = [str(value) for value in updated["args"]]
    assert updated_args[updated_args.index("--spec") + 1] == "ngram", updated_args
    assert updated_args[updated_args.index("--idle-timeout") + 1] == "300", updated_args
    assert updated["max_seq"] == 65536, updated
    assert updated["env"]["HIPFIRE_SERVE_MULTI_SLOT_SLOTS"] == "2", updated
    assert updated["env"]["HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK"] == "2048", updated
    assert updated["env"]["HIPFIRE_KVMEM_BUDGET_TOKENS"] == "16384", updated
    assert updated["env"]["HIPFIRE_KVMEM"] == "1", updated
    assert updated["env"]["HIPFIRE_KVMEM_SINK_BLOCKS"] == "2", updated
    assert updated["env"]["HIPFIRE_KVMEM_RECENT_BLOCKS"] == "3", updated
    assert updated["env"]["HIPFIRE_SPILL_DIR"] == str(BASE / "spill"), updated
    assert updated["env"]["HIPFIRE_DEVICES"] == "0", updated
    last_start = starts()[-1]
    assert last_start["max_seq"] == "65536", last_start
    assert last_start["spec"] == "ngram", last_start
    assert last_start["env"]["HIPFIRE_SERVE_MULTI_SLOT_SLOTS"] == "2", last_start
    assert last_start["env"]["HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK"] == "2048", last_start

    # Clearing returns to router/engine defaults (ceiling --max-seq again).
    cleared = requests.post(url + "/router/params", json={
        "model": model_id, "params": {},
        "engine_params": {
            "spec": None, "idle_timeout": None, "max_seq": None,
            "prefill_chunk": None, "spill_dir": None, "devices": None,
        },
    }, timeout=20)
    assert cleared.status_code == 200, cleared.text
    cleared_values = cleared.json()["engine_params"]
    assert "spec" not in cleared_values and "max_seq" not in cleared_values, cleared_values
    wait_idle()
    cleared_cfg = read_json(model_cfg_path)
    cleared_args = [str(value) for value in cleared_cfg["args"]]
    assert "--spec" not in cleared_args and "--idle-timeout" not in cleared_args, cleared_args
    assert "max_seq" not in cleared_cfg, cleared_cfg
    assert "HIPFIRE_SPILL_DIR" not in cleared_cfg.get("env", {}), cleared_cfg
    assert starts()[-1]["max_seq"] == "262144", starts()[-1]
    for bad, needle in (
        ({"bogus": 1}, "unknown engine parameter"),
        ({"spec": "warp"}, "expected one of"),
        ({"slots": 0}, "slots"),
        ({"kvmem_budget": 64}, "kvmem_budget"),
    ):
        rejected = requests.post(url + "/router/params", json={
            "model": model_id, "params": {}, "engine_params": bad,
        }, timeout=15)
        assert rejected.status_code == 400, (bad, rejected.text)
    checks += 1

    # 5. Streaming chat through the gateway (SSE passthrough).
    stream = requests.post(url + "/v1/chat/completions", json={
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True,
    }, timeout=30, stream=True)
    assert stream.status_code == 200, stream.text
    text = stream.content.decode("utf-8")
    pieces = []
    saw_done = False
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        chunk = json.loads(payload)
        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        pieces.append(delta.get("content") or "")
    assert "".join(pieces) == "pong", text[:400]
    assert saw_done, text[:400]
    checks += 1

    # 6. Validation failures must not write a config or register a model.
    before = set(models())
    bad_cache = requests.post(url + "/router/models/add", json={
        "engine": "kvmem_hipfire", "model": str(main_model),
        "name": "bad-cache", "cache_type_k": "f16", "smoke": False,
    }, timeout=15)
    assert bad_cache.status_code == 400 and "q8" in bad_cache.text, bad_cache.text
    bad_vision = requests.post(url + "/router/models/add", json={
        "engine": "kvmem_hipfire", "model": str(main_model),
        "name": "bad-vision", "vision": str(main_model), "smoke": False,
    }, timeout=15)
    assert bad_vision.status_code == 400 and "sidecar" in bad_vision.text, bad_vision.text
    bad_context = add_model(
        engine="kvmem_hipfire", model=str(main_model),
        name="ctx-too-big", context_length=999999, smoke=False,
    )
    assert bad_context["status"] == "failed", bad_context
    assert "超过模型训练上限" in bad_context["error"], bad_context
    bad_magic = add_model(
        engine="kvmem_hipfire", model=str(bad_model), name="bad-magic", smoke=False,
    )
    assert bad_magic["status"] == "failed", bad_magic
    assert "HFQM" in bad_magic["error"], bad_magic
    bad_dir = add_model(
        engine="kvmem_hipfire", model=str(models_root), name="bad-dir", smoke=False,
    )
    assert bad_dir["status"] == "failed", bad_dir
    unknown = requests.post(url + "/router/models/add", json={
        "engine": "kvmem_hipfire", "model": str(main_model),
        "name": "unknown-param", "smoke": False,
        "engine_params": {"bogus": 1},
    }, timeout=15)
    assert unknown.status_code == 400 and "unknown engine parameter" in unknown.text, unknown.text
    after = set(models())
    assert before == after, (before, after)
    for slug in ("bad-cache", "bad-vision", "ctx-too-big", "bad-magic", "bad-dir", "unknown-param"):
        assert not list(config_dir.glob(f"kvmem-hipfire-{slug}.json")), slug
    checks += 1

    # 7. A second model, restart-on-switch, explicit context, delete.
    job2 = add_model(
        engine="kvmem_hipfire", model=str(second_model),
        context_length=32768, smoke=False,
    )
    assert job2["status"] == "ok", job2
    second_id = "Qwen3.8-27B-2"
    assert models()[second_id]["context_length"] == 32768, models()[second_id]
    second_cfg = read_json(config_dir / "kvmem-hipfire-qwen3.8-27b-2.json")
    assert second_cfg["max_seq"] == 32768, second_cfg
    loaded = requests.post(url + "/models/load", json={"model": second_id}, timeout=20)
    assert loaded.status_code in (200, 202), loaded.text
    wait_idle()
    switched = starts()[-1]
    assert switched["model"] == str(second_model), switched
    assert switched["max_seq"] == "32768", switched
    loaded = requests.post(url + "/models/load", json={"model": model_id}, timeout=20)
    assert loaded.status_code in (200, 202), loaded.text
    wait_idle()
    assert starts()[-1]["model"] == str(main_model), starts()[-1]
    deleted = requests.post(url + "/router/models/delete", json={"model": second_id}, timeout=30)
    assert deleted.status_code == 200, deleted.text
    assert second_id not in models()
    assert not (config_dir / "kvmem-hipfire-qwen3.8-27b-2.json").exists()
    checks += 1

    # 8. Unload stops the engine and the daemon child exits with it.
    unloaded = requests.post(url + "/models/unload", json={"model": model_id}, timeout=20)
    assert unloaded.status_code in (200, 202), unloaded.text
    wait_idle()
    stopped_status = status()
    assert stopped_status["engines"]["kvmem_hipfire"]["alive"] is False, stopped_status
    assert stopped_status["active_model"] is None, stopped_status
    deadline = time.time() + 10
    while time.time() < deadline and daemon_pids():
        time.sleep(0.2)
    assert not daemon_pids(), "hipfire daemon survived the engine stop"
    checks += 1

    # 9. Manage view exposes the slot, schemas and sampling params.
    manage = requests.get(url + "/router/models/manage", timeout=10).json()
    entry = manage["engines"]["kvmem_hipfire"]
    assert entry["kind"] == "kvmem_hipfire" and entry["label"] == "KVMem + hipfire", entry
    assert entry["category"] == "kvmem", entry
    assert entry["defaults"]["kvmem_budget"] == 32768, entry
    schema = manage["kvmem_hipfire_engine_schema"]
    keys = {spec["key"] for spec in schema}
    assert {"max_seq", "slots", "idle_timeout", "spec", "kvmem_enable",
            "kvmem_budget", "paged_pages", "sink_blocks", "recent_blocks",
            "spill_dir", "spill_bytes", "prefill_chunk", "devices"} <= keys, keys
    groups = {spec["group"] for spec in schema}
    assert groups == {"basic", "spec", "kvmem", "spill", "advanced"}, groups
    assert "kvmem_hipfire" in manage["param_schema"], manage["param_schema"].keys()
    assert requests.get(url + "/", timeout=10).status_code == 200
    checks += 1
except BaseException:
    stop_router()
    raise

stop_router()
print(f"kvmem-hipfire e2e: {checks} check groups passed; artifacts in {BASE}")