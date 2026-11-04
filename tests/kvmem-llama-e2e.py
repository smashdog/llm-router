#!/usr/bin/env python3
"""End-to-end test for the KVMem + llama.cpp engine.

Builds a throwaway gateway with a fake llama-kvmem-server that answers the
capability probe and a tiny OpenAI-compatible HTTP surface. Covers engine
creation, adding a model directory (GGUF pick + mmproj), KV dtype/budget
flags, vision CPU/GPU wiring, restart-on-switch, validation failures and the
console dialogs.
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
BASE = ROOT / "tmp" / "kvmem-llama-e2e"
OUT = BASE / "shots"
if BASE.exists():
    shutil.rmtree(BASE)
BASE.mkdir(parents=True)
OUT.mkdir()
shutil.copy2(ROOT / "router.py", BASE / "router.py")
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


# --- fake llama-kvmem-server ----------------------------------------------
engine_dir = BASE / "kvmem-llamacpp090"
engine_dir.mkdir()
fake_binary = engine_dir / "llama-kvmem-server"
fake_server = engine_dir / "fake_server.py"
fake_server.write_text(
    '''#!/usr/bin/env python3
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def arg_value(flag):
    if flag in sys.argv[:-1]:
        return sys.argv[sys.argv.index(flag) + 1]
    for token in sys.argv:
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


if "--kvmem-capabilities" in sys.argv:
    print(json.dumps({
        "schema_version": 1,
        "backend": "llamacpp",
        "kv_dtypes": ["f16", "f32", "q8_0", "q5_0", "q4_0"],
        "kvmem": True,
    }))
    raise SystemExit(0)

model = arg_value("-m") or arg_value("--model")
alias = arg_value("-a") or arg_value("--alias") or "fake"
host = arg_value("--host") or "127.0.0.1"
port = int(arg_value("--port") or "0")
record = {
    "argv": sys.argv[1:],
    "model": model,
    "alias": alias,
    "mmproj": arg_value("--mmproj"),
    "offload": "--mmproj-offload" in sys.argv,
    "no_offload": "--no-mmproj-offload" in sys.argv,
}
with open(os.path.join(os.getcwd(), "starts.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, ensure_ascii=False) + "\\n")


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
        elif self.path.rstrip("/") == "/props":
            self.send_json(200, {"model_path": model, "alias": alias})
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
            text = "pong (image)" if image else "pong"
            self.send_json(200, {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": alias,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
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
''',
    encoding="utf-8",
)
fake_binary.write_text(
    f"#!/bin/sh\nexec {sys.executable} {fake_server} \"$@\"\n", encoding="utf-8"
)
fake_binary.chmod(0o755)

# --- model files ----------------------------------------------------------
models_root = BASE / "models"
vision_model_dir = models_root / "Qwen3.8-27B-UD"
vision_model_dir.mkdir(parents=True)
main_model = vision_model_dir / "Qwen3.8-27B-UD-Q5_K_M.gguf"
write_gguf(main_model, [
    ("general.architecture", 8, "qwen35"),
    ("general.name", 8, "Qwen3.8 27B Ultra Uncensored Heretic"),
    ("qwen35.context_length", 4, 262144),
])
mmproj = vision_model_dir / "mmproj-F16.gguf"
write_gguf(mmproj, [
    ("clip.has_vision_encoder", 7, True),
    ("clip.has_audio_encoder", 7, False),
])
text_model_dir = models_root / "Qwen3.8-27B-UD-2"
text_model_dir.mkdir()
text_model = text_model_dir / "Qwen3.8-27B-UD-2.gguf"
write_gguf(text_model, [
    ("general.architecture", 8, "qwen35"),
    ("general.name", 8, "Qwen3.8 27B Text Only"),
    ("qwen35.context_length", 4, 131072),
])

# --- gateway --------------------------------------------------------------
port = free_port()
engine_port = free_port()
config = {
    "listen": {"host": "127.0.0.1", "port": port},
    "state_file": "state.json",
    "pid_file": "router.pid",
    "log_dir": "logs",
    "log_file": "logs/router.log",
    "model_params_file": "model-params.json",
    "smoke_image": "smoke.png",
    "webui": {
        "enabled": True,
        "root": str(ROOT / "webui"),
        "console": "index.html",
        "ui_dir": "llama-ui",
        "ui_path": "/ui",
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


def stop_router() -> None:
    if router.poll() is None:
        router.send_signal(signal.SIGTERM)
        try:
            router.wait(timeout=20)
        except subprocess.TimeoutExpired:
            router.kill()
            router.wait(timeout=5)
    kill_fakes()


def kill_fakes() -> None:
    try:
        found = subprocess.run(
            ["pgrep", "-f", str(fake_server)], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return
    for token in found.split():
        try:
            os.kill(int(token), signal.SIGTERM)
        except (OSError, ValueError):
            pass


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


checks = 0
wait_api()
try:
    # 1. Create the fixed kvmem_llama slot through the management API.
    config_dir = BASE / "models-kvmem-llama"
    template = config_dir / "kvmem-llama.template.json"
    created = requests.post(url + "/router/engines/paths", json={
        "engine": "kvmem_llama",
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
    assert "--kvmem-budget" in template_doc["args"], template_doc
    assert template_doc["args"][template_doc["args"].index("--kvmem-budget") + 1] == "32768"
    assert template_doc["args"][template_doc["args"].index("--kv-dtype") + 1] == "q8_0"
    paths = requests.get(url + "/router/engines/paths", timeout=10).json()
    slot = next((item for item in paths.get("engines") or [] if item.get("engine") == "kvmem_llama"), None)
    assert slot is not None and slot.get("configured") is True, paths
    assert status()["engines"]["kvmem_llama"]["kind"] == "kvmem_llama"
    checks += 1

    # 2. Add the model directory: GGUF + mmproj auto-picked, 32k budget, q8.
    job = add_model(
        engine="kvmem_llama",
        model=str(vision_model_dir),
        kvmem_budget=32768,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        smoke=True,
    )
    assert job["status"] == "ok", job
    steps = {step["key"]: step for step in job["steps"]}
    assert all(step["status"] == "ok" for step in job["steps"]), job["steps"]
    assert "mmproj-F16.gguf" in steps["paths"]["detail"], steps["paths"]
    assert "262,144" in steps["metadata"]["detail"], steps["metadata"]
    assert "预算 32,768" in steps["kv"]["detail"], steps["kv"]
    assert "q8_0/q8_0" in steps["kv"]["detail"], steps["kv"]
    assert "视觉" in steps["vision_smoke"]["detail"] or steps["vision_smoke"]["status"] == "ok"
    checks += 1

    model_id = "Qwen3.8-27B-UD"
    item = models()[model_id]
    assert item["engine"] == "kvmem_llama", item
    assert item["source"] == "kvmem_llama", item
    assert item["context_length"] == 262144, item
    assert item["cache_type_k"] == "q8_0" and item["cache_type_v"] == "q8_0", item
    assert item["multimodal"] is True and item["vision_gpu"] is False, item
    assert item["engine_params"]["kvmem_budget"] == 32768, item
    assert item["engine_params"]["kvmem_block_tokens"] == 32, item
    assert item["engine_params"]["kvmem_gen_reserve"] == 1024, item
    checks += 1

    model_cfg_path = config_dir / "kvmem-llama-qwen3.8-27b-ud.json"
    assert model_cfg_path.is_file(), list(config_dir.iterdir())
    model_cfg = read_json(model_cfg_path)
    args = [str(value) for value in model_cfg["args"]]
    assert model_cfg["model"] == str(main_model), model_cfg
    assert model_cfg["vision"]["mmproj"] == str(mmproj), model_cfg
    assert model_cfg["vision"]["gpu"] is False, model_cfg
    assert "-m" not in args and "-a" not in args and "--host" not in args and "--port" not in args, args
    assert args[args.index("-c") + 1] == "262144", args
    assert args[args.index("--kvmem-budget") + 1] == "32768", args
    assert args[args.index("--kvmem-block-tokens") + 1] == "32", args
    assert args[args.index("--kvmem-gen-reserve") + 1] == "1024", args
    assert args[args.index("--kv-dtype") + 1] == "q8_0", args
    assert "--kvmem" in args and "--no-ui" in args, args
    checks += 1

    first = starts()[0]
    assert first["model"] == str(main_model), first
    assert first["alias"] == model_id, first
    assert first["mmproj"] == str(mmproj), first
    assert first["no_offload"] is True and first["offload"] is False, first
    assert "--kv-dtype" in first["argv"] and "q8_0" in first["argv"], first
    assert status()["active_model"] == model_id, status()
    checks += 1

    # 2b. Advanced KVMem engine params: write, read back, restart and clear.
    advanced = {
        "kvmem_block_tokens": 64,
        "kvmem_gen_reserve": 2048,
        "kvmem_recent_tokens": 512,
        "kvmem_sink_tokens": 256,
        "kvmem_method": "recency",
        "kvmem_query_scorer": "meanq-cosine",
        "kvmem_score_agg": "max",
        "kvmem_query_last": 128,
        "kvmem_query_max_tokens": 256,
        "kvmem_query_replay_max_tokens": 1024,
        "kvmem_query_replay": "legacy",
        "kvmem_graph_alpha": 0.1,
        "kvmem_lrfu": True,
        "kvmem_lrfu_lambda": 0.25,
        "kvmem_lrfu_top_frac": 0.75,
        "kvmem_gpu_ratio": 0.4,
        "kvmem_host_mib": 4096,
        "kvmem_sessions": 2,
        "parallel": 2,
        "flash_attn": "auto",
        "timeout": 600,
        "verbosity": 2,
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 5,
        "kvmem_mtp_state": "snapshots",
        "enable_thinking": True,
        "reasoning_budget": 2048,
        "image_max_tokens": 4096,
        "video_fps": 1.5,
    }
    saved = requests.post(url + "/router/params", json={
        "model": model_id, "params": {}, "engine_params": advanced,
    }, timeout=20)
    assert saved.status_code == 200, saved.text
    saved_body = saved.json()
    assert saved_body["engine_params"]["kvmem_method"] == "recency", saved_body
    assert saved_body["engine_params"]["parallel"] == 2, saved_body
    wait_idle()
    advanced_args = [str(value) for value in read_json(model_cfg_path)["args"]]
    numeric = {
        "--kvmem-block-tokens": "64",
        "--kvmem-gen-reserve": "2048",
        "--kvmem-recent-tokens": "512",
        "--kvmem-sink-tokens": "256",
        "--kvmem-method": "recency",
        "--kvmem-query-scorer": "meanq-cosine",
        "--kvmem-score-agg": "max",
        "--kvmem-query-last": "128",
        "--kvmem-query-max-tokens": "256",
        "--kvmem-query-replay-max-tokens": "1024",
        "--kvmem-query-replay": "legacy",
        "--kvmem-graph-alpha": "0.1",
        "--kvmem-lrfu-lambda": "0.25",
        "--kvmem-lrfu-top-frac": "0.75",
        "--kvmem-gpu-ratio": "0.4",
        "--kvmem-host-mib": "4096",
        "--kvmem-sessions": "2",
        "-np": "2",
        "-fa": "auto",
        "--timeout": "600",
        "-lv": "2",
        "--spec-type": "draft-mtp",
        "--spec-draft-n-max": "5",
        "--kvmem-mtp-state": "snapshots",
        "--reasoning-budget": "2048",
        "--image-max-tokens": "4096",
        "--video-fps": "1.5",
    }
    for flag, value in numeric.items():
        assert flag in advanced_args and advanced_args[advanced_args.index(flag) + 1] == value, (flag, advanced_args)
    assert "--kvmem-lrfu" in advanced_args and "--no-kvmem-lrfu" not in advanced_args, advanced_args
    assert "--enable-thinking" in advanced_args and "--no-think" not in advanced_args, advanced_args
    last_start = starts()[-1]
    assert "-np" in last_start["argv"] and "2" in last_start["argv"], last_start
    assert "--enable-thinking" in last_start["argv"], last_start

    cleared = requests.post(url + "/router/params", json={
        "model": model_id,
        "params": {},
        "engine_params": {
            "kvmem_sessions": None,
            "kvmem_lrfu": None,
            "enable_thinking": None,
            "parallel": None,
        },
    }, timeout=20)
    assert cleared.status_code == 200, cleared.text
    cleared_values = cleared.json()["engine_params"]
    assert "kvmem_sessions" not in cleared_values and "kvmem_lrfu" not in cleared_values, cleared_values
    wait_idle()
    cleared_args = [str(value) for value in read_json(model_cfg_path)["args"]]
    assert "--kvmem-sessions" not in cleared_args, cleared_args
    assert "--kvmem-lrfu" not in cleared_args and "--no-kvmem-lrfu" not in cleared_args, cleared_args
    assert "--enable-thinking" not in cleared_args and "--no-think" not in cleared_args, cleared_args
    assert "-np" not in cleared_args, cleared_args
    for bad, needle in (
        ({"parallel": 9}, "expected 1..8"),
        ({"kvmem_disk_mib": 1024}, "kvmem-disk-path"),
        ({"image_min_tokens": 4096, "image_max_tokens": 2048}, "不能小于"),
        ({"bogus": 1}, "unknown engine parameter"),
    ):
        rejected = requests.post(url + "/router/params", json={
            "model": model_id, "params": {}, "engine_params": bad,
        }, timeout=15)
        assert rejected.status_code == 400 and needle in rejected.text, (bad, rejected.text)
    checks += 1

    # 3. Vision CPU -> GPU rewrites the config and the next start's flags.
    stopped = requests.post(url + "/router/stop", json={}, timeout=20)
    assert stopped.status_code == 200, stopped.text
    wait_idle()
    saved = requests.post(url + "/router/params", json={
        "model": model_id, "params": {}, "vision_device": "gpu",
    }, timeout=15)
    assert saved.status_code == 200 and saved.json()["vision_gpu"] is True, saved.text
    assert read_json(model_cfg_path)["vision"]["gpu"] is True
    assert models()[model_id]["vision_gpu"] is True
    loaded = requests.post(url + "/models/load", json={"model": model_id}, timeout=20)
    assert loaded.status_code in (200, 202), loaded.text
    wait_idle()
    gpu_start = starts()[-1]
    assert gpu_start["offload"] is True and gpu_start["no_offload"] is False, gpu_start
    checks += 1

    saved = requests.post(url + "/router/params", json={
        "model": model_id, "params": {}, "vision_device": "cpu",
    }, timeout=15)
    assert saved.status_code == 200 and saved.json()["vision_gpu"] is False, saved.text
    wait_idle()
    cpu_start = starts()[-1]
    assert cpu_start["no_offload"] is True and cpu_start["offload"] is False, cpu_start
    assert models()[model_id]["vision_gpu"] is False
    rejected = requests.post(url + "/router/params", json={
        "model": model_id, "params": {"repeat_last_n": 64},
    }, timeout=15)
    assert rejected.status_code == 400 and "repeat_last_n" in rejected.text, rejected.text
    assert models()[model_id]["params"].get("repeat_last_n") is None
    checks += 1

    # 4. Validation failures must not write a config or register a model.
    before = set(models())
    bad_budget = add_model(
        engine="kvmem_llama", model=str(vision_model_dir),
        name="budget-too-big", kvmem_budget=262144,
        cache_type_k="q8_0", cache_type_v="q8_0", smoke=False,
    )
    assert bad_budget["status"] == "failed", bad_budget
    assert "超过上下文" in bad_budget["error"], bad_budget
    bad_cache = add_model(
        engine="kvmem_llama", model=str(vision_model_dir),
        name="bad-cache", cache_type_k="f16", cache_type_v="q8_0", smoke=False,
    )
    assert bad_cache["status"] == "failed", bad_cache
    assert "不受支持" in bad_cache["error"], bad_cache
    bad_combo = add_model(
        engine="kvmem_llama", model=str(vision_model_dir),
        name="q5-v", cache_type_k="q5_0", cache_type_v="q5_0", smoke=False,
    )
    # q5_0/q5_0 is valid and shares --kv-dtype; clean it up below instead.
    assert bad_combo["status"] == "ok", bad_combo
    unknown = requests.post(url + "/router/models/add", json={
        "engine": "kvmem_llama", "model": str(vision_model_dir),
        "name": "unknown-engine-param", "smoke": False,
        "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "engine_params": {"bogus": 1},
    }, timeout=15)
    assert unknown.status_code == 400 and "unknown engine parameter" in unknown.text, unknown.text
    bad_disk = add_model(
        engine="kvmem_llama", model=str(vision_model_dir),
        name="disk-no-path", cache_type_k="q8_0", cache_type_v="q8_0",
        engine_params={"kvmem_disk_mib": 512}, smoke=False,
    )
    assert bad_disk["status"] == "failed", bad_disk
    assert "kvmem-disk-path" in bad_disk["error"], bad_disk
    bad_image = requests.post(url + "/router/models/add", json={
        "engine": "kvmem_llama", "model": str(vision_model_dir),
        "name": "image-range", "smoke": False,
        "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "engine_params": {"image_min_tokens": 8192, "image_max_tokens": 4096},
    }, timeout=15)
    assert bad_image.status_code == 400 and "不能小于" in bad_image.text, bad_image.text
    deleted = requests.post(url + "/router/models/delete", json={"model": "q5-v"}, timeout=30)
    assert deleted.status_code == 200, deleted.text
    after = set(models())
    assert before == after, (before, after)
    assert not list(config_dir.glob("kvmem-llama-budget-too-big.json"))
    assert not list(config_dir.glob("kvmem-llama-bad-cache.json"))
    assert not list(config_dir.glob("kvmem-llama-disk-no-path.json"))
    assert not list(config_dir.glob("kvmem-llama-image-range.json"))
    checks += 1

    # 5. The console renders the fourth engine and its dialogs.
    assert requests.get(url + "/", timeout=10).status_code == 200
except BaseException:
    stop_router()
    raise

# --- browser UI -----------------------------------------------------------
driver_port = free_port()
driver = subprocess.Popen(
    ["/snap/bin/geckodriver", "--port", str(driver_port)],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
driver_base = f"http://127.0.0.1:{driver_port}"
for _ in range(100):
    try:
        requests.get(driver_base + "/status", timeout=1)
        break
    except requests.RequestException:
        time.sleep(0.1)
else:
    stop_router()
    driver.terminate()
    raise SystemExit("geckodriver did not start")
session = requests.post(driver_base + "/session", json={"capabilities": {"alwaysMatch": {
    "browserName": "firefox",
    "moz:firefoxOptions": {"args": ["-headless", "--width=1400", "--height=1000"]},
}}}, timeout=60).json()["value"]["sessionId"]


def wd(method: str, path: str, payload=None, timeout: int = 90):
    response = requests.request(method, f"{driver_base}/session/{session}{path}",
                                json=payload, timeout=timeout)
    data = response.json()
    value = data.get("value")
    if isinstance(value, dict) and value.get("error"):
        raise RuntimeError(value)
    return value


def js(script: str):
    return wd("POST", "/execute/sync", {"script": script, "args": []})


def wait_js(script: str, timeout: float = 30, message: str = ""):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = js(script)
            if last:
                return last
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.2)
    raise SystemExit(f"wait failed: {message or script}: {last}")


def shot(name: str) -> None:
    data = wd("GET", "/screenshot")
    (OUT / name).write_bytes(base64.b64decode(data))


try:
    wd("POST", "/url", {"url": url + "/"})
    wait_js("return document.querySelectorAll('#models .model').length >= 1;", message="model cards")
    model_json = json.dumps(model_id)
    js(
        "const card=[...document.querySelectorAll('#models .model')]"
        ".find((el)=>el.querySelector('.id')&&el.querySelector('.id').textContent.trim()==="
        + model_json + ");"
        "if(!card) return false;"
        "card.querySelector('[data-action=params]').click(); return true;"
    )
    wait_js("return document.querySelector('#params-dialog').open;", message="params dialog")
    surface = js(
        "const tab=document.querySelector('#params-kvmem-tab');"
        "const host=document.querySelector('#params-kvmem');"
        "return {"
        "hint: document.querySelector('#params-hint').textContent,"
        "repeat: document.querySelector('[name=repeat_last_n]').disabled,"
        "visionHidden: document.querySelector('#params-vision').hidden,"
        "visionDisabled: document.querySelector('#params-vision-device').classList.contains('disabled'),"
        "engineHidden: document.querySelector('#params-engine').hidden,"
        "budgetHidden: document.querySelector('#params-budget-field').hidden,"
        "budget: document.querySelector('#params-budget').value,"
        "contextHidden: document.querySelector('#params-context-field').hidden,"
        "cacheHidden: document.querySelector('#params-cache-k-field').hidden,"
        "kvmemTabHidden: !tab || tab.hidden,"
        "kvmemCount: document.querySelector('#params-kvmem-count').textContent,"
        "kvmemFields: host.querySelectorAll('[data-param-key]').length,"
        "method: host.querySelector('[data-param-key=kvmem_method]').value,"
        "recent: host.querySelector('[data-param-key=kvmem_recent_tokens]').value,"
        "sessions: host.querySelector('[data-param-key=kvmem_sessions]').value,"
        "hostMib: host.querySelector('[data-param-key=kvmem_host_mib]').value,"
        "gpu: document.querySelector('input[name=params-vision-device][value=gpu]').checked,"
        "};"
    )
    assert "KVMem + llama.cpp" in surface["hint"], surface
    assert surface["repeat"] is True, surface
    assert surface["visionHidden"] is False and surface["visionDisabled"] is False, surface
    assert surface["engineHidden"] is False, surface
    assert surface["budgetHidden"] is False and surface["budget"] == "32768", surface
    assert surface["contextHidden"] is True and surface["cacheHidden"] is True, surface
    assert surface["kvmemTabHidden"] is False, surface
    assert int(surface["kvmemCount"] or 0) >= 40, surface
    assert surface["kvmemFields"] == int(surface["kvmemCount"]), surface
    assert surface["method"] == "recency", surface
    assert surface["recent"] == "512", surface
    assert surface["sessions"] == "", surface
    assert surface["hostMib"] == "4096", surface
    assert surface["gpu"] is False, surface
    shot("params-kvmem-llama.png")
    js("document.querySelector('#params-kvmem-tab').click();")
    wait_js(
        "return !document.querySelector('[data-param-panel=kvmem]').hidden;",
        message="kvmem params tab",
    )
    shot("params-kvmem-engine.png")
    js("document.querySelector('#params-cancel').click();")

    js("document.querySelector('#btn-add-model').click();")
    wait_js("return document.querySelector('#add-dialog').open;", message="add dialog")
    form = js(
        "return {"
        "engine: document.querySelector('input[name=add-engine]:checked').value,"
        "budgetHidden: document.querySelector('#add-budget-field').hidden,"
        "budget: document.querySelector('#add-budget').value,"
        "context: document.querySelector('#add-context').value,"
        "cacheK: [...document.querySelector('#add-cache-k').options].map(o=>o.value),"
        "cacheV: document.querySelector('#add-cache-v').value,"
        "repeat: document.querySelector('[name=repeat_last_n]').disabled,"
        "visionHidden: document.querySelector('#add-vision-field').hidden,"
        "kvmemTabHidden: document.querySelector('#add-kvmem-tab').hidden,"
        "kvmemCount: document.querySelector('#add-kvmem-count').textContent,"
        "kvmemFields: document.querySelectorAll('#add-kvmem [data-param-key]').length,"
        "kvmemBlank: [...document.querySelectorAll('#add-kvmem [data-param-key]')]"
        ".every((el)=>el.value===''),"
        "hint: document.querySelector('#add-model-hint').textContent,"
        "caption: document.querySelector('#add-engine-caption').textContent,"
        "defaults: document.querySelector('#add-defaults').textContent,"
        "};"
    )
    assert form["engine"] == "kvmem_llama", form
    assert form["budgetHidden"] is False and form["budget"] == "32768", form
    assert form["context"] == "", form
    assert form["cacheK"] == ["q8_0", "f16", "f32", "q5_0", "q4_0"], form
    assert form["cacheV"] == "q8_0", form
    assert form["repeat"] is True, form
    assert form["visionHidden"] is False, form
    assert form["kvmemTabHidden"] is False, form
    assert int(form["kvmemCount"] or 0) >= 40, form
    assert form["kvmemFields"] == int(form["kvmemCount"]), form
    assert form["kvmemBlank"] is True, form
    assert "目录" in form["hint"], form
    assert "32,768" in form["defaults"] or "32768" in form["defaults"], form
    shot("add-kvmem-llama.png")

    # Add the text-only directory through the console form with one advanced
    # KVMem option so the UI -> job -> config path is covered end to end.
    js(
        "const host=document.querySelector('#add-kvmem');"
        "host.querySelector('[data-param-key=kvmem_recent_tokens]').value='384';"
        "const input=document.querySelector('#add-model');"
        "input.value=" + json.dumps(str(text_model_dir)) + ";"
        "input.dispatchEvent(new Event('input',{bubbles:true}));"
        "document.querySelector('#add-submit').click();"
    )
    wait_js(
        "const s=document.querySelector('#add-submit');"
        "return s.dataset.done==='1'||s.textContent==='完成';",
        timeout=120, message="console add finished",
    )
    shot("add-kvmem-llama-done.png")
    added = models()["Qwen3.8-27B-UD-2"]
    assert added["engine"] == "kvmem_llama" and added["context_length"] == 131072, added
    assert added["cache_type_k"] == "q8_0" and added["multimodal"] is False, added
    ui_cfg = read_json(config_dir / "kvmem-llama-qwen3.8-27b-ud-2.json")
    ui_args = [str(value) for value in ui_cfg["args"]]
    assert ui_args[ui_args.index("--kvmem-budget") + 1] == "32768", ui_args
    assert ui_args[ui_args.index("--kvmem-recent-tokens") + 1] == "384", ui_args
    assert added["engine_params"]["kvmem_recent_tokens"] == 384, added
    wait_js(
        "return [...document.querySelectorAll('#models .model .id')]"
        ".some((el)=>el.textContent.trim()==='Qwen3.8-27B-UD-2');",
        message="second model card",
    )

    # Switching back restarts the single-model engine with the right artifact.
    loaded = requests.post(url + "/models/load", json={"model": model_id}, timeout=20)
    assert loaded.status_code in (200, 202), loaded.text
    wait_idle()
    switched = starts()[-1]
    assert switched["alias"] == model_id and switched["model"] == str(main_model), switched
    assert switched["mmproj"] == str(mmproj), switched
    unloaded = requests.post(url + "/models/unload", json={"model": model_id}, timeout=20)
    assert unloaded.status_code in (200, 202), unloaded.text
    wait_idle()
    stopped_status = status()
    assert stopped_status["engines"]["kvmem_llama"]["alive"] is False, stopped_status
    assert stopped_status["active_model"] is None, stopped_status
    checks += 1
    print(f"kvmem-llama e2e: {checks} check groups passed; shots in {OUT}")
finally:
    try:
        wd("DELETE", "")
    except Exception:  # noqa: BLE001
        pass
    try:
        driver.terminate()
    except OSError:
        pass
    finally:
        stop_router()
