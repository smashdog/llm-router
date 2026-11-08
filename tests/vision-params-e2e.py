#!/usr/bin/env python3
"""End-to-end test for editing a model's vision CPU/GPU device.

Starts a real gateway with fake llama.cpp and KVMem engines, checks the
/router/params API and the console dialog, and verifies the engine configs.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tmp" / "vision-params-e2e"
OUT = BASE / "shots"
if BASE.exists():
    shutil.rmtree(BASE)
BASE.mkdir(parents=True)
OUT.mkdir()
shutil.copy2(ROOT / "router.py", BASE / "router.py")


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# --- fake llama.cpp -------------------------------------------------------
llama_cwd = BASE / "llama.cpp"
llama_binary = llama_cwd / "build" / "bin" / "llama-server"
llama_binary.parent.mkdir(parents=True)
llama_binary.write_text('#!/bin/sh\necho "version: 9999 (fake llama-server)"\n')
llama_binary.chmod(0o755)
models_dir = BASE / "models"
vision_dir = models_dir / "vision-model"
vision_dir.mkdir(parents=True)
(vision_dir / "vision-model.gguf").write_bytes(b"GGUF")
(vision_dir / "mmproj.gguf").write_bytes(b"GGUF")
text_dir = models_dir / "text-model"
text_dir.mkdir()
(text_dir / "text-model.gguf").write_bytes(b"GGUF")
preset = BASE / "models.ini"
preset.write_text(
    "version = 1\n\n"
    "[*]\n"
    "cache-type-k = q8_0\n"
    "cache-type-v = q8_0\n\n"
    "[vision-model]\n"
    "c = 4096\n"
    "model = vision-model/vision-model.gguf\n"
    "mmproj = vision-model/mmproj.gguf\n"
    "mmproj-offload = off\n\n"
    "[text-model]\n"
    "c = 4096\n"
    "model = text-model/text-model.gguf\n",
    encoding="utf-8",
)

# --- fake KVMem -----------------------------------------------------------
kv_cwd = BASE / "kvmem"
(kv_cwd / "serve").mkdir(parents=True)
(kv_cwd / "serve" / "__init__.py").write_text("", encoding="utf-8")
(kv_cwd / "serve" / "server.py").write_text("", encoding="utf-8")
strata_exe = kv_cwd / "build" / "strata"
strata_exe.parent.mkdir()
strata_exe.write_text("#!/bin/sh\nexit 0\n")
strata_exe.chmod(0o755)
vision_exe = kv_cwd / "build" / "strata-vision"
vision_exe.write_text("#!/bin/sh\nexit 0\n")
vision_exe.chmod(0o755)
kv_model = kv_cwd / "model.gguf"
kv_model.write_bytes(b"GGUF")
kv_mmproj = kv_cwd / "mmproj.gguf"
kv_mmproj.write_bytes(b"GGUF")
tokenizer = kv_cwd / "tokenizer"
tokenizer.mkdir()
(tokenizer / "vocab.json").write_text("{}", encoding="utf-8")
kv_models = kv_cwd / "models"
kv_models.mkdir()
kv_config = kv_models / "strata-kv-vision.json"
kv_config.write_text(json.dumps({
    "model_name": "kv-vision",
    "display_name": "KV Vision",
    "exe": str(strata_exe),
    "args": ["--model", str(kv_model), "--max-context", "4096"],
    "tokenizer": str(tokenizer),
    "vision": {
        "exe": str(vision_exe),
        "mmproj": str(kv_mmproj),
        "model": str(kv_model),
        "gpu": False,
        "threads": 2,
        "max_tokens": 1024,
    },
}, ensure_ascii=False, indent=2), encoding="utf-8")
# A text-only polystrata model: the vision_* engine parameters must be hidden
# in the console and rejected by the API.
kv_text_config = kv_models / "strata-zz-kv-text.json"
kv_text_config.write_text(json.dumps({
    "model_name": "kv-text",
    "display_name": "KV Text",
    "exe": str(strata_exe),
    "args": ["--model", str(kv_model), "--max-context", "4096"],
    "tokenizer": str(tokenizer),
}, ensure_ascii=False, indent=2), encoding="utf-8")

# --- fake NInfer ----------------------------------------------------------
ninfer_cwd = BASE / "ninfer"
ninfer_cwd.mkdir()
ninfer_binary = ninfer_cwd / "ninfer-serve"
ninfer_binary.write_text("#!/bin/sh\nexit 0\n")
ninfer_binary.chmod(0o755)
ninfer_model = ninfer_cwd / "model.ninfer"
ninfer_model.write_bytes(b"ninfer")
ninfer_models = ninfer_cwd / "models"
ninfer_models.mkdir()
(ninfer_models / "ninfer-vision.json").write_text(json.dumps({
    "model_name": "ninfer-vision",
    "display_name": "NInfer Vision",
    "exe": str(ninfer_binary),
    "model": str(ninfer_model),
    "args": [],
    "vision": {"mmproj": str(kv_mmproj)},
}, ensure_ascii=False, indent=2), encoding="utf-8")

port = free_port()
config = {
    "listen": {"host": "127.0.0.1", "port": port},
    "state_file": "state.json",
    "pid_file": "router.pid",
    "log_dir": "logs",
    "log_file": "logs/router.log",
    "model_params_file": "model-params.json",
    "console": {
        "enabled": True,
        "root": str(ROOT / "console"),
        "file": "index.html",
    },
    "engines": {
        "llama": {
            "kind": "llama_router",
            "host": "127.0.0.1",
            "port": free_port(),
            "cwd": str(llama_cwd),
            "command": [
                str(llama_binary), "--models-preset", str(preset),
                "--models-dir", str(models_dir),
                "--host", "{host}", "--port", "{port}",
            ],
            "start_timeout_s": 30,
            "log_path": "logs/llama-engine.log",
            "models": {
                "kind": "llama_dir",
                "dir": str(models_dir),
                "preset": str(preset),
            },
        },
        "kvmem": {
            "kind": "kvmem",
            "host": "127.0.0.1",
            "port": free_port(),
            "cwd": str(kv_cwd),
            "command": [
                sys.executable, "-m", "serve.server", "--engine", "strata",
                "--config", "{config}", "--host", "{host}", "--port", "{port}",
            ],
            "start_timeout_s": 30,
            "log_path": "logs/kvmem-engine.log",
            "models": {"kind": "kvmem_dir", "dir": str(kv_models)},
        },
        "ninfer": {
            "kind": "ninfer",
            "host": "127.0.0.1",
            "port": free_port(),
            "cwd": str(ninfer_cwd),
            "command": [str(ninfer_binary)],
            "start_timeout_s": 30,
            "log_path": "logs/ninfer-engine.log",
            "models": {"kind": "ninfer_dir", "dir": str(ninfer_models)},
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
            router.wait(timeout=15)
        except subprocess.TimeoutExpired:
            router.kill()
            router.wait(timeout=5)


def wait_api() -> None:
    deadline = time.time() + 20
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


def save(model: str, **extra) -> requests.Response:
    payload = {"model": model, "params": {}}
    payload.update(extra)
    return requests.post(url + "/router/params", json=payload, timeout=10)


def preset_text() -> str:
    return preset.read_text(encoding="utf-8")


def kv_gpu() -> bool:
    return json.loads(kv_config.read_text(encoding="utf-8"))["vision"]["gpu"]


wait_api()
checks = 0
try:
    items = models()
    assert items["vision-model"]["multimodal"] is True, items.get("vision-model")
    assert items["vision-model"]["vision_gpu"] is False, items.get("vision-model")
    assert items["vision-model"]["source"] == "preset", items.get("vision-model")
    assert items["text-model"]["multimodal"] is False, items.get("text-model")
    assert items["kv-vision"]["multimodal"] is True, items.get("kv-vision")
    assert items["kv-vision"]["vision_gpu"] is False, items.get("kv-vision")
    assert items["kv-vision"]["engine_params"]["vision_threads"] == 2, items["kv-vision"]
    assert items["kv-vision"]["engine_params"]["vision_max_tokens"] == 1024, items["kv-vision"]
    assert items["kv-text"]["multimodal"] is False, items.get("kv-text")
    assert "vision_max_tokens" not in (items["kv-text"].get("engine_params") or {}), items["kv-text"]
    assert items["ninfer-vision"]["multimodal"] is True, items.get("ninfer-vision")
    checks += 1

    saved = save("vision-model", vision_device="gpu")
    body = saved.json()
    assert saved.status_code == 200, body
    assert body["vision_gpu"] is True, body
    assert models()["vision-model"]["vision_gpu"] is True
    assert "mmproj-offload = on" in preset_text(), preset_text()
    checks += 1

    saved = save("vision-model", vision_device="cpu")
    body = saved.json()
    assert saved.status_code == 200, body
    assert body["vision_gpu"] is False, body
    assert models()["vision-model"]["vision_gpu"] is False
    assert "mmproj-offload = off" in preset_text(), preset_text()
    checks += 1

    bad = save("vision-model", vision_device="npu")
    assert bad.status_code == 400, bad.text
    bad = save("text-model", vision_device="gpu")
    assert bad.status_code == 400 and "没有视觉编码器" in bad.text, bad.text
    bad = save("vision-model", vision_device=True)
    assert bad.status_code == 400, bad.text
    checks += 1

    saved = save("kv-vision", vision_device="gpu")
    body = saved.json()
    assert saved.status_code == 200, body
    assert body["vision_gpu"] is True, body
    assert models()["kv-vision"]["vision_gpu"] is True
    assert kv_gpu() is True
    doc = json.loads(kv_config.read_text(encoding="utf-8"))
    assert doc["vision"]["threads"] == 2 and doc["vision"]["max_tokens"] == 1024, doc
    checks += 1

    saved = save("kv-vision", vision_device="cpu")
    assert saved.status_code == 200 and saved.json()["vision_gpu"] is False, saved.text
    assert kv_gpu() is False
    checks += 1

    # NInfer carries the vision tower inside the .ninfer file: the device is
    # written as --vision/--vision-residency (GPU) or --vision-cpu (CPU).
    ninfer_config = ninfer_models / "ninfer-vision.json"

    def ninfer_args() -> list:
        return json.loads(ninfer_config.read_text(encoding="utf-8"))["args"]

    saved = save("ninfer-vision", vision_device="gpu")
    body = saved.json()
    assert saved.status_code == 200, body
    assert body["vision_gpu"] is True, body
    args = ninfer_args()
    assert "--vision" in args and "--vision-cpu" not in args, args
    assert args[args.index("--vision-residency") + 1] == "resident", args
    assert json.loads(ninfer_config.read_text(encoding="utf-8"))["vision"]["gpu"] is True
    assert models()["ninfer-vision"]["vision_gpu"] is True
    checks += 1

    saved = save("ninfer-vision", vision_device="cpu")
    body = saved.json()
    assert saved.status_code == 200 and body["vision_gpu"] is False, body
    args = ninfer_args()
    assert args == ["--vision-cpu"], args
    assert models()["ninfer-vision"]["vision_gpu"] is False
    checks += 1

    # NInfer engine parameters round-trip through the same endpoint.
    saved = save("ninfer-vision", engine_params={
        "max_context": 16384, "prefill_chunk": 2048, "kv_dtype": "int8",
        "vision_max_merged": 1024,
    })
    body = saved.json()
    assert saved.status_code == 200, body
    params = body["engine_params"]
    assert params["max_context"] == 16384, params
    assert params["prefill_chunk"] == 2048, params
    assert params["kv_dtype"] == "int8", params
    args = ninfer_args()
    for pair in (["--max-context", "16384"], ["--prefill-chunk", "2048"],
                 ["--kv-dtype", "int8"], ["--vision-max-merged", "1024"]):
        index = args.index(pair[0])
        assert args[index + 1] == pair[1], args
    assert "--vision-cpu" in args, args
    checks += 1

    # NInfer's KVMem extension flags round-trip separately.
    saved = save("ninfer-vision", engine_params={
        "kvmem": True, "kvmem_budget": 8192, "kvmem_gen_reserve": 2048,
        "kvmem_sessions": 2, "kvmem_lrfu": False,
    })
    body = saved.json()
    assert saved.status_code == 200, body
    params = body["engine_params"]
    assert params["kvmem"] is True and params["kvmem_budget"] == 8192, params
    assert params["kvmem_sessions"] == 2 and params["kvmem_lrfu"] is False, params
    args = ninfer_args()
    assert args[args.index("--kvmem-budget") + 1] == "8192", args
    assert "--kvmem" in args and "--no-kvmem-lrfu" in args, args
    saved = save("ninfer-vision", engine_params={"kvmem": False, "kvmem_lrfu": True})
    body = saved.json()
    assert saved.status_code == 200, body
    args = ninfer_args()
    assert "--no-kvmem" in args and "--kvmem-lrfu" in args, args
    assert "--kvmem" not in args, args
    checks += 1

    bad = save("ninfer-vision", engine_params={"prefill_chunk": 1000})
    assert bad.status_code == 400 and "128" in bad.text, bad.text
    bad = save("ninfer-vision", engine_params={"bogus": 1})
    assert bad.status_code == 400, bad.text
    bad = save("ninfer-vision", engine_params={"kvmem_disk_path": "/tmp/kv"})
    assert bad.status_code == 400 and "kvmem_disk_mib" in bad.text, bad.text
    checks += 1

    # strata-poly: native --max-context lives beside the KVMem extension.
    saved = save("kv-vision", engine_params={"max_context": 16384})
    body = saved.json()
    assert saved.status_code == 200, body
    assert body["engine_params"]["max_context"] == 16384, body
    doc = json.loads(kv_config.read_text(encoding="utf-8"))
    args = doc["args"]
    assert args[args.index("--max-context") + 1] == "16384", args
    saved = save("kv-vision", engine_params={
        "kvmem_budget": 4096, "kvmem_gen_reserve": 2048, "kvmem_sessions": 1,
    })
    body = saved.json()
    assert saved.status_code == 200, body
    assert body["engine_params"]["kvmem_budget"] == 4096, body
    doc = json.loads(kv_config.read_text(encoding="utf-8"))
    args = doc["args"]
    assert args[args.index("--kvmem-budget") + 1] == "4096", args
    assert args[args.index("--max-context") + 1] == "16384", args
    assert args[args.index("--window") + 1] == "1", args
    assert "--no-draft" in args and args[args.index("--lookup") + 1] == "0", args
    bad = save("kv-vision", engine_params={"kvmem_budget": 100})
    assert bad.status_code == 400, bad.text
    checks += 1

    # PolyStrata's vision encoder settings live in the model's "vision"
    # object (strata-vision --threads/--max-tokens/--min-tokens).
    saved = save("kv-vision", engine_params={
        "vision_max_tokens": 2048, "vision_threads": 8, "vision_min_tokens": 512,
    })
    body = saved.json()
    assert saved.status_code == 200, body
    params = body["engine_params"]
    assert params["vision_max_tokens"] == 2048, params
    assert params["vision_threads"] == 8, params
    assert params["vision_min_tokens"] == 512, params
    doc = json.loads(kv_config.read_text(encoding="utf-8"))
    assert doc["vision"]["max_tokens"] == 2048, doc["vision"]
    assert doc["vision"]["threads"] == 8, doc["vision"]
    assert doc["vision"]["min_tokens"] == 512, doc["vision"]
    assert doc["vision"]["gpu"] is False and doc["vision"]["mmproj"], doc["vision"]
    assert "--vision-max-tokens" not in doc["args"], doc["args"]
    assert models()["kv-vision"]["engine_params"]["vision_max_tokens"] == 2048
    bad = save("kv-vision", engine_params={"vision_min_tokens": 4096})
    assert bad.status_code == 400 and "vision_min_tokens" in bad.text, bad.text
    bad = save("kv-vision", engine_params={"vision_max_tokens": 0})
    assert bad.status_code == 400, bad.text
    bad = save("kv-text", engine_params={"vision_max_tokens": 1024})
    assert bad.status_code == 400 and "没有 vision" in bad.text, bad.text
    checks += 1

    manage = requests.get(url + "/router/models/manage", timeout=10).json()
    ninfer_specs = {spec["key"]: spec for spec in manage["ninfer_engine_schema"]}
    assert {"max_context", "vision_max_merged", "kv_dtype", "spec"} <= set(ninfer_specs)
    assert ninfer_specs["max_context"]["owner"] == "engine"
    assert ninfer_specs["kvmem_budget"]["owner"] == "kvmem"
    poly_specs = {spec["key"]: spec for spec in manage["kvmem_poly_engine_schema"]}
    assert {"max_context", "window", "lookup", "no_draft", "kv", "layer_split"} <= set(poly_specs)
    assert {"vision_threads", "vision_max_tokens", "vision_min_tokens"} <= set(poly_specs)
    assert poly_specs["max_context"]["owner"] == "engine"
    assert poly_specs["vision_max_tokens"]["owner"] == "engine"
    assert poly_specs["vision_max_tokens"]["config_path"] == ["vision", "max_tokens"]
    assert poly_specs["vision_max_tokens"]["vision_only"] is True
    assert poly_specs["kvmem_budget"]["owner"] == "kvmem"
    kvllama_specs = {spec["key"]: spec for spec in manage["kvmem_engine_schema"]}
    assert kvllama_specs["n_predict"]["owner"] == "engine"
    assert kvllama_specs["kvmem_budget"]["owner"] == "kvmem"
    checks += 1

    # Sampling parameters still save in the same request as a vision change.
    saved = save("vision-model", params={"temperature": 0.6}, vision_device="gpu")
    body = saved.json()
    assert saved.status_code == 200 and body["params"]["temperature"] == 0.6, body
    assert models()["vision-model"]["vision_gpu"] is True
    assert body["params"]["temperature"] == 0.6, body
    checks += 1

    # Leave the llama model on CPU so the UI below can switch it to GPU.
    saved = save("vision-model", vision_device="cpu")
    assert saved.status_code == 200, saved.text
    assert models()["vision-model"]["vision_gpu"] is False
finally:
    pass

# --- browser UI -----------------------------------------------------------
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
    stop_router()
    driver.terminate()
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


def wait(script: str, timeout: float = 20, message: str = ""):
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


def shot(name: str) -> None:
    data = wd("GET", "/screenshot")
    (OUT / name).write_bytes(base64.b64decode(data))


def click(selector: str) -> None:
    js(f"""
      const el = document.querySelector({json.dumps(selector)});
      if (!el) throw new Error('missing ' + {json.dumps(selector)});
      el.click();
      return true;
    """)


try:
    wd("POST", "/window/rect", {"width": 1440, "height": 1200})
    wd("POST", "/url", {"url": url + "/"})
    wait("return document.querySelectorAll('#models .model').length >= 3", message="model cards")
    js("""
      window.__uiErrors = [];
      window.addEventListener('error', e => window.__uiErrors.push(String(e.message)));
      window.addEventListener('unhandledrejection',
        e => window.__uiErrors.push(String(e.reason && e.reason.message || e.reason)));
      return true;
    """)

    def open_model(model_id: str) -> None:
        js(f"""
          const card = [...document.querySelectorAll('#models .model')]
            .find(c => c.querySelector('.id').textContent.trim() === {json.dumps(model_id)});
          if (!card) throw new Error('missing model ' + {json.dumps(model_id)});
          card.querySelector('button[data-action="params"]').click();
          return true;
        """)
        wait("return document.getElementById('params-dialog').open", message="params dialog")

    open_model("vision-model")
    state = js("""
      const row = document.getElementById('params-vision');
      const cpu = document.querySelector('input[name="params-vision-device"][value="cpu"]');
      const gpu = document.querySelector('input[name="params-vision-device"][value="gpu"]');
      return {
        hidden: row.hidden,
        caption: document.getElementById('params-vision-caption').textContent,
        hint: document.getElementById('params-vision-hint').textContent,
        cpu: cpu.checked,
        gpu: gpu.checked,
        disabled: [...row.querySelectorAll('input')].map(i => i.disabled),
      };
    """)
    assert state["hidden"] is False, state
    assert state["cpu"] is True and state["gpu"] is False, state
    assert state["disabled"] == [False, False], state
    assert "mmproj" in state["caption"], state
    shot("vision-params-desktop.png")
    checks += 1

    click('input[name="params-vision-device"][value="gpu"]')
    click("#params-save")
    wait("return !document.getElementById('params-dialog').open", message="dialog closed")
    deadline = time.time() + 15
    while time.time() < deadline:
        if models()["vision-model"]["vision_gpu"] is True:
            break
        time.sleep(0.2)
    else:
        raise AssertionError("UI save did not switch vision to GPU")
    assert models()["vision-model"]["vision_gpu"] is True
    assert "mmproj-offload = on" in preset_text()
    checks += 1

    wait("return document.querySelectorAll('#models .model').length >= 3")
    open_model("vision-model")
    assert js("return document.querySelector('input[name=\"params-vision-device\"][value=\"gpu\"]').checked") is True
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")

    open_model("text-model")
    assert js("return document.getElementById('params-vision').hidden") is True
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")

    open_model("kv-vision")
    kv_state = js("""
      const row = document.getElementById('params-vision');
      const cpu = document.querySelector('input[name="params-vision-device"][value="cpu"]');
      return {
        hidden: row.hidden,
        cpu: cpu.checked,
        disabled: [...row.querySelectorAll('input')].map(i => i.disabled),
        hint: document.getElementById('params-vision-hint').textContent,
        tabLabel: document.getElementById('params-kvmem-tab').firstChild.textContent.trim(),
        native: [...document.querySelectorAll('#params-more [data-param-key]')]
          .map(el => el.dataset.paramKey),
        kvmem: [...document.querySelectorAll('#params-kvmem [data-param-key]')]
          .map(el => el.dataset.paramKey),
        visionValues: ["vision_max_tokens", "vision_threads", "vision_min_tokens"]
          .map(k => (document.querySelector('#params-more [data-param-key="' + k + '"]') || {}).value),
        moreCaption: document.getElementById('params-more-caption').textContent,
        kvHint: document.getElementById('params-kvmem-hint').textContent,
      };
    """)
    assert kv_state["hidden"] is False and kv_state["cpu"] is True, kv_state
    assert kv_state["disabled"] == [False, False], kv_state
    assert "vision.gpu" in kv_state["hint"], kv_state
    assert kv_state["tabLabel"] == "kvmem参数", kv_state
    for key in ("max_context", "window", "lookup", "no_draft", "kv", "layer_split", "threads"):
        assert key in kv_state["native"], (key, kv_state["native"])
    for key in ("vision_threads", "vision_max_tokens", "vision_min_tokens"):
        assert key in kv_state["native"], (key, kv_state["native"])
    assert kv_state["visionValues"] == ["2048", "8", "512"], kv_state["visionValues"]
    for key in ("kvmem_budget", "kvmem_block_tokens", "kvmem_sessions", "kvmem_mtp"):
        assert key in kv_state["kvmem"], (key, kv_state["kvmem"])
    for key in ("max_context", "no_draft", "threads", "vision_max_tokens"):
        assert key not in kv_state["kvmem"], (key, kv_state["kvmem"])
    assert "strata-poly" in kv_state["moreCaption"], kv_state
    assert "--kvmem-*" in kv_state["kvHint"], kv_state
    click('[data-param-tab="more"]')
    shot("kvmem-more-vision-desktop.png")
    click("#params-kvmem-tab")
    shot("kvmem-tabs-desktop.png")
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")

    # A text-only polystrata model has no vision object: the vision specs are
    # not rendered in its "更多参数" tab.
    open_model("kv-text")
    kv_text_state = js("""
      return {
        native: [...document.querySelectorAll('#params-more [data-param-key]')]
          .map(el => el.dataset.paramKey),
      };
    """)
    assert "max_context" in kv_text_state["native"], kv_text_state
    assert "vision_max_tokens" not in kv_text_state["native"], kv_text_state
    assert "vision_threads" not in kv_text_state["native"], kv_text_state
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")

    open_model("text-model")
    llama_state = js("""
      return {
        tabHidden: document.getElementById('params-kvmem-tab').hidden,
        more: [...document.querySelectorAll('#params-more [data-param-key]')]
          .map(el => el.dataset.paramKey),
      };
    """)
    assert llama_state["tabHidden"] is True, llama_state
    assert "typical_p" in llama_state["more"], llama_state["more"]
    assert "max_context" not in llama_state["more"], llama_state["more"]
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")

    open_model("ninfer-vision")
    ninfer_state = js("""
      const row = document.getElementById('params-vision');
      const tab = document.getElementById('params-kvmem-tab');
      return {
        hidden: row.hidden,
        tabHidden: tab.hidden,
        tabLabel: tab.firstChild.textContent.trim(),
        cpu: document.querySelector('input[name="params-vision-device"][value="cpu"]').checked,
        disabled: [...row.querySelectorAll('input')].map(i => i.disabled),
        hint: document.getElementById('params-vision-hint').textContent,
        native: [...document.querySelectorAll('#params-more [data-param-key]')]
          .map(el => el.dataset.paramKey),
        kvmem: [...document.querySelectorAll('#params-kvmem [data-param-key]')]
          .map(el => el.dataset.paramKey),
        moreCaption: document.getElementById('params-more-caption').textContent,
      };
    """)
    assert ninfer_state["hidden"] is False, ninfer_state
    assert ninfer_state["tabHidden"] is False and ninfer_state["tabLabel"] == "kvmem参数", ninfer_state
    assert ninfer_state["cpu"] is True, ninfer_state
    assert ninfer_state["disabled"] == [False, False], ninfer_state
    assert "vision-cpu" in ninfer_state["hint"], ninfer_state
    assert "max_context" in ninfer_state["native"], ninfer_state["native"][:10]
    assert "vision_max_merged" in ninfer_state["native"], ninfer_state["native"][:10]
    for key in ("kvmem", "kvmem_budget", "kvmem_sessions", "kvmem_disk_mib"):
        assert key in ninfer_state["kvmem"], (key, ninfer_state["kvmem"])
    assert "max_context" not in ninfer_state["kvmem"], ninfer_state["kvmem"]
    assert "ninfer-serve" in ninfer_state["moreCaption"], ninfer_state
    click("#params-kvmem-tab")
    shot("ninfer-engine-desktop.png")
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")

    # The add dialog must allow picking CPU/GPU before any mmproj path is set,
    # and expose the NInfer vision switch plus its own engine parameters.
    click("#btn-add-model")
    wait("return document.getElementById('add-dialog').open", message="add dialog")
    wait("return document.querySelectorAll('#add-engines input[name=\"add-engine\"]').length >= 3",
         message="add engines")
    add_state = js("""
      const device = document.getElementById('add-vision-device');
      return {
        disabled: [...device.querySelectorAll('input')].map(i => i.disabled),
        visionHidden: document.getElementById('add-vision-field').hidden,
        enabledHidden: document.getElementById('add-vision-enabled-field').hidden,
        tabHidden: document.getElementById('add-kvmem-tab').hidden,
        engine: (document.querySelector('#add-engines input[name="add-engine"]:checked') || {}).value,
      };
    """)
    assert add_state["disabled"] == [False, False], add_state
    assert add_state["visionHidden"] is False and add_state["enabledHidden"] is True, add_state
    assert add_state["tabHidden"] is True, add_state
    js("""
      const radio = document.querySelector('#add-engines input[name="add-engine"][value="ninfer"]');
      radio.checked = true;
      radio.dispatchEvent(new Event('change', {bubbles: true}));
      return true;
    """)
    ninfer_add = js("""
      const device = document.getElementById('add-vision-device');
      return {
        engine: (document.querySelector('#add-engines input[name="add-engine"]:checked') || {}).value,
        disabled: [...device.querySelectorAll('input')].map(i => i.disabled),
        visionHidden: document.getElementById('add-vision-field').hidden,
        enabledHidden: document.getElementById('add-vision-enabled-field').hidden,
        enabled: document.getElementById('add-vision-enabled').checked,
        tabHidden: document.getElementById('add-kvmem-tab').hidden,
        tabLabel: document.getElementById('add-kvmem-tab').firstChild.textContent.trim(),
        count: document.getElementById('add-kvmem-count').textContent,
        native: [...document.querySelectorAll('#add-more [data-param-key]')]
          .map(el => el.dataset.paramKey),
        kvmem: [...document.querySelectorAll('#add-kvmem [data-param-key]')]
          .map(el => el.dataset.paramKey),
        hint: document.getElementById('add-vision-hint').textContent,
        engineParamsHidden: document.getElementById('add-engine-params').hidden,
      };
    """)
    assert ninfer_add["engine"] == "ninfer", ninfer_add
    assert ninfer_add["disabled"] == [False, False], ninfer_add
    assert ninfer_add["visionHidden"] is True and ninfer_add["enabledHidden"] is False, ninfer_add
    assert ninfer_add["tabHidden"] is False and ninfer_add["tabLabel"] == "kvmem参数", ninfer_add
    assert len(ninfer_add["native"]) >= 20 and "max_context" in ninfer_add["native"], ninfer_add
    assert len(ninfer_add["kvmem"]) == 18, ninfer_add
    for key in ("kvmem_budget", "kvmem_sessions", "kvmem_disk_mib"):
        assert key in ninfer_add["kvmem"], (key, ninfer_add["kvmem"])
    assert ninfer_add["engineParamsHidden"] is True, ninfer_add
    assert "内嵌" in ninfer_add["hint"], ninfer_add
    click("#add-kvmem-tab")
    shot("ninfer-add-desktop.png")

    # Switching to KVMem shows its native process options in “更多参数” and
    # the KVMem extension in “kvmem参数”; the context field lives in the
    # native tab (--max-context), so the basic engine block stays hidden.
    js("""
      const radio = document.querySelector('#add-engines input[name="add-engine"][value="kvmem"]');
      radio.checked = true;
      radio.dispatchEvent(new Event('change', {bubbles: true}));
      return true;
    """)
    kv_add = js("""
      return {
        engine: (document.querySelector('#add-engines input[name="add-engine"]:checked') || {}).value,
        tabHidden: document.getElementById('add-kvmem-tab').hidden,
        tabLabel: document.getElementById('add-kvmem-tab').firstChild.textContent.trim(),
        count: document.getElementById('add-kvmem-count').textContent,
        native: [...document.querySelectorAll('#add-more [data-param-key]')]
          .map(el => el.dataset.paramKey),
        kvmem: [...document.querySelectorAll('#add-kvmem [data-param-key]')]
          .map(el => el.dataset.paramKey),
        engineParamsHidden: document.getElementById('add-engine-params').hidden,
        caption: document.getElementById('add-more-caption').textContent,
      };
    """)
    assert kv_add["engine"] == "kvmem", kv_add
    assert kv_add["tabHidden"] is False and kv_add["tabLabel"] == "kvmem参数", kv_add
    assert kv_add["count"] == "22", kv_add
    assert "max_context" in kv_add["native"] and "no_draft" in kv_add["native"], kv_add
    assert "vision_max_tokens" in kv_add["native"], kv_add
    assert "kvmem_budget" in kv_add["kvmem"] and "max_context" not in kv_add["kvmem"], kv_add
    assert kv_add["engineParamsHidden"] is True, kv_add
    assert "strata-poly" in kv_add["caption"], kv_add
    click("#add-cancel")
    wait("return !document.getElementById('add-dialog').open")

    # Statistics card, fed by the gateway's stats payload.
    wd("POST", "/window/rect", {"width": 1440, "height": 1200})
    js("""
      window.__realFetch = window.fetch;
      window.__fakeStats = {
        at: '2026-11-06T12:00:00Z', model: 'ninfer-vision', engine: 'ninfer',
        prompt_tokens: 25038, completion_tokens: 1827, total_tokens: 26865,
        cached_tokens: 23387, context_length: 262144,
        gen_per_second: 79.9, prefill_per_second: 1356.4,
        prefill_ms: 1220, first_token_ms: 1220, total_ms: 24000,
        draft_n: null, draft_n_accepted: null,
      };
      window.fetch = (input, init) => {
        if (String(input).endsWith('/router/status')) {
          return window.__realFetch(input, init).then(async (response) => {
            const data = await response.json();
            data.stats = { last: window.__fakeStats };
            return new Response(JSON.stringify(data), {
              status: 200, headers: {'Content-Type': 'application/json'},
            });
          });
        }
        return window.__realFetch(input, init);
      };
      return true;
    """)
    wait("return !document.getElementById('stats-section').hidden", timeout=8,
         message="stats section")
    stats_state = js("""
      const cards = [...document.querySelectorAll('#stats .stat-card')];
      return {
        count: cards.length,
        values: cards.map(c => (c.querySelector('.stat-value') || {}).textContent || null),
        subs: cards.map(c => (c.querySelector('.stat-sub') || {}).textContent || ''),
        ring: (document.querySelector('#stats .stat-ring span') || {}).textContent,
        inline: (document.querySelector('#stats .stat-inline') || {}).textContent,
        bar: (document.querySelector('#stats .stat-bar i') || {}).style.width,
        meta: document.getElementById('stats-meta').textContent,
      };
    """)
    assert stats_state["count"] == 4, stats_state
    assert stats_state["values"][0] == "79.9", stats_state
    assert stats_state["values"][1] == "1,356", stats_state
    assert stats_state["ring"] == "10%", stats_state
    assert stats_state["inline"] == "26,865 / 262,144", stats_state
    assert stats_state["values"][3] == "93%", stats_state
    assert stats_state["subs"][1] == "预填充 1.22s · 首 token 1.22s", stats_state
    assert stats_state["subs"][3] == "23,387 / 25,038 tokens", stats_state
    assert stats_state["bar"].startswith("93.4"), stats_state
    assert "ninfer-vision" in stats_state["meta"], stats_state
    shot("stats-desktop.png")
    js("window.fetch = window.__realFetch; delete window.__realFetch;"
       " delete window.__fakeStats; return true;")
    wait("return document.getElementById('stats-section').hidden", timeout=8,
         message="stats hidden without data")
    checks += 1

    wd("POST", "/window/rect", {"width": 420, "height": 1000})
    time.sleep(0.3)
    open_model("vision-model")
    shot("vision-params-mobile.png")
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")
    checks += 1

    errors = js("return window.__uiErrors")
    assert errors == [], errors
    checks += 1
    print(f"PASS {checks} checks")
finally:
    try:
        wd("DELETE", "")
    except Exception:
        pass
    for proc in (driver, router):
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in (driver, router):
        try:
            proc.wait(5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    log.close()
