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
    "webui": {
        "enabled": True,
        "root": str(ROOT / "webui"),
        "console": "index.html",
        "ui_dir": "llama-ui",
        "ui_path": "/ui",
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
    bad = save("ninfer-vision", vision_device="gpu")
    assert bad.status_code == 400 and "NInfer" in bad.text, bad.text
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
      };
    """)
    assert kv_state["hidden"] is False and kv_state["cpu"] is True, kv_state
    assert kv_state["disabled"] == [False, False], kv_state
    assert "vision.gpu" in kv_state["hint"], kv_state
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")

    open_model("ninfer-vision")
    assert js("return document.getElementById('params-vision').hidden") is True
    click("#params-cancel")
    wait("return !document.getElementById('params-dialog').open")

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
