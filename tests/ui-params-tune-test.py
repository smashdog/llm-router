#!/usr/bin/env python3
"""Firefox-headless test for the params-dialog tune button.

Runs the real console against an in-process mock gateway (no engine, no
weights).  Checks:

* the 调优 button shows for tune-eligible engines and hides for others;
* unsaved engine/file changes block tuning with a save-first hint;
* clicking 调优 closes the params dialog and drives /router/models/tune,
  rendering the finished suggestions and applying them;
* a job refused for resource conflict shows the gateway error verbatim.

Usage: python3 tests/ui-params-tune-test.py
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "ui-params-tune"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
(OUT / "shots").mkdir()

MODEL_ID = "Qwen3.8-27B-UD-Q4_K_M"
MODEL = {
    "id": MODEL_ID, "object": "model", "created": 1794494030,
    "owned_by": "KVMem + llama.cpp", "model": MODEL_ID, "name": "Qwen3.8-27B",
    "status": {"value": "loaded", "args": []}, "state": "loaded",
    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
    "meta": {"n_ctx_train": 262144}, "can_remove": False, "tags": [],
    "capabilities": ["vision"], "engine": "kvmem_llama", "engine_model": MODEL_ID,
    "display_name": "Qwen3.8-27B", "display_name_default": "Qwen3.8-27B",
    "display_name_custom": False, "aliases": [], "context_length": 262144,
    "cache_type_k": "q8_0", "cache_type_v": "q8_0", "multimodal": True,
    "vision_gpu": False, "video": True, "audio": False, "reasoning": False,
    "notes": "", "quantization": None, "source": "kvmem_llama",
    "config_path": "/tmp/mock-config.json", "engine_slot": "kvmem_llama",
    "model_path": "/mnt/ai/models/Qwen3.8-27B-UD/Qwen3.8-27B-UD-Q4_K_M.gguf",
    "vision_path": "/mnt/ai/models/Qwen3.8-27B-UD/mmproj-F16.gguf",
    "draft_path": None, "deletable": True, "loaded": True, "available": True,
    "problem": None, "params": {},
    "engine_params": {"kvmem_budget": 32768, "kvmem_block_tokens": 32,
                      "kvmem_gen_reserve": 1024, "kvmem_host_mib": 22016,
                      "batch_size": 2048, "ubatch_size": 512, "ngl": 999,
                      "threads": 8, "flash_attn": "on", "threads_batch": 8},
}
CONFLICT_ERROR = ("显存被其它进程占用（999(comfyui, 17000 MiB)），无法调优；"
                  "请先释放这些进程占用的显存后重试")
STATE = {
    "mode": "ok",            # ok | conflict
    "model": json.loads(json.dumps(MODEL)),
    "tune_calls": [],
    "apply_calls": [],
    "job": None,
    "job_gets": 0,
}


def final_job() -> dict:
    base = {
        "id": "mockjob000001", "kind": "tune", "engine": "kvmem_llama",
        "engine_kind": "kvmem_llama", "model": MODEL_ID,
        "status": "ok", "phase": "done", "message": "",
        "started_at": "2026-11-12T22:00:00+0800",
        "finished_at": "2026-11-12T22:01:00+0800",
        "error": None, "progress": [{"at": "x", "message": "停止当前引擎，释放显存…"}],
        "hardware": {"vram_total_mib": 20464, "vram_free_mib": 20438,
                     "ram_total_mib": 60938, "ram_available_mib": 44292,
                     "cpu_physical_cores": 8, "cpu_logical_cores": 16,
                     "gpus": [{"index": 0, "name": "AMD Radeon RX 7900 XT",
                               "vram_total_mib": 20464, "vram_free_mib": 20438}]},
        "plan": {"context": 262144}, "model_info": {"context_length": 262144},
        "candidates": [{"id": "c1", "label": "推荐卸载（ngl=999）", "ok": True,
                        "decode_tps": 29.08, "prefill_tps": 695.32,
                        "reuse_prompt_tokens": 1749, "reuse_cached_tokens": 1724,
                        "reuse_ratio": 0.986}],
        "notes": ["全量卸载"], "restored": True, "restore_error": None,
        "applied": False, "applied_at": None,
    }
    if STATE["mode"] == "conflict":
        base.update(status="failed", error=CONFLICT_ERROR, candidates=[],
                    suggestions=None, notes=[])
        return base
    base["suggestions"] = {
        "args": ["-ngl", "999"],
        "params": [{"key": "ngl", "label": "GPU 卸载层数", "before": "6", "after": "999"}],
        "summary": "decode 29.08 tok/s", "best_candidate": "c1",
        "context_length": 262144,
    }
    return base


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = (ROOT / "console" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/models":
            self._send({"object": "list", "data": [STATE["model"]]})
        elif path == "/router/status":
            self._send({"service": "llm-router", "version": "0.4.3", "phase": "ready",
                        "switching": False, "in_flight": 0, "queue": 0,
                        "active_engine": "kvmem_llama", "active_model": MODEL_ID,
                        "active_engine_model": MODEL_ID, "target_model": None,
                        "pending_load": None, "pending_unload": None,
                        "async_operation": None, "last_error": None,
                        "engines": {"kvmem_llama": {
                            "engine": "kvmem_llama", "kind": "kvmem_llama",
                            "host": "127.0.0.1", "port": 18083, "pid": 1,
                            "pgid": 1, "alive": True, "model": MODEL_ID}}})
        elif path == "/router/models/manage":
            self._send({"engines": {"kvmem_llama": {
                "engine": "kvmem_llama", "kind": "kvmem_llama",
                "name": "KVMem + llama.cpp", "label": "KVMem + llama.cpp",
                "installed": True}}})
        elif path == "/router/downloads":
            self._send({"active_jobs": []})
        elif path == "/router/models/job":
            STATE["job_gets"] += 1
            job = STATE["job"]
            if job is None:
                self._send({"error": {"message": "no job"}}, 404)
            elif STATE["job_gets"] >= 2 and job["status"] == "running":
                STATE["job"] = final_job()
                self._send(STATE["job"])
            else:
                self._send(job)
        else:
            self._send({})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if path == "/router/models/tune":
            STATE["tune_calls"].append(payload)
            STATE["job_gets"] = 0
            STATE["job"] = {
                "id": "mockjob000001", "kind": "tune", "engine": "kvmem_llama",
                "engine_kind": "kvmem_llama", "model": payload.get("model"),
                "status": "running", "phase": "benchmarking",
                "message": "运行候选基准…", "started_at": "2026-11-12T22:00:00+0800",
                "finished_at": None, "error": None,
                "progress": [{"at": "x", "message": "停止当前引擎，释放显存…"}],
                "hardware": None, "plan": None, "model_info": None,
                "candidates": [], "notes": [], "suggestions": None,
                "restored": False, "restore_error": None,
                "applied": False, "applied_at": None,
            }
            self._send(STATE["job"])
        elif path == "/router/models/tune/apply":
            STATE["apply_calls"].append(payload)
            if STATE["job"]:
                STATE["job"]["applied"] = True
            self._send({"status": "ok", "model": MODEL_ID, "engine": "kvmem_llama",
                        "restarting": True, "restart_pending": False, "note": None})
        else:
            self._send({"ok": True})


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


server = ThreadingHTTPServer(("127.0.0.1", free_port()), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
UI = f"http://127.0.0.1:{server.server_address[1]}/"

driver_port = free_port()
driver = subprocess.Popen(
    ["/snap/bin/geckodriver", "--port", str(driver_port)],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
base = f"http://127.0.0.1:{driver_port}"
for _ in range(60):
    try:
        requests.get(base + "/status", timeout=1)
        break
    except requests.RequestException:
        time.sleep(0.1)
else:
    driver.terminate()
    raise SystemExit("geckodriver did not start")

session = requests.post(base + "/session", json={"capabilities": {"alwaysMatch": {
    "browserName": "firefox",
    "moz:firefoxOptions": {"args": ["-headless"]},
}}}, timeout=60).json()["value"]["sessionId"]


def wd(method, path, payload=None, timeout=60):
    response = requests.request(method, f"{base}/session/{session}{path}",
                                json=payload, timeout=timeout)
    data = response.json()
    if data.get("value") and isinstance(data["value"], dict) and data["value"].get("error"):
        raise RuntimeError(data["value"])
    return data.get("value")


def js(script):
    return wd("POST", "/execute/sync", {"script": script, "args": []})


def wait(script, timeout=20, message=""):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = js(f"return ({script})")
        if last:
            return last
        time.sleep(0.2)
    raise AssertionError(f"timeout: {message or script} (last={last!r})")


def shot(name):
    png = wd("GET", "/screenshot")
    import base64
    (OUT / "shots" / f"{name}.png").write_bytes(base64.b64decode(png))


def open_params():
    js(f'document.querySelector(\'button[data-action="params"][data-id="{MODEL_ID}"]\').click()')
    wait('document.getElementById("params-dialog").open', message="params dialog open")


failures = []
try:
    wd("POST", "/window/rect", {"width": 1400, "height": 1000})
    wd("POST", "/url", {"url": UI})
    wait(f'document.querySelector(\'button[data-action="params"][data-id="{MODEL_ID}"]\')',
         message="model card rendered")

    # 1. tune button visible for kvmem_llama
    open_params()
    assert js('return !document.getElementById("params-tune").hidden') is True
    shot("01-params-tune-visible")
    print("ok: 调优按钮在 kvmem_llama 参数界面可见")

    # 2. unsaved file change blocks tuning
    js('''const el = document.getElementById("params-model-path");
          el.value = "/tmp/other.gguf";
          el.dispatchEvent(new Event("input", {bubbles: true}));''')
    js('document.getElementById("params-tune").click()')
    wait('!document.getElementById("params-error").hidden', message="save-first error")
    err = js('return document.getElementById("params-error").textContent')
    assert "请先保存" in err, err
    assert js('return document.getElementById("tune-dialog").open') is False
    assert len(STATE["tune_calls"]) == 0
    shot("02-save-first-guard")
    print("ok: 未保存的文件改动会阻止调优 ->", err)
    js('''const el = document.getElementById("params-model-path");
          el.value = %s;
          el.dispatchEvent(new Event("input", {bubbles: true}));'''
       % json.dumps(MODEL["model_path"]))

    # 3. tune flow: params closes, tune dialog runs to suggestions, apply works
    js('document.getElementById("params-tune").click()')
    wait('document.getElementById("tune-dialog").open', message="tune dialog open")
    assert js('return document.getElementById("params-dialog").open') is False
    assert STATE["tune_calls"] and STATE["tune_calls"][0]["model"] == MODEL_ID
    wait('!document.getElementById("tune-apply").disabled', timeout=30,
         message="tune finished, apply enabled")
    summary = js('return document.getElementById("tune-summary").textContent')
    assert "29.08" in summary, summary
    shot("03-tune-suggestions")
    print("ok: 调优流程完成并渲染建议 ->", summary)
    js('document.getElementById("tune-apply").click()')
    wait('!document.getElementById("tune-dialog").open', message="tune dialog closed after apply")
    assert len(STATE["apply_calls"]) == 1
    print("ok: 应用建议已发送并关闭对话框")

    # 4. resource-conflict job shows the gateway error verbatim
    STATE["mode"] = "conflict"
    open_params()
    js('document.getElementById("params-tune").click()')
    wait('!document.getElementById("tune-error").hidden', timeout=30,
         message="conflict error shown")
    err = js('return document.getElementById("tune-error").textContent')
    assert "comfyui" in err and "无法调优" in err, err
    label = js('return document.getElementById("tune-decline").textContent')
    assert label == "关闭", label
    shot("04-conflict-error")
    print("ok: 资源冲突错误原样展示 ->", err)
    js('document.getElementById("tune-decline").click()')

    # 5. non-eligible engine hides the button
    STATE["model"]["engine"] = "llama"
    wd("POST", "/url", {"url": UI})
    wait(f'document.querySelector(\'button[data-action="params"][data-id="{MODEL_ID}"]\')',
         message="model card re-rendered")
    open_params()
    assert js('return document.getElementById("params-tune").hidden') is True
    shot("05-llama-hidden")
    print("ok: 非调优引擎（llama）不显示调优按钮")

    print("ALL PARAMS-TUNE UI TESTS PASSED")
except Exception as exc:
    failures.append(str(exc))
    shot("99-failure")
    print("FAILED:", exc)
finally:
    try:
        wd("DELETE", "")
    except Exception:
        pass
    # The snap-confined geckodriver may refuse signals; never fail a passed
    # test because of cleanup.
    try:
        driver.terminate()
    except Exception:
        pass
    try:
        driver.wait(5)
    except Exception:
        try:
            driver.kill()
        except Exception:
            if driver.poll() is None:
                try:
                    subprocess.run(
                        ["snap", "run", "--shell", "firefox", "-c",
                         f"kill -9 {driver.pid}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=20,
                    )
                except Exception:
                    pass
    server.shutdown()

sys.exit(1 if failures else 0)