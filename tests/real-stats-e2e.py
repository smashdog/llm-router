#!/usr/bin/env python3
"""Load one real local model through one engine and verify generation stats.

Usage: real-stats-e2e.py --engine llama|kvmem_llama|kvmem [--ui]

The router runs from a throwaway directory with a copy of router.py, its own
state and logs, and the real engine binaries/models. Non-streaming and
streaming chat requests are sent through /v1; /router/status.stats.last must
carry prompt/completion tokens, rates, context length and first-token time.
With --ui the console is opened in headless Firefox and screenshotted.
"""
from __future__ import annotations

import argparse
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
MODELS = Path("/mnt/ai/models")
QWEN_DIR = MODELS / "Qwen3.8-27B-UD"
QWEN_GGUF = QWEN_DIR / "Qwen3.8-27B-Uncensored-Heretic-MTP-UD-IQ4_XS.gguf"
QWEN_MMPROJ = QWEN_DIR / "mmproj-F16.gguf"
LOCAL_KV = Path("/mnt/ai/llm/kvmem/local_config")
LOCAL_KV_LLAMA = ROOT / "models-kvmem-llama"

MODELS_BY_ENGINE = {
    "llama": "Qwen3.8-27B",
    "kvmem_llama": "Qwen3.8-27B-Uncensored-Heretic-MTP-UD-IQ4_XS",
    "kvmem": "ornith-1.5-35b-a3b-vl-thinking",
}


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def engine_configs(base: Path, preset: Path) -> dict:
    return {
        "llama": {
            "kind": "llama_router",
            "host": "127.0.0.1",
            "port": 18080,
            "cwd": "/mnt/ai/llm/llama.cpp",
            "command": [
                "/mnt/ai/llm/llama.cpp/build/bin/llama-server",
                "--models-preset", str(preset),
                "--models-dir", str(base / "llama-models"),
                "--host", "{host}", "--port", "{port}",
            ],
            "start_timeout_s": 300,
            "log_path": "logs/llama-engine.log",
            "models": {"kind": "llama_dir", "dir": str(base / "llama-models"), "preset": str(preset)},
            "env": {"HIP_VISIBLE_DEVICES": "0"},
        },
        "kvmem": {
            "kind": "kvmem",
            "host": "127.0.0.1",
            "port": 18081,
            "cwd": "/mnt/ai/llm/kvmem/backends/polystrata",
            "command": [
                "/mnt/ai/llm/kvmem/.venv/bin/python", "-m", "serve.server",
                "--engine", "strata", "--config", "{config}",
                "--host", "{host}", "--port", "{port}", "--switch-on-request",
            ],
            "start_timeout_s": 900,
            "log_path": "logs/kvmem-engine.log",
            "env": {"PYTHONUNBUFFERED": "1"},
            "models": {"kind": "kvmem_dir", "dir": str(LOCAL_KV), "meta": str(LOCAL_KV / "models.json")},
        },
        "kvmem_llama": {
            "kind": "kvmem_llama",
            "host": "127.0.0.1",
            "port": 18083,
            "cwd": "/mnt/ai/llm/kvmem-llamacpp090",
            "command": [
                "/mnt/ai/llm/kvmem-llamacpp090/build-hip/bin/llama-kvmem-server",
                "--host", "{host}", "--port", "{port}",
            ],
            "start_timeout_s": 900,
            "log_path": "logs/kvmem-llama-engine.log",
            "models": {
                "kind": "kvmem_llama_dir",
                "dir": str(LOCAL_KV_LLAMA),
                "template": str(LOCAL_KV_LLAMA / "kvmem-llama.template.json"),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=sorted(MODELS_BY_ENGINE))
    parser.add_argument("--ui", action="store_true")
    args = parser.parse_args()

    base = ROOT / "tmp" / f"real-stats-{args.engine}"
    if base.exists():
        shutil.rmtree(base)
    (base / "logs").mkdir(parents=True)
    (base / "llama-models").mkdir()
    shutil.copy2(ROOT / "router.py", base / "router.py")
    (base / "console").symlink_to(ROOT / "console")
    shutil.copy2(ROOT / "smoke.png", base / "smoke.png")
    shutil.copy2(ROOT / "api-key", base / "api-key")
    api_key = (ROOT / "api-key").read_text(encoding="utf-8").strip()

    preset = base / "models.ini"
    preset.write_text(
        "version = 1\n\n[*]\n"
        "cache-type-k = q8_0\ncache-type-v = q8_0\nflash-attn = on\nn-gpu-layers = 999\n\n"
        "[Qwen3.8-27B]\n"
        "c = 32768\n"
        f"model = {QWEN_GGUF}\n"
        f"mmproj = {QWEN_MMPROJ}\n",
        encoding="utf-8",
    )

    port = free_port()
    config = {
        "listen": {"host": "127.0.0.1", "port": port},
        "state_file": "state.json",
        "pid_file": "router.pid",
        "log_dir": "logs",
        "log_file": "logs/router.log",
        "model_params_file": "model-params.json",
        "console": {"enabled": True, "root": "console", "file": "index.html"},
        "auth": {"api_key_file": "api-key"},
        "engines": {args.engine: engine_configs(base, preset)[args.engine]},
        "timeouts": {"load_timeout_s": 1500, "stop_timeout_s": 120,
                     "queue_timeout_s": 1800, "proxy_read_timeout_s": 3600},
    }
    (base / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    url = f"http://127.0.0.1:{port}"
    auth = {"Authorization": f"Bearer {api_key}"}
    log = (base / "router.out").open("w", encoding="utf-8")
    router = subprocess.Popen(
        [sys.executable, "router.py", "--config", "config.json"],
        cwd=base, stdout=log, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    driver = None
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if router.poll() is not None:
                raise SystemExit("router exited:\n" + (base / "router.out").read_text())
            try:
                requests.get(url + "/router/status", timeout=1)
                break
            except requests.RequestException:
                time.sleep(0.2)
        else:
            raise SystemExit("router did not start")

        model_id = MODELS_BY_ENGINE[args.engine]
        items = requests.get(url + "/models", timeout=10).json()["data"]
        ids = [item["id"] for item in items]
        if model_id not in ids:
            raise SystemExit(f"model {model_id!r} not discovered: {ids}")
        entry = next(item for item in items if item["id"] == model_id)
        print(f"model {model_id}: engine={entry['engine']} ctx={entry.get('context_length')} "
              f"vision={entry.get('multimodal')}", flush=True)

        response = requests.post(url + "/models/load", json={"model": model_id},
                                 headers=auth, timeout=30)
        assert response.status_code in (200, 202), response.text
        deadline = time.time() + 1500
        last_phase = None
        while time.time() < deadline:
            status = requests.get(url + "/router/status", timeout=10).json()
            if status.get("active_model") == model_id:
                break
            if status.get("phase") == "error":
                raise SystemExit(f"load failed: {status.get('last_error')}\n"
                                 + (base / "router.out").read_text()[-4000:])
            if status.get("phase") != last_phase:
                print(f"  phase={status.get('phase')} target={status.get('target_model')}",
                      flush=True)
                last_phase = status.get("phase")
            time.sleep(1.5)
        else:
            raise SystemExit("model load timed out")
        print("loaded", flush=True)

        long_prompt = (
            "请先仔细阅读下面这段话，然后用一句话概括它的意思。\n"
            "推理服务需要在有限的显存里同时容纳权重、KV cache 和临时缓冲区，"
            "Prefill 阶段一次性处理整段提示，Decode 阶段逐个 token 生成；"
            "两者的吞吐量差别很大，前端展示运行统计时要区分预填充速度和生成速度，"
            "并把上下文占用、缓存复用比例一并呈现给使用者。\n"
        )
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": long_prompt + "再用一句话说明你是谁。"}],
            "max_tokens": 48,
            "temperature": 0,
        }
        first = requests.post(url + "/v1/chat/completions", json=payload,
                              headers=auth, timeout=600)
        assert first.status_code == 200, first.text
        first_stats = requests.get(url + "/router/status", timeout=10).json()["stats"]["last"]

        stream_payload = dict(payload, max_tokens=8, stream=True)
        events = 0
        with requests.post(url + "/v1/chat/completions", json=stream_payload,
                           headers=auth, timeout=600, stream=True) as streamed:
            assert streamed.status_code == 200, streamed.text
            for line in streamed.iter_lines():
                if line.startswith(b"data:"):
                    events += 1
        assert events >= 2, events

        stats = requests.get(url + "/router/status", timeout=10).json()["stats"]["last"]
        print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
        assert stats["model"] == model_id, stats
        assert stats["prompt_tokens"] > 0 and stats["completion_tokens"] > 0, stats
        assert stats["total_tokens"] == stats["prompt_tokens"] + stats["completion_tokens"], stats
        assert stats["gen_per_second"] and stats["gen_per_second"] > 0, stats
        assert stats["prefill_per_second"] and stats["prefill_per_second"] > 0, stats
        assert stats["prefill_ms"] and stats["prefill_ms"] > 0, stats
        assert stats["first_token_ms"] is not None, stats
        assert stats["context_length"] == entry.get("context_length"), stats
        assert (stats["completion_tokens"], stats["total_ms"]) != (
            first_stats["completion_tokens"], first_stats["total_ms"]), (first_stats, stats)
        assert stats["total_ms"] > 0, stats

        if args.ui:
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
            session = requests.post(driver_base + "/session", json={"capabilities": {"alwaysMatch": {
                "browserName": "firefox",
                "moz:firefoxOptions": {"args": ["-headless"]},
            }}}, timeout=60).json()["value"]["sessionId"]

            def wd(method, path, payload=None):
                return requests.request(method, f"{driver_base}/session/{session}{path}",
                                        json=payload, timeout=120).json()["value"]

            wd("POST", "/window/rect", {"width": 1440, "height": 1100})
            wd("POST", "/url", {"url": url + "/"})
            deadline = time.time() + 20
            while time.time() < deadline:
                shown = wd("POST", "/execute/sync", {"script":
                    "return !document.getElementById('stats-section').hidden;", "args": []})
                if shown:
                    break
                time.sleep(0.3)
            else:
                raise SystemExit("stats card did not appear")
            (base / "shots").mkdir(exist_ok=True)
            shot = wd("GET", "/screenshot")
            (base / "shots" / f"stats-{args.engine}.png").write_bytes(base64.b64decode(shot))
            print(f"screenshot: {base / 'shots' / f'stats-{args.engine}.png'}", flush=True)
            try:
                wd("DELETE", "")
            except Exception:  # noqa: BLE001
                pass

        print(f"PASS {args.engine}", flush=True)
        return 0
    finally:
        if driver is not None:
            # Snap geckodriver can live in a namespace where signals are
            # rejected; never let cleanup mask the test result.
            try:
                driver.terminate()
                try:
                    driver.wait(5)
                except subprocess.TimeoutExpired:
                    driver.kill()
            except OSError:
                pass
        if router.poll() is None:
            try:
                router.send_signal(signal.SIGTERM)
                try:
                    router.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    router.kill()
                    router.wait(timeout=10)
            except OSError:
                pass
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
