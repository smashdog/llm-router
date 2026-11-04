#!/usr/bin/env python3
"""Real-network smoke test for the HF-Mirror and ModelScope download sources.

Only hf-mirror.com and modelscope.cn are contacted; the direct Hugging Face
source is deliberately not exercised here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "tmp" / "downloads-real"
APP = WORK / "app"
MODELS = WORK / "models"
TOOLS = ROOT / "tmp" / "tool-venv" / "bin"
PORT = 18100
BASE = f"http://127.0.0.1:{PORT}"


def request(method, path, payload=None, expect=200, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, body = response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read().decode()
    parsed = json.loads(body) if body else None
    assert status == expect, f"{method} {path}: {status} {parsed}"
    return parsed


def wait_ready(proc):
    for _ in range(150):
        if proc.poll() is not None:
            raise AssertionError("router exited before becoming ready")
        try:
            return request("GET", "/router/downloads", timeout=2)
        except Exception:
            time.sleep(0.1)
    raise AssertionError("router did not become ready")


def wait_job(job_id, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request("GET", f"/router/downloads/job?id={job_id}")
        if job["status"] != "running":
            return job
        time.sleep(0.4)
    raise AssertionError(f"job {job_id} did not finish")


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    APP.mkdir(parents=True)
    MODELS.mkdir(parents=True)
    shutil.copy2(ROOT / "router.py", APP / "router.py")
    config = {
        "listen": {"host": "127.0.0.1", "port": PORT},
        "engines": {},
        "timeouts": {"queue_s": 60},
        "log_file": str(WORK / "router.log"),
        "state_file": str(WORK / "state.json"),
        "pid_file": str(WORK / "router.pid"),
        "model_params_file": str(WORK / "model-params.json"),
        "api_keys_file": str(WORK / "api-keys.json"),
        "proxy": {
            "enabled": False, "type": "http", "host": "", "port": 0,
            "username": "", "password": "",
        },
        "downloads": {
            "dir": "",
            "sources": {
                "huggingface": {"use_proxy": False},
                "hf-mirror": {"use_proxy": False},
                "modelscope": {"use_proxy": False},
            },
        },
    }
    (APP / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = str(TOOLS) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "HF_ENDPOINT"):
        env.pop(key, None)
    log_handle = (WORK / "server.out").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(APP / "router.py"), "--config", str(APP / "config.json")],
        cwd=str(APP), env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    checks = 0
    try:
        view = wait_ready(proc)
        assert view["tools"]["hf"]["available"], view
        assert view["tools"]["modelscope"]["available"], view
        view = request("POST", "/router/downloads/settings", {"dir": str(MODELS)})
        assert view["dir_exists"], view
        checks += 1

        plan = request("POST", "/router/downloads/plan", {
            "source": "hf-mirror", "repo_id": "hf-internal-testing/tiny-random-bert",
        })
        paths = {item["path"]: item for item in plan["files"]}
        assert "config.json" in paths, plan
        assert not any(name in paths for name in ("onnx", "onnx/")), plan
        assert plan["target_dir"] == str(
            MODELS / "hf-internal-testing" / "tiny-random-bert"
        ), plan
        job = request("POST", "/router/downloads/start", {
            "source": "hf-mirror", "repo_id": "hf-internal-testing/tiny-random-bert",
            "files": ["config.json"],
        }, expect=202)
        job = wait_job(job["id"])
        assert job["status"] == "success", job
        target = MODELS / "hf-internal-testing" / "tiny-random-bert" / "config.json"
        assert target.is_file() and target.stat().st_size > 100, target
        checks += 1

        plan = request("POST", "/router/downloads/plan", {
            "source": "modelscope", "repo_id": "inclusionAI/Ling-3.0-tiny",
        })
        paths = {item["path"]: item for item in plan["files"]}
        assert "config.json" in paths, plan
        job = request("POST", "/router/downloads/start", {
            "source": "modelscope", "repo_id": "inclusionAI/Ling-3.0-tiny",
            "files": ["config.json"],
        }, expect=202)
        job = wait_job(job["id"])
        assert job["status"] == "success", job
        target = MODELS / "inclusionAI" / "Ling-3.0-tiny" / "config.json"
        assert target.is_file() and target.stat().st_size > 100, target
        checks += 1

        print(f"PASS: {checks} real-network download checks (mirror + modelscope)")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)
        log_handle.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
