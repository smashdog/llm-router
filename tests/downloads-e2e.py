#!/usr/bin/env python3
"""End-to-end tests for proxy settings, model search and model downloads.

The test starts a throwaway copy of router.py with fake hf / modelscope CLIs in
front of PATH, so no request ever reaches huggingface.co or modelscope.cn.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "tmp" / "downloads-e2e"
APP = WORK / "app"
FAKE_BIN = WORK / "bin"
MODELS = WORK / "models"
PORT = 18099
BASE = f"http://127.0.0.1:{PORT}"
PASSWORD = "secret-test-pass"

FAKE_HF = f"""#!/usr/bin/python3
import json, os, pathlib, sys, time

def log(entry):
    path = os.environ.get("FAKE_CLI_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\\n")

args = sys.argv[1:]
env = {{key: os.environ.get(key, "") for key in
       ("HF_ENDPOINT", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}}
log({{"tool": "hf", "args": args, "env": env}})

if args[0] == "models" and args[1] == "list" and "--search" in args:
    query = args[args.index("--search") + 1]
    if "boom" in query:
        print("hf search failed password={PASSWORD}", file=sys.stderr)
        sys.exit(3)
    mirror = "hf-mirror" in env["HF_ENDPOINT"]
    prefix = "mirror" if mirror else "hf"
    items = [
        {{"id": prefix + "/alpha", "downloads": 1200 if not mirror else 800,
          "likes": 12, "lastModified": "2026-10-01T10:00:00.000Z",
          "pipeline_tag": "text-generation", "library_name": "transformers",
          "tags": ["license:apache-2.0", "gguf"],
          "safetensors": {{"total": 7600000000}}}},
        {{"id": prefix + "/beta-gguf", "downloads": 45, "likes": 3,
          "lastModified": "2026-09-01T10:00:00.000Z",
          "pipeline_tag": "text-generation", "library_name": "gguf",
          "tags": ["license:mit", "gguf"]}},
    ]
    print(json.dumps(items))
    sys.exit(0)

if args[0] == "models" and args[1] == "list" and "-R" in args:
    repo = args[2]
    if repo.startswith("fake/slow"):
        files = [{{"path": "slow.bin", "size": 2048, "type": "file"}}]
    else:
        files = [
            {{"path": "config.json", "size": 123, "type": "file"}},
            {{"path": "sub/tokenizer.json", "size": 456, "type": "file"}},
            {{"path": "model.gguf", "size": 1048576, "type": "file"}},
            {{"path": "dir", "type": "directory"}},
            {{"path": "onnx", "tree_id": "abc", "last_commit": {{}}}},
        ]
    print(json.dumps(files))
    sys.exit(0)

if args[0] == "download":
    repo = args[1]
    target = pathlib.Path(args[args.index("--local-dir") + 1])
    files = args[args.index("--max-workers") + 2:]
    log({{"phase": "download", "repo": repo, "files": files}})
    print("proxy=" + env["HTTPS_PROXY"])
    print("token {PASSWORD}")
    for name in files:
        if name == "slow.bin":
            time.sleep(60)
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("fake " + name + "\\n", encoding="utf-8")
    sys.exit(0)

print("unexpected hf args: " + repr(args), file=sys.stderr)
sys.exit(4)
"""

FAKE_MODELSCOPE = f"""#!/usr/bin/python3
import json, os, pathlib, sys, time

def log(entry):
    path = os.environ.get("FAKE_CLI_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\\n")

args = sys.argv[1:]
env = {{key: os.environ.get(key, "") for key in
       ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}}
log({{"tool": "modelscope", "args": args, "env": env}})

if args[0] == "list":
    query = args[args.index("--search") + 1]
    if "boom" in query:
        print("modelscope search failed password={PASSWORD}", file=sys.stderr)
        sys.exit(2)
    print("repo_id                downloads  likes  license")
    print("---------------------  ---------  -----  ---------")
    print("ms/alpha               321        7      Apache-2.0")
    print("ms/beta                654        2      MIT")
    print("page 1/1 total 2")
    sys.exit(0)

if args[0] == "download":
    repo = args[1]
    target = pathlib.Path(args[args.index("--local-dir") + 1])
    files = args[args.index("--max-workers") + 2:]
    log({{"phase": "download", "repo": repo, "files": files}})
    print("modelscope download proxy=" + env["HTTPS_PROXY"])
    for name in files:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("modelscope " + name + "\\n", encoding="utf-8")
    sys.exit(0)

print("unexpected modelscope args: " + repr(args), file=sys.stderr)
sys.exit(4)
"""

FAKE_TOOL_PYTHON = """#!/usr/bin/python3
import json, os, sys

if len(sys.argv) >= 3 and sys.argv[1] == "-c" and "HubApi" in sys.argv[2]:
    print(json.dumps([
        {"Path": "config.json", "Size": 123},
        {"Path": "model.gguf", "Size": 1048576},
    ]))
    sys.exit(0)
os.execv("/usr/bin/python3", ["/usr/bin/python3"] + sys.argv[1:])
"""


def request(method: str, path: str, payload=None, expect=200, timeout=40):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8")
    parsed = json.loads(body) if body else None
    assert status == expect, f"{method} {path}: expected {expect}, got {status}: {parsed}"
    return parsed


def wait_ready(proc: subprocess.Popen) -> None:
    for _ in range(120):
        if proc.poll() is not None:
            raise AssertionError("router exited before becoming ready")
        try:
            request("GET", "/router/downloads", timeout=2)
            return
        except Exception:
            time.sleep(0.1)
    raise AssertionError("router did not become ready")


def wait_job(job_id: str, timeout=20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request("GET", f"/router/downloads/job?id={job_id}")
        if job["status"] != "running":
            return job
        time.sleep(0.15)
    raise AssertionError(f"download job {job_id} did not finish")


def fake_log() -> list[dict]:
    path = WORK / "fake.log"
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def start_fake_proxy():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    server.settimeout(0.5)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    conn.recv(65536)
                    conn.sendall(
                        b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n"
                        b"Connection: close\r\n\r\n"
                    )
                except OSError:
                    pass

    thread = threading.Thread(target=loop, name="fake-proxy", daemon=True)
    thread.start()

    def close():
        stop.set()
        server.close()

    return server.getsockname()[1], close


def prepare() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    FAKE_BIN.mkdir(parents=True)
    APP.mkdir(parents=True)
    MODELS.mkdir(parents=True)
    shutil.copy2(ROOT / "router.py", APP / "router.py")
    for name, content in (
        ("hf", FAKE_HF),
        ("modelscope", FAKE_MODELSCOPE),
        ("python3", FAKE_TOOL_PYTHON),
        ("python", FAKE_TOOL_PYTHON),
    ):
        path = FAKE_BIN / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
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


def main() -> int:
    prepare()
    env = os.environ.copy()
    env["PATH"] = str(FAKE_BIN) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["FAKE_CLI_LOG"] = str(WORK / "fake.log")
    env["HTTP_PROXY"] = "http://127.0.0.1:19999"
    env["HTTPS_PROXY"] = "http://127.0.0.1:19999"
    env["ALL_PROXY"] = "http://127.0.0.1:19999"
    env.pop("HF_ENDPOINT", None)
    log_handle = (WORK / "server.out").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(APP / "router.py"), "--config", str(APP / "config.json")],
        cwd=str(APP), env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    checks = 0
    fake_proxy_close = None
    try:
        wait_ready(proc)

        # -- proxy configuration -------------------------------------------
        view = request("GET", "/router/proxy")
        assert view["configured"] is False and view["has_password"] is False, view
        checks += 1

        view = request("POST", "/router/proxy", {
            "enabled": True, "type": "http", "host": "127.0.0.1", "port": 9,
            "username": "tester", "password": PASSWORD,
        })
        assert view["configured"] is True and view["has_password"] is True, view
        assert "password" not in view, view
        mode = stat.S_IMODE((APP / "config.json").stat().st_mode)
        assert mode == 0o600, oct(mode)
        checks += 1

        result = request("POST", "/router/proxy/test", {
            "enabled": True, "type": "http", "host": "127.0.0.1", "port": 9,
            "username": "tester", "password": PASSWORD,
        })
        assert result["ok"] is False and result["reachable"] is False, result
        assert result["tests"] == [], result
        checks += 1

        proxy_port, fake_proxy_close = start_fake_proxy()
        result = request("POST", "/router/proxy/test", {
            "enabled": True, "type": "http", "host": "127.0.0.1",
            "port": proxy_port, "username": "tester", "password": PASSWORD,
        })
        assert result["reachable"] is True and result["ok"] is False, result
        assert result["error"] == "代理端口可连接，但通过代理访问目标站点失败", result
        assert len(result["tests"]) == 2, result
        checks += 1
        fake_proxy_close()
        fake_proxy_close = None

        # -- download directory and source proxy ---------------------------
        error = request("POST", "/router/downloads/plan", {
            "source": "huggingface", "repo_id": "fake/repo",
        }, expect=400)
        assert "下载目录" in error["error"]["message"], error
        checks += 1

        view = request("POST", "/router/downloads/settings", {"dir": str(MODELS)})
        assert view["dir_configured"] is True and view["dir_exists"] is True, view
        assert view["dir"] == str(MODELS), view
        checks += 1

        view = request("GET", "/router/downloads")
        assert view["proxy_configured"] is True, view
        assert view["tools"]["hf"]["available"] is True, view
        assert view["tools"]["modelscope"]["available"] is True, view
        checks += 1

        view = request("POST", "/router/downloads/settings", {
            "sources": {"hf-mirror": {"use_proxy": True}},
        })
        sources = {item["id"]: item for item in view["sources"]}
        assert sources["hf-mirror"]["use_proxy"] is True, view
        assert sources["huggingface"]["use_proxy"] is False, view
        checks += 1

        # The proxy panel can select source usage in the same save request.
        view = request("POST", "/router/proxy", {
            "enabled": True, "type": "http", "host": "127.0.0.1", "port": 9,
            "sources": {"huggingface": False, "hf-mirror": True, "modelscope": False},
        })
        flags = {item["id"]: item["use_proxy"] for item in view["sources"]}
        assert flags == {
            "huggingface": False, "hf-mirror": True, "modelscope": False,
        }, view
        checks += 1
        error = request("POST", "/router/proxy", {
            "enabled": True, "type": "http", "host": "127.0.0.1", "port": 9,
            "sources": {"unknown": True},
        }, expect=400)
        assert "下载源" in error["error"]["message"], error
        checks += 1

        # -- search aggregation --------------------------------------------
        search = request("GET", "/router/downloads/search?q=qwen")
        assert [item["id"] for item in search["sources"]] == [
            "huggingface", "hf-mirror", "modelscope",
        ], search
        assert all(item["ok"] for item in search["sources"]), search
        assert len(search["items"]) == 6, search
        assert {item["source"] for item in search["items"]} == {
            "huggingface", "hf-mirror", "modelscope",
        }, search
        checks += 1

        entries = fake_log()
        hf_searches = [
            entry for entry in entries
            if entry.get("tool") == "hf" and "--search" in entry.get("args", [])
        ]
        assert len(hf_searches) == 2, hf_searches
        direct = next(entry for entry in hf_searches if not entry["env"]["HF_ENDPOINT"])
        mirror = next(entry for entry in hf_searches if entry["env"]["HF_ENDPOINT"])
        assert all(direct["env"][key] == "" for key in
                   ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")), direct
        assert mirror["env"]["HF_ENDPOINT"] == "https://hf-mirror.com", mirror
        assert PASSWORD in mirror["env"]["HTTPS_PROXY"], mirror
        ms_searches = [
            entry for entry in entries
            if entry.get("tool") == "modelscope" and "list" in entry.get("args", [])
        ]
        assert len(ms_searches) == 1, ms_searches
        assert all(ms_searches[0]["env"][key] == "" for key in
                   ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")), ms_searches
        checks += 1

        before = len(fake_log())
        search = request(
            "GET",
            "/router/downloads/search?q=qwen&sources=huggingface,modelscope",
        )
        assert [item["id"] for item in search["sources"]] == [
            "huggingface", "modelscope",
        ], search
        assert {item["source"] for item in search["items"]} == {
            "huggingface", "modelscope",
        }, search
        invoked = {entry.get("tool") for entry in fake_log()[before:]}
        assert invoked == {"hf", "modelscope"}, invoked
        checks += 1
        error = request(
            "GET", "/router/downloads/search?q=qwen&sources=unknown", expect=400
        )
        assert "下载源" in error["error"]["message"], error
        checks += 1

        search = request("GET", "/router/downloads/search?q=boom")
        assert all(not item["ok"] for item in search["sources"]), search
        for item in search["sources"]:
            assert PASSWORD not in (item["error"] or ""), item
            assert "***" in (item["error"] or ""), item
        checks += 1

        # -- HF / mirror plan and download ---------------------------------
        plan = request("POST", "/router/downloads/plan", {
            "source": "huggingface", "repo_id": "fake/repo",
        })
        assert plan["target_dir"] == str(MODELS / "fake" / "repo"), plan
        assert plan["file_count"] == 3, plan
        assert plan["existing_count"] == 0, plan
        assert plan["use_proxy"] is False, plan
        checks += 1

        job = request("POST", "/router/downloads/start", {
            "source": "huggingface", "repo_id": "fake/repo",
            "files": ["model.gguf"],
        }, expect=202)
        job = wait_job(job["id"])
        assert job["status"] == "success", job
        assert job["files"] == ["model.gguf"], job
        assert (MODELS / "fake" / "repo" / "model.gguf").is_file()
        assert PASSWORD not in job["log_tail"], job["log_tail"]
        assert "***" in job["log_tail"], job["log_tail"]
        checks += 1

        plan = request("POST", "/router/downloads/plan", {
            "source": "huggingface", "repo_id": "fake/repo",
        })
        assert plan["existing_count"] == 1, plan
        assert any(item["path"] == "model.gguf" and item["exists"] for item in plan["files"])
        error = request("POST", "/router/downloads/start", {
            "source": "huggingface", "repo_id": "fake/repo", "files": ["model.gguf"],
        }, expect=400)
        assert "文件已存在" in error["error"]["message"], error
        checks += 1

        job = request("POST", "/router/downloads/start", {
            "source": "huggingface", "repo_id": "fake/repo",
        }, expect=202)
        job = wait_job(job["id"])
        assert job["status"] == "success", job
        assert sorted(job["files"]) == ["config.json", "sub/tokenizer.json"], job
        assert (MODELS / "fake" / "repo" / "sub" / "tokenizer.json").is_file()
        checks += 1

        error = request("POST", "/router/downloads/start", {
            "source": "huggingface", "repo_id": "fake/repo",
        }, expect=400)
        assert "均已存在" in error["error"]["message"], error
        checks += 1

        # -- ModelScope plan and download ----------------------------------
        plan = request("POST", "/router/downloads/plan", {
            "source": "modelscope", "repo_id": "ms/repo",
        })
        assert plan["target_dir"] == str(MODELS / "ms" / "repo"), plan
        assert plan["file_count"] == 2, plan
        job = request("POST", "/router/downloads/start", {
            "source": "modelscope", "repo_id": "ms/repo", "files": ["config.json"],
        }, expect=202)
        job = wait_job(job["id"])
        assert job["status"] == "success", job
        assert (MODELS / "ms" / "repo" / "config.json").is_file()
        checks += 1

        # -- cancel --------------------------------------------------------
        job = request("POST", "/router/downloads/start", {
            "source": "huggingface", "repo_id": "fake/slow", "files": ["slow.bin"],
        }, expect=202)
        time.sleep(0.5)
        cancelled = request("POST", "/router/downloads/cancel", {"id": job["id"]}, expect=202)
        assert cancelled["status"] in ("running", "cancelled"), cancelled
        job = wait_job(job["id"])
        assert job["status"] == "cancelled", job
        assert not (MODELS / "fake" / "slow" / "slow.bin").exists()
        checks += 1

        # -- concurrent downloads ------------------------------------------
        job_a = request("POST", "/router/downloads/start", {
            "source": "huggingface", "repo_id": "fake/slow", "files": ["slow.bin"],
        }, expect=202)
        job_b = request("POST", "/router/downloads/start", {
            "source": "huggingface", "repo_id": "fake/slow-b", "files": ["slow.bin"],
        }, expect=202)
        assert job_a["id"] != job_b["id"], (job_a, job_b)
        view = request("GET", "/router/downloads")
        active_ids = {item["id"] for item in view.get("active_jobs") or []}
        assert {job_a["id"], job_b["id"]} <= active_ids, view
        assert view["running_count"] == len(active_ids) >= 2, view
        assert all(
            item.get("status") == "running" for item in view.get("active_jobs") or []
        ), view
        checks += 1
        for entry in (job_a, job_b):
            stopped = request(
                "POST", "/router/downloads/cancel", {"id": entry["id"]}, expect=202
            )
            assert stopped["status"] in ("running", "cancelled"), stopped
        for entry in (job_a, job_b):
            done = wait_job(entry["id"])
            assert done["status"] == "cancelled", done
        view = request("GET", "/router/downloads")
        assert not view.get("active_jobs"), view
        assert view["running_count"] == 0, view
        checks += 1

        # -- validation ----------------------------------------------------
        error = request("POST", "/router/downloads/plan", {
            "source": "huggingface", "repo_id": "not-a-repo",
        }, expect=400)
        assert "owner/name" in error["error"]["message"], error
        error = request("POST", "/router/downloads/start", {
            "source": "huggingface", "repo_id": "fake/repo", "files": ["../evil"],
        }, expect=400)
        assert "不合法" in error["error"]["message"], error
        error = request("POST", "/router/downloads/plan", {
            "source": "unknown", "repo_id": "fake/repo",
        }, expect=400)
        assert "source" in error["error"]["message"], error
        checks += 1

        # -- disabling the proxy clears source proxy flags -----------------
        view = request("POST", "/router/proxy", {
            "enabled": False, "type": "http", "host": "127.0.0.1", "port": 9,
            "username": "tester",
        })
        assert view["configured"] is False and view["has_password"] is True, view
        view = request("GET", "/router/downloads")
        assert all(not item["use_proxy"] for item in view["sources"]), view
        error = request("POST", "/router/downloads/settings", {
            "sources": {"hf-mirror": {"use_proxy": True}},
        }, expect=400)
        assert "代理" in error["error"]["message"], error
        checks += 1

        view = request("POST", "/router/proxy", {
            "enabled": False, "type": "http", "host": "", "port": "",
            "username": "tester", "clear_password": True,
        })
        assert view["has_password"] is False, view
        checks += 1

        print(f"PASS: {checks} download/proxy checks")
        return 0
    finally:
        if fake_proxy_close:
            fake_proxy_close()
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
