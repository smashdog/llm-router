#!/usr/bin/env python3
"""Real-gateway test: engine-page proxy checkboxes survive UI reloads and restarts."""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tmp" / "setup-proxy-test"
OUT = ROOT / "tmp" / "setup-proxy"
OUT.mkdir(parents=True, exist_ok=True)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wd(api, session, method, path, payload=None, timeout=60):
    response = requests.request(method, f"{api}/session/{session}{path}", json=payload, timeout=timeout)
    data = response.json()
    value = data.get("value")
    if isinstance(value, dict) and value.get("error"):
        raise RuntimeError(value)
    return value


def js(api, session, script):
    return wd(api, session, "POST", "/execute/sync", {"script": script, "args": []})


def wait(api, session, script, timeout=20, message=""):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = js(api, session, script)
            if last:
                return last
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.15)
    raise SystemExit(f"wait failed: {message or script}: {last}")


def wait_http(url, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1)
            if response.ok:
                return response
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise SystemExit(f"server did not start: {url}")


def start_router(port: int):
    log = (OUT / "router.out").open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "router.py", "--host", "127.0.0.1", "--port", str(port)],
        cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
    )
    wait_http(f"http://127.0.0.1:{port}/router/status")
    return proc


def stop_router(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ui", action="store_true", help="only test the HTTP persistence")
    args = parser.parse_args()

    if BASE.exists():
        shutil.rmtree(BASE)
    (BASE / "logs").mkdir(parents=True)
    shutil.copy2(ROOT / "router.py", BASE / "router.py")
    shutil.copy2(ROOT / "engine_setup.py", BASE / "engine_setup.py")
    (BASE / "console").symlink_to(ROOT / "console")

    port = free_port()
    config = {
        "listen": {"host": "127.0.0.1", "port": port},
        "state_file": "state.json",
        "pid_file": "router.pid",
        "log_file": "logs/router.log",
        "model_params_file": "model-params.json",
        "model_meta_file": "model-meta.json",
        "console": {"enabled": True, "root": "console", "file": "index.html"},
        "proxy": {
            "enabled": True, "type": "http", "host": "127.0.0.1", "port": 10809,
            "username": "", "password": "",
        },
        "engines": {},
        "setup": {},
    }
    config_path = BASE / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    url = f"http://127.0.0.1:{port}"

    router = start_router(port)
    driver = None
    try:
        view = requests.get(url + "/router/setup", timeout=10).json()
        assert view.get("setup_proxy") and not any(view["setup_proxy"].values()), view
        assert view.get("proxy_configured") is True, view

        saved = requests.post(
            url + "/router/setup/proxy",
            json={"key": "engine:kvmem", "enabled": True}, timeout=10,
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["setup_proxy"].get("engine:kvmem") is True, saved.text
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        assert on_disk["setup_proxy"]["engine:kvmem"] is True, on_disk
        bad = requests.post(
            url + "/router/setup/proxy", json={"key": "engine:nope", "enabled": True}, timeout=10,
        )
        assert bad.status_code == 400, bad.text
        print("proxy persistence: HTTP ok")

        if not args.no_ui:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            driver_port = sock.getsockname()[1]
            sock.close()
            driver = subprocess.Popen(
                ["/snap/bin/geckodriver", "--port", str(driver_port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            api = f"http://127.0.0.1:{driver_port}"
            wait_http(api + "/status")
            session = requests.post(api + "/session", json={"capabilities": {"alwaysMatch": {
                "browserName": "firefox",
                "moz:firefoxOptions": {"args": ["-headless"]},
            }}}, timeout=60).json()["value"]["sessionId"]
            wd(api, session, "POST", "/window/rect", {"width": 1280, "height": 1000})

            def open_engines():
                wd(api, session, "POST", "/url", {"url": url + "/"})
                wait(api, session, "return !!document.querySelector('#btn-manage')", message="app boot")
                js(api, session, "document.getElementById('btn-manage').click(); return true;")
                js(api, session, """
                  const item = [...document.querySelectorAll('.manage-menu .menu-item')]
                    .find((node) => node.dataset.panel === 'engines');
                  if (item) item.click();
                  return true;
                """)
                return wait(
                    api, session,
                    "return document.querySelector('input[data-setup-proxy=\"engine:kvmem\"]')",
                    message="kvmem proxy checkbox",
                )

            def checked():
                return js(api, session, """
                  return document.querySelector('input[data-setup-proxy="engine:kvmem"]').checked;
                """)

            open_engines()
            assert checked() is True, "persisted checkbox should load checked"
            js(api, session, """
              document.querySelector('input[data-setup-proxy="engine:kvmem"]').click();
              return true;
            """)
            deadline = time.time() + 10
            while time.time() < deadline:
                if requests.get(url + "/router/setup", timeout=5).json()["setup_proxy"].get("engine:kvmem") is False:
                    break
                time.sleep(0.15)
            else:
                raise SystemExit("unchecking did not persist")
            assert checked() is False, "checkbox should be off after click"
            print("proxy persistence: UI toggle off ok")

            wd(api, session, "POST", "/url", {"url": url + "/"})
            open_engines()
            assert checked() is False, "off state lost after reload"
            print("proxy persistence: reload keeps off ok")

            stop_router(router)
            router = start_router(port)
            open_engines()
            assert checked() is False, "off state lost after restart"
            print("proxy persistence: restart keeps off ok")

            js(api, session, """
              document.querySelector('input[data-setup-proxy="engine:kvmem"]').click();
              return true;
            """)
            deadline = time.time() + 10
            while time.time() < deadline:
                if requests.get(url + "/router/setup", timeout=5).json()["setup_proxy"].get("engine:kvmem") is True:
                    break
                time.sleep(0.15)
            else:
                raise SystemExit("rechecking did not persist")
            stop_router(router)
            router = start_router(port)
            open_engines()
            assert checked() is True, "on state lost after restart"
            print("proxy persistence: restart keeps on ok")
    finally:
        if driver is not None:
            try:
                driver.terminate()
            except OSError:
                pass
        stop_router(router)
    print("setup-proxy-test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
