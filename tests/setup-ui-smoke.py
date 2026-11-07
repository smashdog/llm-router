#!/usr/bin/env python3
"""Headless-Firefox smoke test for the engine provisioning panel."""
from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18109"
OUT = Path(__file__).resolve().parent.parent / "tmp" / "setup-ui"
OUT.mkdir(parents=True, exist_ok=True)

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()
driver = subprocess.Popen(
    ["/snap/bin/geckodriver", "--port", str(port)],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
api = f"http://127.0.0.1:{port}"
for _ in range(100):
    try:
        requests.get(api + "/status", timeout=1)
        break
    except requests.RequestException:
        time.sleep(0.1)
else:
    driver.terminate()
    raise SystemExit("geckodriver did not start")

session = requests.post(api + "/session", json={"capabilities": {"alwaysMatch": {
    "browserName": "firefox",
    "moz:firefoxOptions": {"args": ["-headless"]},
}}}, timeout=60).json()["value"]["sessionId"]


def wd(method, path, payload=None, timeout=60):
    response = requests.request(method, f"{api}/session/{session}{path}", json=payload, timeout=timeout)
    data = response.json()
    if isinstance(data.get("value"), dict) and data["value"].get("error"):
        raise RuntimeError(data["value"])
    return data.get("value")


def js(script):
    return wd("POST", "/execute/sync", {"script": script, "args": []})


def wait(script, timeout=25, message=""):
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


def shot(name):
    data = wd("GET", "/screenshot")
    (OUT / name).write_bytes(base64.b64decode(data))
    print("shot", OUT / name)


wd("POST", "/window/rect", {"width": 1280, "height": 1000})
wd("POST", "/url", {"url": BASE + "/"})
wait("return !!document.querySelector('#btn-manage')", message="app boot")
js("document.querySelector('#btn-manage').click(); return true;")
js("""
  const item = [...document.querySelectorAll('.manage-menu .menu-item')]
    .find((node) => node.dataset.panel === 'engines');
  if (item) item.click();
  return true;
""")
wait("return !!document.querySelector('#engine-install-list .provision-engine')", message="setup cards")
result = js("""
  return {
    toolchains: document.querySelectorAll('#toolchain-rows .toolchain-row').length,
    engines: [...document.querySelectorAll('#engine-install-list .provision-engine')].map((card) => card.innerText.split('\\n').slice(0, 4).join(' | ')),
    downloadButtons: document.querySelectorAll('#toolchain-rows button[data-setup-action="toolchain"]').length,
  };
""")
print(json.dumps(result, ensure_ascii=False, indent=2))
shot("setup-panel.png")
try:
    requests.delete(f"{api}/session/{session}", timeout=10)
except requests.RequestException:
    pass
try:
    driver.terminate()
except OSError:
    pass
print("smoke ok")
