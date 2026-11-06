#!/usr/bin/env python3
"""Firefox-headless UI test for the grouped engine page and two-column engine picker."""
from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "ui-engines"
MOCK = "http://127.0.0.1:18097/"
MOCK_LOG = OUT / "mock.log"
OUT.mkdir(parents=True, exist_ok=True)

mock = subprocess.Popen(
    [sys.executable, str(ROOT / "tests" / "ui-engines-mock.py")],
    stdout=MOCK_LOG.open("w", encoding="utf-8"), stderr=subprocess.STDOUT,
)
for _ in range(60):
    try:
        requests.get(MOCK + "router/status", timeout=1)
        break
    except requests.RequestException:
        time.sleep(0.1)
else:
    mock.terminate()
    raise SystemExit("mock did not start")

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()
driver = subprocess.Popen(
    ["/snap/bin/geckodriver", "--port", str(port)],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
base = f"http://127.0.0.1:{port}"
for _ in range(60):
    try:
        requests.get(base + "/status", timeout=1)
        break
    except requests.RequestException:
        time.sleep(0.1)
else:
    mock.terminate()
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


def rect(width, height):
    wd("POST", "/window/rect", {"width": width, "height": height})


def js(script):
    return wd("POST", "/execute/sync", {"script": script, "args": []})


def wait(script, timeout=20, message=""):
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
    print("shot", name)


def click(selector):
    js(f"""
      const el = document.querySelector({json.dumps(selector)});
      if (!el) throw new Error('missing ' + {json.dumps(selector)});
      el.click();
      return true;
    """)


def add_subs():
    return js("""
      return [...document.querySelectorAll('#add-engine-subs .engine-sub')].map(card => ({
        name: card.querySelector('.title').textContent.trim(),
        value: card.querySelector('input').value,
        checked: card.querySelector('input').checked,
        on: card.classList.contains('on'),
      }));
    """)


try:
    rect(1440, 1200)
    wd("POST", "/url", {"url": MOCK})
    wait("return !document.getElementById('console-content').hidden",
         message="console content")
    js("""
      window.__uiErrors = [];
      window.addEventListener('error', e => window.__uiErrors.push(String(e.message)));
      window.addEventListener('unhandledrejection',
        e => window.__uiErrors.push(String(e.reason && e.reason.message || e.reason)));
      return true;
    """)

    # -- grouped engine management page --------------------------------
    click("#btn-manage")
    wait("return !document.getElementById('manage-view').hidden", message="manage view")
    wait("return document.querySelectorAll('#engine-paths .engine-group').length === 2",
         message="engine groups")

    assert js("""return document.querySelector(
      '.manage-menu .menu-item[data-panel="engines"]').textContent.trim()""") == "引擎"
    assert js("return document.getElementById('engine-paths-title').textContent.trim()") == "引擎"

    groups = js("""
      return [...document.querySelectorAll('#engine-paths .engine-group')].map(group => ({
        category: group.dataset.category,
        names: [...group.querySelectorAll('.engine-path-head h3')].map(h => h.textContent.trim()),
        engines: [...group.querySelectorAll('.engine-path')].map(card => card.dataset.engine),
      }));
    """)
    assert groups == [
        {"category": "llama.cpp", "names": ["llama.cpp"], "engines": ["llama"]},
        {"category": "kvmem", "names": ["llama.cpp", "polystrata", "ninfer"],
         "engines": ["kvmem_llama", "kvmem", "ninfer"]},
    ], groups

    tags = js("""
      return Object.fromEntries([...document.querySelectorAll('.engine-path')].map(card => [
        card.dataset.engine,
        [...card.querySelectorAll('.engine-path-head .tag')].map(t => t.textContent.trim()),
      ]));
    """)
    assert tags["llama"] == ["已配置", "运行中"], tags
    assert tags["kvmem_llama"] == ["已配置", "运行中", "需重启"], tags
    assert tags["kvmem"] == ["已配置", "已停止"], tags
    assert tags["ninfer"] == ["未配置"], tags
    shot("engines-manage-desktop.png")

    rect(430, 1100)
    time.sleep(0.4)
    overflow = js("""
      const host = document.getElementById('engine-paths');
      return {scroll: host.scrollWidth, client: host.clientWidth};
    """)
    assert overflow["scroll"] <= overflow["client"] + 1, overflow
    shot("engines-manage-mobile.png")
    rect(1440, 1200)
    time.sleep(0.3)

    # -- two-column engine picker in the add dialog ---------------------
    click("#btn-manage")
    wait("return document.getElementById('manage-view').hidden", message="console view")
    click("#btn-add-model")
    wait("return document.getElementById('add-dialog').open", message="add dialog")
    wait("return document.querySelectorAll('#add-engines .engine-cat').length === 2",
         message="engine categories")

    cats = js("""return [...document.querySelectorAll('#add-engines .engine-cat')]
      .map(b => b.textContent.trim())""")
    assert cats == ["llama.cpp", "kvmem"], cats
    assert js("return document.querySelector('#add-engines .engine-cat.on').textContent.trim()") == "llama.cpp"

    subs = add_subs()
    assert subs == [{"name": "llama.cpp", "value": "llama", "checked": True, "on": True}], subs
    assert js("return document.querySelector('input[name=\"add-engine\"]:checked').value") == "llama"

    click('#add-engines .engine-cat[data-add-category="kvmem"]')
    wait("return document.querySelectorAll('#add-engine-subs .engine-sub').length === 3",
         message="kvmem sub engines")
    subs = add_subs()
    assert [item["name"] for item in subs] == ["llama.cpp", "polystrata", "ninfer"], subs
    assert [item["value"] for item in subs] == ["kvmem_llama", "kvmem", "ninfer"], subs
    assert subs[0]["checked"] and subs[0]["on"], subs
    assert not subs[1]["checked"] and not subs[2]["checked"], subs
    assert js("return document.querySelector('#add-engines .engine-cat.on').textContent.trim()") == "kvmem"
    assert "llama-kvmem-server" in js("return document.getElementById('add-more-caption').textContent")

    click('#add-engine-subs input[value="kvmem"]')
    wait("return document.querySelector('#add-engine-subs input[value=\"kvmem\"]').checked",
         message="polystrata selected")
    subs = add_subs()
    assert [item["on"] for item in subs] == [False, True, False], subs
    assert "strata-poly" in js("return document.getElementById('add-more-caption').textContent")
    assert "模型目录或 .gguf" in js("return document.getElementById('add-model-hint').textContent")
    shot("add-dialog-desktop.png")

    rect(430, 1100)
    time.sleep(0.4)
    bounds = js("""
      const picker = document.getElementById('add-engines');
      const dialog = document.getElementById('add-dialog');
      const r = dialog.getBoundingClientRect();
      return {
        pickerScroll: picker.scrollWidth, pickerClient: picker.clientWidth,
        left: r.left, right: r.right, vw: window.innerWidth,
      };
    """)
    assert bounds["pickerScroll"] <= bounds["pickerClient"] + 1, bounds
    assert bounds["left"] >= 0 and bounds["right"] <= bounds["vw"], bounds
    shot("add-dialog-mobile.png")
    rect(1440, 1200)
    time.sleep(0.3)

    click("#add-cancel")
    wait("return !document.getElementById('add-dialog').open", message="add dialog closed")
    errors = js("return window.__uiErrors || []")
    assert not errors, errors
    print("PASS: engine management + two-column picker checks")
finally:
    try:
        wd("DELETE", "")
    except Exception:
        pass
    for proc in (driver, mock):
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in (driver, mock):
        try:
            proc.wait(5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
