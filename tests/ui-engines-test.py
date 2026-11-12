#!/usr/bin/env python3
"""Firefox-headless UI test for the engine provisioning page and two-column engine picker."""
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


def mock_state():
    return requests.get(MOCK + "mock/state", timeout=5).json()


def mock_config(**values):
    return requests.post(MOCK + "mock/config", json=values, timeout=5).json()


def wait_mock(predicate, timeout=20, message=""):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = mock_state()
        if predicate(last):
            return last
        time.sleep(0.15)
    raise SystemExit(f"wait failed: {message}: {last}")


def card(index):
    return f"document.querySelectorAll('#engine-install-list .provision-engine')[{index}]"


def click_recompile(index):
    js(f"""
      const node = {card(index)};
      const btn = [...node.querySelectorAll(
        'button[data-setup-action="engine"][data-op="install"]')]
        .find(b => !b.dataset.force);
      if (!btn) throw new Error('missing recompile button');
      btn.click();
      return true;
    """)


def reenter_manage():
    click("#btn-manage")
    wait("return document.getElementById('manage-view').hidden", message="console view")
    click("#btn-manage")
    wait("return !document.getElementById('manage-view').hidden", message="manage view")
    wait("return document.querySelectorAll('#engine-install-list .provision-engine').length === 3",
         message="setup cards")


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

    # -- engine provisioning page ---------------------------------------
    click("#btn-manage")
    wait("return !document.getElementById('manage-view').hidden", message="manage view")
    wait("return document.querySelectorAll('#engine-install-list .provision-engine').length === 3",
         message="setup cards")

    assert js("""return document.querySelector(
      '.manage-menu .menu-item[data-panel="engines"]').textContent.trim()""") == "引擎"
    assert js("return document.getElementById('engine-paths-title').textContent.trim()") == "引擎"

    cards = js("""
      return [...document.querySelectorAll('#engine-install-list .provision-engine')].map(node => ({
        engine: node.querySelector('button[data-setup-action="engine"][data-op="update"]').dataset.engine,
        label: node.querySelector('.provision-engine-head h4').textContent.trim(),
        tag: node.querySelector('.provision-engine-head .tag').textContent.trim(),
        bins: [...node.querySelectorAll('.provision-bin')].map(b => b.textContent.trim()),
      }));
    """)
    assert [c["engine"] for c in cards] == ["llama", "strata", "kvmem"], cards
    assert [c["label"] for c in cards] == ["llama.cpp", "Strata", "KVMem"], cards
    assert [c["tag"] for c in cards] == ["已编译", "已编译", "已编译"], cards
    assert cards[2]["bins"] == ["llama-kvmem-server", "strata-poly", "ninfer-serve"], cards

    toolchains = js("""
      return [...document.querySelectorAll('#toolchain-rows .toolchain-row')].map(node => ({
        label: node.querySelector('h4').textContent.trim(),
        tag: node.querySelector('.tag').textContent.trim(),
      }));
    """)
    assert toolchains == [{"label": "ROCm", "tag": "ROCm 6.2.0 不满足"}], toolchains
    assert js("""return !!document.querySelector('select[data-toolchain-version="rocm"]')""")
    shot("setup-desktop.png")

    # -- setup proxy switches persist across reloads ---------------------
    assert js("""return !!document.querySelector('input[data-setup-proxy="engine:kvmem"]')""")
    assert not js("""return document.querySelector('input[data-setup-proxy="engine:kvmem"]').checked""")
    click('input[data-setup-proxy="engine:kvmem"]')
    state = wait_mock(lambda s: s["last"]["proxy"] == {"key": "engine:kvmem", "enabled": True},
                      message="proxy save")
    assert state["setup_proxy"]["engine:kvmem"] is True, state
    click('input[data-setup-proxy="toolchain:rocm"]')
    wait_mock(lambda s: s["last"]["proxy"] == {"key": "toolchain:rocm", "enabled": True},
              message="toolchain proxy save")
    reenter_manage()
    assert js("""return document.querySelector('input[data-setup-proxy="engine:kvmem"]').checked""")
    assert js("""return document.querySelector('input[data-setup-proxy="toolchain:rocm"]').checked""")

    # -- engine rebuild job runs to completion ---------------------------
    click_recompile(0)
    wait(f"return !!{card(0)}.querySelector('.provision-step.running')",
         message="job running")
    wait(f"return {card(0)}.querySelectorAll('.provision-step.ok').length >= 2",
         timeout=30, message="job finished")
    state = mock_state()
    assert state["last"]["install"] == {"engine": "llama", "use_proxy": False}, state
    assert state["last"]["install_op"] == "install", state
    assert any(job["status"] == "ok" for job in state["jobs"].values()), state
    shot("setup-install-ok.png")

    # -- failed build surfaces the error on the card ---------------------
    mock_config(install_fails=True)
    click_recompile(2)
    wait(f"""return ({card(2)}.querySelector('.provision-warn') || {{textContent: ''}})
      .textContent.includes('编译失败')""", timeout=30, message="job error")
    text = js(f"return {card(2)}.textContent")
    assert "已恢复原二进制" in text and "已回滚源码" in text, text
    state = mock_state()
    assert any(job["status"] == "error" for job in state["jobs"].values()), state
    shot("setup-install-error.png")

    # -- no proxy configured: switches disappear -------------------------
    mock_config(proxy_configured=False, install_fails=False)
    reenter_manage()
    # loadSetup() re-fetches asynchronously and the stale DOM already has
    # three cards, so wait for the re-render instead of asserting at once.
    wait("return document.querySelectorAll('input[data-setup-proxy]').length === 0",
         message="proxy switches removed")

    # -- two-column engine picker in the add dialog ----------------------
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
    assert [item["value"] for item in subs] == ["kvmem_llama", "kvmem_polystrata", "kvmem_ninfer"], subs
    assert subs[0]["checked"] and subs[0]["on"], subs
    assert not subs[1]["checked"] and not subs[2]["checked"], subs
    assert js("return document.querySelector('#add-engines .engine-cat.on').textContent.trim()") == "kvmem"
    assert "llama-kvmem-server" in js("return document.getElementById('add-more-caption').textContent")

    click('#add-engine-subs input[value="kvmem_polystrata"]')
    wait("return document.querySelector('#add-engine-subs input[value=\"kvmem_polystrata\"]').checked",
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
    print("PASS: engine provisioning + two-column picker checks")
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
