#!/usr/bin/env python3
"""Firefox-headless UI test for the proxy and model-download management panels."""
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
OUT = ROOT / "tmp" / "ui-downloads"
MOCK = "http://127.0.0.1:18098/"
MOCK_LOG = OUT / "mock.log"
OUT.mkdir(parents=True, exist_ok=True)

mock = subprocess.Popen(
    [sys.executable, str(ROOT / "tests" / "ui-downloads-mock.py")],
    stdout=MOCK_LOG.open("w", encoding="utf-8"), stderr=subprocess.STDOUT,
)
for _ in range(60):
    try:
        requests.get(MOCK + "state", timeout=1)
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


def mock_state():
    return requests.get(MOCK + "state", timeout=5).json()


def set_value(element_id, value):
    js(f"""
      const el = document.getElementById({json.dumps(element_id)});
      el.value = {json.dumps(value)};
      el.dispatchEvent(new Event('input', {{bubbles: true}}));
      el.dispatchEvent(new Event('change', {{bubbles: true}}));
      return true;
    """)


def click(selector):
    js(f"""
      const el = document.querySelector({json.dumps(selector)});
      if (!el) throw new Error('missing ' + {json.dumps(selector)});
      el.click();
      return true;
    """)


try:
    rect(1440, 1200)
    wd("POST", "/url", {"url": MOCK})
    wait("return !document.getElementById('console-empty').hidden", message="console empty")
    js("""
      window.__uiErrors = [];
      window.addEventListener('error', e => window.__uiErrors.push(String(e.message)));
      window.addEventListener('unhandledrejection',
        e => window.__uiErrors.push(String(e.reason && e.reason.message || e.reason)));
      return true;
    """)

    click("#btn-downloads")
    wait("return !document.getElementById('download-popover').hidden",
         message="empty header popover")
    assert js("return document.getElementById('download-popover-list').textContent.trim()") == "没有下载"
    assert js("return document.getElementById('download-badge').hidden") is True
    shot("download-header-empty.png")
    click("#btn-downloads")
    wait("return document.getElementById('download-popover').hidden",
         message="header popover closed")

    click("#btn-manage")
    wait("return !document.getElementById('manage-view').hidden", message="manage view")
    click('.manage-menu .menu-item[data-panel="proxy"]')
    wait("return document.getElementById('proxy-state').textContent === '未启用'",
         message="proxy view")
    shot("proxy-empty-desktop.png")

    click("#proxy-enabled")
    set_value("proxy-type", "http")
    set_value("proxy-host", "127.0.0.1")
    set_value("proxy-port", "10809")
    set_value("proxy-user", "tester")
    set_value("proxy-pass", "ui-secret")
    assert js("""return !document.querySelector(
      '#proxy-sources input[data-proxy-source="huggingface"]').disabled""")
    click('#proxy-sources input[data-proxy-source="huggingface"]')
    assert js("""return document.querySelector(
      '#proxy-sources input[data-proxy-source="huggingface"]').checked""")
    click("#proxy-save")
    wait("return document.getElementById('proxy-state').textContent === '已启用'",
         message="proxy saved")
    wait("return document.getElementById('proxy-save').textContent === '保存代理'")
    assert mock_state()["sources"]["huggingface"] is True
    assert js("return document.getElementById('proxy-pass').value") == ""
    assert js("return document.getElementById('proxy-pass').placeholder").startswith("已保存")
    shot("proxy-saved-desktop.png")

    click("#proxy-test")
    wait("return !document.getElementById('proxy-result').hidden", message="proxy test result")
    text = js("return document.getElementById('proxy-result').textContent")
    assert "代理可连通" in text, text
    assert "HTTP 200" in text, text
    shot("proxy-tested-desktop.png")

    click('.manage-menu .menu-item[data-panel="downloads"]')
    wait("return document.querySelectorAll('#download-sources .source-filter').length === 3",
         message="download sources")
    assert js("""return [...document.querySelectorAll('#download-sources input')]
      .every(i => i.checked && !i.disabled)""")
    set_value("download-dir", "/srv/models")
    wait("return document.getElementById('download-dir-state').textContent === '已就绪'",
         message="download dir saved")

    set_value("download-query", "qwen3 gguf")
    click("#download-search")
    wait("return document.querySelectorAll('.download-card').length === 6",
         message="search results")
    assert mock_state()["search_sources"] == [
        "huggingface", "hf-mirror", "modelscope",
    ]
    stats = js("""
      const cards = [...document.querySelectorAll('.download-card')];
      return {
        cards: cards.length,
        badges: cards.filter(card => card.querySelector('.source-badge')).length,
        repos: cards.map(card => card.querySelector('.repo').textContent.trim()),
        heights: cards.map(card => card.querySelector('.repo').getBoundingClientRect().height),
        buttons: document.querySelectorAll('.download-card button[data-plan-source]').length,
        filters: [...document.querySelectorAll('#download-summary button')].map(b => b.textContent.trim()),
      };
    """)
    assert stats["badges"] == 0, stats
    assert min(stats["heights"]) >= 36, stats
    assert stats["buttons"] == 6, stats
    assert len(stats["filters"]) == 4, stats
    shot("downloads-desktop.png")

    click('#download-summary button[data-download-filter="modelscope"]')
    wait("return document.querySelectorAll('.download-card').length === 2",
         message="modelscope filter")
    click('#download-summary button[data-download-filter="all"]')
    wait("return document.querySelectorAll('.download-card').length === 6")

    click('#download-sources input[data-download-source="modelscope"]')
    assert js("""return !document.querySelector(
      '#download-sources input[data-download-source="modelscope"]').checked""")
    click("#download-search")
    wait("return document.querySelectorAll('.download-card').length === 4",
         message="filtered source search")
    assert mock_state()["search_sources"] == ["huggingface", "hf-mirror"]
    labels = js("""return [...document.querySelectorAll('.download-card .actions .hint')]
      .map(el => el.textContent.trim())""")
    assert set(labels) == {"Hugging Face", "HF-Mirror"}, labels
    click('#download-sources input[data-download-source="modelscope"]')
    click("#download-search")
    wait("return document.querySelectorAll('.download-card').length === 6",
         message="restored source search")

    rect(420, 1000)
    time.sleep(0.3)
    shot("downloads-mobile.png")
    rect(1440, 1200)
    time.sleep(0.3)

    click(".download-card button[data-plan-source]")
    wait("return document.getElementById('download-files-dialog').open",
         message="file dialog")
    wait("return document.querySelectorAll('#download-file-list .file-row').length === 3")
    wait("return document.getElementById('download-files-summary').textContent.includes('已选 2')",
         message="plan selection")
    checks = js("""
      const rows = [...document.querySelectorAll('#download-file-list .file-row')];
      return {
        existing: rows.filter(row => row.classList.contains('existing')).length,
        disabled: rows.filter(row => row.querySelector('input').disabled).length,
        startDisabled: document.getElementById('download-files-start').disabled,
        sub: document.getElementById('download-files-sub').textContent,
      };
    """)
    assert checks["existing"] == 1, checks
    assert checks["disabled"] == 1, checks
    assert checks["startDisabled"] is False, checks
    assert "/srv/models/" in checks["sub"], checks
    shot("download-dialog-desktop.png")

    click('#download-file-list input[data-file-path="config.json"]')
    wait("return document.getElementById('download-files-summary').textContent.includes('已选 1')")
    click('#download-file-list input[data-file-path="config.json"]')
    wait("return document.getElementById('download-files-summary').textContent.includes('已选 2')")
    click("#download-files-start")
    wait("return !document.getElementById('download-files-dialog').open",
         message="dialog closed after start")
    wait("return document.querySelector('#download-job .tag') && "
         "document.querySelector('#download-job .tag').textContent === '进行中'",
         message="job running")
    wait("return document.querySelector('#download-job .job-log')", message="job log")
    log_text = js("return document.querySelector('#download-job .job-log').textContent")
    assert "tokenizer.json" in log_text, log_text
    shot("download-job-desktop.png")

    # -- header download button: badge, popover and progress modal ------
    wait("return !document.getElementById('download-badge').hidden && "
         "document.getElementById('download-badge').textContent === '1'",
         message="header badge count")
    click("#btn-downloads")
    wait("return !document.getElementById('download-popover').hidden",
         message="header popover")
    wait("return document.querySelectorAll('.download-popover-item').length === 1",
         message="header popover item")
    popover = js("""return {
      repo: document.querySelector('.download-popover-item .repo').textContent,
      meta: document.querySelector('.download-popover-item .meta').textContent,
      expanded: document.getElementById('btn-downloads').getAttribute('aria-expanded'),
    }""")
    assert "Qwen/Qwen3-8B-GGUF" in popover["repo"], popover
    assert "个文件" in popover["meta"], popover
    assert popover["expanded"] == "true", popover
    shot("download-header-desktop.png")

    click(".download-popover-item")
    wait("return document.getElementById('download-progress-dialog').open",
         message="progress dialog")
    wait("return document.getElementById('download-progress-title')"
         ".textContent.includes('Qwen/Qwen3-8B-GGUF')",
         message="progress title")
    wait("return document.querySelector('#download-progress-job .job-log')",
         message="progress log")
    progress = js("""return {
      log: document.querySelector('#download-progress-job .job-log').textContent,
      stopHidden: document.getElementById('download-progress-cancel').hidden,
      status: document.querySelector('#download-progress-job .tag').textContent,
      popoverHidden: document.getElementById('download-popover').hidden,
    }""")
    assert "tokenizer.json" in progress["log"], progress
    assert progress["stopHidden"] is False, progress
    assert progress["status"] == "进行中", progress
    assert progress["popoverHidden"] is True, progress
    shot("download-progress-desktop.png")
    click("#download-progress-close")
    wait("return !document.getElementById('download-progress-dialog').open",
         message="progress dialog closed")

    click("#download-job button[data-download-cancel]")
    wait("return document.getElementById('confirm-dialog').open",
         message="cancel confirm")
    click("#confirm-ok")
    wait("return document.querySelector('#download-job .tag').textContent === '已取消'",
         message="job cancelled")
    wait("return document.getElementById('download-badge').hidden",
         message="header badge hidden")
    shot("download-job-cancelled-desktop.png")

    rect(420, 1000)
    time.sleep(0.3)
    shot("download-job-mobile.png")
    rect(1440, 1200)

    # -- two downloads at the same time ---------------------------------
    click(".download-card button[data-plan-source]")
    wait("return document.getElementById('download-files-dialog').open",
         message="first concurrent plan")
    wait("return document.getElementById('download-files-start').disabled === false")
    click("#download-files-start")
    wait("return !document.getElementById('download-files-dialog').open",
         message="first concurrent started")
    wait("return document.getElementById('download-badge').textContent === '1'",
         message="first concurrent badge")

    js("""
      const buttons = document.querySelectorAll('.download-card button[data-plan-source]');
      buttons[1].click();
      return true;
    """)
    wait("return document.getElementById('download-files-dialog').open",
         message="second concurrent plan")
    wait("return document.getElementById('download-files-start').disabled === false")
    click("#download-files-start")
    wait("return !document.getElementById('download-files-dialog').open",
         message="second concurrent started")
    wait("return document.getElementById('download-badge').textContent === '2'",
         message="two running downloads")

    click("#btn-downloads")
    wait("return !document.getElementById('download-popover').hidden")
    wait("return document.querySelectorAll('.download-popover-item').length === 2",
         message="two popover items")
    assert js("return document.getElementById('download-popover-count').textContent") == "进行中 2"
    shot("download-header-multi-desktop.png")

    rect(420, 1000)
    time.sleep(0.3)
    bounds = js("""const rect = document.getElementById('download-popover').getBoundingClientRect();
      return {left: rect.left, right: rect.right, width: rect.width, vw: window.innerWidth};""")
    assert bounds["left"] >= 0 and bounds["right"] <= bounds["vw"], bounds
    shot("download-header-multi-mobile.png")
    rect(1440, 1200)
    time.sleep(0.3)

    click(".download-popover-item")
    wait("return document.getElementById('download-progress-dialog').open")
    wait("return document.getElementById('download-progress-cancel').hidden === false")
    click("#download-progress-cancel")
    wait("return document.getElementById('confirm-dialog').open")
    click("#confirm-ok")
    wait("return document.querySelector('#download-progress-job .tag').textContent === '已取消'",
         message="first concurrent cancelled")
    wait("return document.getElementById('download-badge').textContent === '1'",
         message="badge after one concurrent cancel")
    click("#download-progress-close")
    wait("return !document.getElementById('download-progress-dialog').open")

    click("#btn-downloads")
    wait("return !document.getElementById('download-popover').hidden")
    wait("return document.querySelectorAll('.download-popover-item').length === 1",
         message="one popover item left")
    click(".download-popover-item")
    wait("return document.getElementById('download-progress-dialog').open")
    wait("return document.getElementById('download-progress-cancel').hidden === false")
    click("#download-progress-cancel")
    wait("return document.getElementById('confirm-dialog').open")
    click("#confirm-ok")
    wait("return document.querySelector('#download-progress-job .tag').textContent === '已取消'",
         message="second concurrent cancelled")
    wait("return document.getElementById('download-badge').hidden",
         message="badge hidden after all concurrent downloads")
    click("#download-progress-close")

    state = mock_state()
    assert state["proxy"]["enabled"] is True, state["proxy"]
    assert state["proxy"]["host"] == "127.0.0.1", state["proxy"]
    assert state["proxy"]["port"] == 10809, state["proxy"]
    assert state["proxy"]["password"] == "ui-secret", state["proxy"]
    assert state["test_calls"], state
    assert state["test_calls"][0]["host"] == "127.0.0.1", state["test_calls"]
    assert state["test_calls"][0]["port"] == "10809", state["test_calls"]
    assert state["dir"] == "/srv/models", state
    assert state["search_q"] == "qwen3 gguf", state
    assert state["sources"]["hf-mirror"] is False, state
    assert state["sources"]["huggingface"] is True, state
    assert state["sources"]["modelscope"] is False, state
    assert state["start"]["files"] == ["config.json", "sub/tokenizer.json"], state["start"]
    assert state["cancel"] is True, state
    assert len(state["starts"]) == 3, state["starts"]
    assert len(state["cancels"]) == 3, state["cancels"]
    assert all(job["status"] == "cancelled" for job in state["jobs"]), state["jobs"]
    errors = js("return window.__uiErrors || []")
    assert not errors, errors
    print("PASS: proxy + download UI checks")
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
