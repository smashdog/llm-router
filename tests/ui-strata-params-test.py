#!/usr/bin/env python3
"""Firefox-headless test for the Strata parameter tab.

Runs against the throwaway router configured by tmp/ui-strata/router.json
(127.0.0.1:18098, engine port 18099, auth disabled).  It opens the params
dialog of the copied gyro model, checks that the model-config fields are
rendered (drafting switch, expert tuning, user_args...), saves a set of
values through the real form and verifies the JSON on disk plus the values
read back from the API.  No real engine is started: the isolated router has
no live engine, so saving only writes the temp model config.
"""
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
OUT = ROOT / "tmp" / "ui-strata"
CONFIG = OUT / "models" / "strata-gyro-test.json"
UI = "http://127.0.0.1:18098/"
OUT.mkdir(parents=True, exist_ok=True)

try:
    requests.get(UI + "v1/models", timeout=3).raise_for_status()
except requests.RequestException as exc:
    raise SystemExit(f"isolated router is not running: {exc}")

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


def flag_value(args, flag):
    if flag not in args:
        return None
    index = args.index(flag)
    return args[index + 1] if index + 1 < len(args) else True


def model_entry():
    data = requests.get(UI + "v1/models", timeout=5).json()["data"]
    for item in data:
        if item["id"] == "gyro-test":
            return item
    raise SystemExit("gyro-test missing from /v1/models")


def save_closed():
    outcome = wait("""
      const dialog = document.getElementById('params-dialog');
      const error = document.getElementById('params-error');
      if (!dialog.open) return 'closed';
      if (!error.hidden && error.textContent.trim()) return 'error:' + error.textContent.trim();
      return false;
    """, 40, "params save")
    if outcome != "closed":
        raise SystemExit(f"save failed: {outcome}")


FIELDS = [
    "drafting", "mtp_path", "lookup_sizes", "coupled_draft", "spec",
    "draft_conf", "draft_cap", "pcie_frac", "adapt_every", "adapt_swaps",
    "resident_cpu_experts", "expert_profile_save", "expert_cache_device1",
    "peer_device", "pool_tasks", "session_save_reclaim", "vram_segment_mib",
    "split_skip_if_fits", "replicas", "user_args", "arch", "ple_io",
    "rope_freq_base", "vision_mmproj",
]

try:
    rect(1440, 1200)
    wd("POST", "/url", {"url": UI})
    wait("return !!document.querySelector('button[data-action=\"params\"][data-id=\"gyro-test\"]')",
         30, "model card params button")
    js("""
      window.__errs = [];
      window.addEventListener('error', (event) => window.__errs.push(String(event.message || event)));
      window.addEventListener('unhandledrejection',
        (event) => window.__errs.push('rejection: ' + String(event.reason)));
      return true;
    """)

    click('button[data-action="params"][data-id="gyro-test"]')
    wait("return document.getElementById('params-dialog').open === true", 10, "params dialog")
    click('[data-param-tab="more"]')
    wait("""return document.querySelector('[data-param-tab="more"]').getAttribute('aria-selected') === 'true'""",
         5, "more tab")

    info = js(f"""
      const keys = {json.dumps(FIELDS)};
      const host = document.getElementById('params-more');
      const missing = keys.filter((key) => !host.querySelector('[data-param-key="' + key + '"]'));
      const drafting = host.querySelector('select[data-param-key="drafting"]');
      return {{
        missing,
        count: host.querySelectorAll('[data-param-key]').length,
        options: drafting ? [...drafting.options].map((option) => option.value) : null,
        drafting: drafting ? drafting.value : null,
        visible: drafting ? !!(drafting.offsetWidth || drafting.offsetHeight || drafting.getClientRects().length) : false,
      }};
    """)
    print("fields", info)
    if info["missing"]:
        raise SystemExit(f"missing fields: {info['missing']}")
    if info["count"] < 90:
        raise SystemExit(f"too few strata fields rendered: {info['count']}")
    if info["drafting"] != "mtp":
        raise SystemExit(f"drafting should read the config value mtp, got {info['drafting']!r}")
    if not info["visible"]:
        raise SystemExit("drafting control not visible on the more tab")
    if sorted(info["options"] or []) != ["", "auto", "dflash2", "mtp", "off"]:
        raise SystemExit(f"unexpected drafting choices: {info['options']}")

    js("""
      const set = (key, value) => {
        const el = document.querySelector('#params-more [data-param-key="' + key + '"]');
        if (!el) throw new Error('missing ' + key);
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      };
      set('drafting', 'dflash2');
      set('spec', '6');
      set('draft_conf', '0.35');
      set('draft_cap', '6');
      set('pcie_frac', '0.4');
      set('adapt_every', '0');
      set('adapt_swaps', '80');
      set('replicas', '2');
      set('split_skip_if_fits', 'true');
      set('coupled_draft', 'true');
      set('session_save_reclaim', 'true');
      set('user_args', '--foo 1');
      return true;
    """)
    js("""
      const el = document.querySelector('#params-more [data-param-key="drafting"]');
      el.scrollIntoView({ block: 'center' });
      return true;
    """)
    time.sleep(0.4)
    shot("params-draft-before-save.png")

    click("#params-save")
    save_closed()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    args = [str(item) for item in config.get("args") or []]
    checks = {
        "drafting": (config.get("drafting"), "dflash2"),
        "mtp": (config.get("mtp"), True),
        "replicas": (config.get("replicas"), 2),
        "split_skip_if_fits": (config.get("split_skip_if_fits"), True),
        "arg --pcie-frac": (flag_value(args, "--pcie-frac"), "0.4"),
        "arg --adapt-every": (flag_value(args, "--adapt-every"), "0"),
        "arg --adapt-swaps": (flag_value(args, "--adapt-swaps"), "80"),
        "arg --coupled-draft": ("--coupled-draft" in args, True),
        "arg --session-save-reclaim": ("--session-save-reclaim" in args, True),
        "arg --spec removed": ("--spec" in args, False),
        "user_args": (config.get("user_args"),
                      ["--foo", "1", "--spec", "6", "--draft-conf", "0.35", "--draft-cap", "6"]),
        "args_written": (config.get("args_written"), args),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise SystemExit(f"config {name}: expected {expected!r}, got {actual!r}")

    entry = model_entry()
    read = entry.get("engine_params") or {}
    expected_read = {
        "drafting": "dflash2",
        "spec": 6,
        "draft_conf": 0.35,
        "draft_cap": 6,
        "pcie_frac": 0.4,
        "adapt_every": 0,
        "adapt_swaps": 80,
        "replicas": 2,
        "split_skip_if_fits": True,
        "coupled_draft": True,
        "session_save_reclaim": True,
        "user_args": "--foo 1 --spec 6 --draft-conf 0.35 --draft-cap 6",
    }
    for key, expected in expected_read.items():
        if read.get(key) != expected:
            raise SystemExit(f"read-back {key}: expected {expected!r}, got {read.get(key)!r}")

    # Reopen the dialog: the saved values must be rendered back into the form.
    click('button[data-action="params"][data-id="gyro-test"]')
    wait("return document.getElementById('params-dialog').open === true", 10, "params dialog reopen")
    click('[data-param-tab="more"]')
    wait("""return document.querySelector('[data-param-tab="more"]').getAttribute('aria-selected') === 'true'""",
         5, "more tab again")
    reopened = js("""
      const host = document.getElementById('params-more');
      const read = (key) => host.querySelector('[data-param-key="' + key + '"]').value;
      return {
        drafting: read('drafting'),
        spec: read('spec'),
        draft_conf: read('draft_conf'),
        user_args: read('user_args'),
        replicas: read('replicas'),
      };
    """)
    print("reopened", reopened)
    if reopened != {
        "drafting": "dflash2",
        "spec": "6",
        "draft_conf": "0.35",
        "user_args": "--foo 1 --spec 6 --draft-conf 0.35 --draft-cap 6",
        "replicas": "2",
    }:
        raise SystemExit(f"reopened form mismatch: {reopened}")
    js("""
      const el = document.querySelector('#params-more [data-param-key="drafting"]');
      el.scrollIntoView({ block: 'center' });
      return true;
    """)
    time.sleep(0.4)
    shot("params-draft-saved.png")

    # Switch drafting off and make sure mtp follows.
    js("""
      const el = document.querySelector('#params-more [data-param-key="drafting"]');
      el.value = 'off';
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    """)
    click("#params-save")
    save_closed()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("drafting") != "off" or config.get("mtp") is not False:
        raise SystemExit(f"drafting off not applied: {config.get('drafting')!r} / mtp {config.get('mtp')!r}")
    entry = model_entry()
    read = entry.get("engine_params") or {}
    if read.get("drafting") != "off":
        raise SystemExit(f"read-back drafting after off: {read.get('drafting')!r}")

    # Put the switch back to mtp so the temp copy stays usable until cleanup.
    click('button[data-action="params"][data-id="gyro-test"]')
    wait("return document.getElementById('params-dialog').open === true", 10, "params dialog third open")
    click('[data-param-tab="more"]')
    wait("""return document.querySelector('[data-param-tab="more"]').getAttribute('aria-selected') === 'true'""",
         5, "more tab third")
    js("""
      const el = document.querySelector('#params-more [data-param-key="drafting"]');
      el.value = 'mtp';
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    """)
    click("#params-save")
    save_closed()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("drafting") != "mtp" or config.get("mtp") is not True:
        raise SystemExit("restore to mtp failed")

    errors = js("return window.__errs || []")
    if errors:
        raise SystemExit(f"page errors: {errors}")
    print("ui strata params test passed")
finally:
    try:
        wd("DELETE", "")
    except Exception:  # noqa: BLE001
        pass
    # The snap-confined geckodriver may refuse signals; never fail a passed
    # test because of cleanup.
    try:
        driver.terminate()
    except Exception:  # noqa: BLE001
        pass
    try:
        driver.wait(5)
    except Exception:  # noqa: BLE001
        try:
            driver.kill()
        except Exception:  # noqa: BLE001
            # The Firefox snap runs geckodriver under an AppArmor profile that
            # refuses signals from outside; send the signal from inside it.
            if driver.poll() is None:
                try:
                    subprocess.run(
                        ["snap", "run", "--shell", "firefox", "-c",
                         f"kill -9 {driver.pid}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=20,
                    )
                except Exception:  # noqa: BLE001
                    pass
