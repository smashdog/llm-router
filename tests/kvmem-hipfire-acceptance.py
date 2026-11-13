#!/usr/bin/env python3
"""Real-artifact acceptance for the KVMem + hipfire engine (text + vision).

Run against the production router after the ROCm install produced
bin/kvmem-hipfire-server and bin/kvmem-hipfire-daemon.  Checks the engine
probe, registers the real HFQM model with its HFQ vision sidecar (CPU
host-mapped), loads it, runs text and image chats (non-streaming + streaming,
verifying the image inflates prompt tokens), refuses multi-image/remote-URL
requests, then unloads and stops the engine, verifying the daemon exits and
nothing stays resident.  Old engines must already be stopped and are not
restored.
"""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests

BASE = Path(__file__).resolve().parents[1]
URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL = Path("/mnt/ai/models/hipfire/Qwen3.8-27B/qwen3.8-27b.mq4-xts")
VISION = Path("/mnt/ai/models/hipfire/Qwen3.8-27B/qwen3.8-27b-vision.hfq")
NAME = "qwen3.8-27b-hipfire"
ENGINE = "kvmem_hipfire"
SERVER_BIN = BASE / "bin" / "kvmem-hipfire-server"
DAEMON_BIN = BASE / "bin" / "kvmem-hipfire-daemon"
IMAGE = BASE / "smoke.png"
IMAGE_B64 = base64.b64encode(IMAGE.read_bytes()).decode("ascii") if IMAGE.is_file() else ""
IMAGE_URL = f"data:image/png;base64,{IMAGE_B64}"
TEXT_MESSAGES = [{"role": "user", "content": "用一句话介绍你自己。"}]
IMAGE_MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "用一句话描述这张图片。"},
    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
]}]

KEY = ""
keys_file = BASE / "api-keys.json"
if keys_file.is_file():
    try:
        data = json.loads(keys_file.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            first = data[0]
            KEY = str(first.get("key") if isinstance(first, dict) else first or "")
    except (OSError, ValueError):
        KEY = ""
AUTH = {"Authorization": f"Bearer {KEY}"} if KEY else {}


def api(method, path, body=None, timeout=30):
    resp = requests.request(method, URL + path, json=body, timeout=timeout)
    return resp


def status():
    resp = api("GET", "/router/status")
    assert resp.status_code == 200, resp.text
    return resp.json()


def wait_for(predicate, timeout, message):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(2)
    raise AssertionError(f"timeout waiting for {message}")


def engine_view():
    return status()["engines"].get(ENGINE) or {}


def vram_free_ratio():
    ratios = []
    for card in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
        try:
            total = int(card.read_text())
            used = int((card.parent / "mem_info_vram_used").read_text())
        except (OSError, ValueError):
            continue
        if total > 0:
            ratios.append(1.0 - used / total)
    return max(ratios) if ratios else 1.0


def engine_pids():
    """Exact argv[0] matches for the managed CLI and daemon binaries."""
    out = subprocess.run(
        ["ps", "-eo", "pid,args"], capture_output=True, text=True, check=True
    ).stdout
    needles = {str(SERVER_BIN), str(DAEMON_BIN)}
    pids = []
    for line in out.splitlines()[1:]:
        pid, _, args = line.strip().partition(" ")
        tokens = args.split()
        if tokens and tokens[0] in needles:
            pids.append(int(pid))
    return pids


def completion(messages, stream):
    """One chat through the router; returns (content, prompt_tokens).

    prompt_tokens comes from the non-streaming usage block (authoritative for
    the image-vs-text growth check); a streaming reply also captures usage when
    the engine emits it, but the growth assertion uses the non-streaming count.
    """
    body = {
        "model": NAME,
        "messages": messages,
        "max_tokens": 96,
        "temperature": 0.0,
        "stream": stream,
    }
    if not stream:
        resp = requests.post(
            URL + "/v1/chat/completions", json=body, headers=AUTH, timeout=300
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        content = str(data["choices"][0]["message"]["content"] or "")
        assert content.strip(), data
        usage = data.get("usage") or {}
        assert int(usage.get("completion_tokens") or 0) > 0, data
        return content, int(usage.get("prompt_tokens") or 0)
    pieces = []
    done = False
    prompt_tokens = 0
    with requests.post(
        URL + "/v1/chat/completions", json=body, headers=AUTH,
        stream=True, timeout=(30, 300),
    ) as resp:
        assert resp.status_code == 200, resp.text
        # Decode SSE as UTF-8 bytes: iter_lines(decode_unicode=True) would use
        # the response's ISO-8859-1 default and mojibake the Chinese content.
        for raw in resp.iter_lines(decode_unicode=False):
            if not raw or not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            obj = json.loads(payload)
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    pieces.append(str(delta["content"]))
            usage = obj.get("usage")
            if isinstance(usage, dict) and usage.get("prompt_tokens"):
                prompt_tokens = int(usage["prompt_tokens"])
    content = "".join(pieces)
    assert content.strip(), "streaming chat produced no content"
    print(f"  stream: {len(pieces)} chunks, [DONE]={done}")
    return content, prompt_tokens


def main():
    checks = 0
    assert SERVER_BIN.is_file(), f"missing {SERVER_BIN}; run the ROCm install first"
    assert DAEMON_BIN.is_file(), f"missing {DAEMON_BIN}; run the ROCm install first"
    assert MODEL.is_file(), f"missing acceptance model {MODEL}"
    assert VISION.is_file(), f"missing HFQ vision sidecar {VISION}"
    assert IMAGE.is_file() and IMAGE_B64, f"missing acceptance image {IMAGE}"

    # 1. The router runs the current code and the hipfire slot is configured.
    view = engine_view()
    assert view, "kvmem_hipfire slot missing; restart the router with the current code"
    assert view.get("alive") is False, f"hipfire engine already running: {view}"
    st = status()
    slots = set(st["engines"])
    assert "kvmem_polystrata" in slots, f"startup migration missing: {sorted(slots)}"
    assert not slots & {"kvmem", "ninfer"}, f"legacy engine ids still live: {sorted(slots)}"
    for old in ("kvmem_llama", "strata"):
        slot = st["engines"].get(old) or {}
        assert slot.get("alive") is False, f"stop {old} before acceptance (no restore)"
    assert st.get("active_model") is None, st
    assert not engine_pids(), f"leftover hipfire processes: {engine_pids()}"
    free = vram_free_ratio()
    assert free >= 0.9, f"VRAM not free enough for acceptance: {free:.1%}"
    checks += 1
    print(f"ok: slot ready, old engines stopped, VRAM {free:.1%} free")

    # 2. Console probe: re-saving the engine paths runs `hipfire --version`.
    paths_view = api("GET", "/router/engines/paths").json()
    item = next((e for e in paths_view["engines"] if e["engine"] == ENGINE), None)
    assert item and item["configured"], f"hipfire slot not configured: {item}"
    resp = api("POST", "/router/engines/paths", {"engine": ENGINE, "paths": item["fields"]})
    assert resp.status_code == 200, resp.text
    checks += 1
    print("ok: engine probe (--version) passed")

    # 3. Register the real HFQM model (validation only; load is explicit).
    # Idempotent: a prior partial run may have left the registration behind
    # (registration survives a failed load/chat), so clear it before adding.
    existing = api("GET", "/models").json().get("data", [])
    if any(m.get("id") == NAME for m in existing):
        resp = api("POST", "/router/models/delete", {"model": NAME})
        assert resp.status_code == 200, resp.text
    resp = api("POST", "/router/models/add", {
        "engine": ENGINE, "model": str(MODEL), "name": NAME,
        "vision": str(VISION), "vision_device": "cpu", "smoke": False,
    })
    assert resp.status_code == 202, resp.text
    job = resp.json()
    deadline = time.time() + 300
    while True:
        view_job = api("GET", f"/router/models/job?id={job['id']}").json()
        if view_job.get("status") != "running":
            break
        assert time.time() < deadline, f"add job stuck: {view_job}"
        time.sleep(2)
    assert view_job.get("status") == "ok", view_job
    resp = api("GET", "/models")
    assert resp.status_code == 200, resp.text
    listed = next((m for m in resp.json().get("data", []) if m.get("id") == NAME), None)
    assert listed and listed.get("engine") == ENGINE, listed
    assert listed.get("multimodal") is True, listed
    assert listed.get("vision_max_images") == 1, listed
    assert listed.get("vision_gpu") is False, listed
    assert listed.get("vision_path") == str(VISION), listed
    checks += 1
    print(f"ok: model {NAME} registered with CPU vision sidecar")

    # 4. Load the model on the real engine.
    resp = api("POST", "/models/load", {"model": NAME})
    assert resp.status_code in (200, 202), resp.text
    try:
        wait_for(
            lambda: engine_view().get("alive") is True and status().get("active_model") == NAME,
            900, "hipfire engine load",
        )
    except AssertionError:
        log = BASE / "log" / "kvmem-hipfire-engine.log"
        if log.is_file():
            print(log.read_text(encoding="utf-8", errors="replace")[-2000:])
        raise
    checks += 1
    print("ok: model loaded, engine alive")

    # 5. Text and image chats, each non-streaming + streaming. Greedy (temp 0)
    # decoding makes the streamed reply match the non-streamed one byte-for-byte
    # (proving the SSE path delivers correct UTF-8), and the image must inflate
    # prompt tokens versus the text-only turn (proving the vision tower ran).
    plain_text, text_tokens = completion(TEXT_MESSAGES, stream=False)
    print(f"  text: {plain_text[:80]!r} (prompt {text_tokens} tokens)")
    streamed_text, _ = completion(TEXT_MESSAGES, stream=True)
    assert streamed_text == plain_text, (streamed_text, plain_text)
    plain_img, img_tokens = completion(IMAGE_MESSAGES, stream=False)
    print(f"  image: {plain_img[:80]!r} (prompt {img_tokens} tokens)")
    streamed_img, _ = completion(IMAGE_MESSAGES, stream=True)
    assert streamed_img == plain_img, (streamed_img, plain_img)
    assert img_tokens > text_tokens, (img_tokens, text_tokens)
    checks += 1
    print(f"ok: text + image chat (stream == plain; image prompt {text_tokens} -> {img_tokens})")

    # 5b. Single-image contract: multiple images and remote URLs are refused
    # with a clear 400 (the engine only loads one base64 data URI per request).
    multi = requests.post(URL + "/v1/chat/completions", json={
        "model": NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "描述这两张图片。"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ]}],
        "max_tokens": 32,
    }, headers=AUTH, timeout=60)
    assert multi.status_code == 400 and "只支持一张图片" in multi.text, multi.text
    remote = requests.post(URL + "/v1/chat/completions", json={
        "model": NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "描述这张图片。"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]}],
        "max_tokens": 32,
    }, headers=AUTH, timeout=60)
    assert remote.status_code == 400 and "base64 内嵌图片" in remote.text, remote.text
    checks += 1
    print("ok: multi-image and remote-URL requests refused (single base64 image only)")

    # 6. Unload, stop, and verify nothing stays resident.
    resp = api("POST", "/models/unload", {"model": NAME})
    assert resp.status_code in (200, 202), resp.text
    wait_for(
        lambda: engine_view().get("alive") is False and status().get("active_model") is None,
        120, "hipfire engine stop",
    )
    deadline = time.time() + 15
    while time.time() < deadline and engine_pids():
        time.sleep(0.5)
    assert not engine_pids(), f"hipfire processes survived the stop: {engine_pids()}"
    wait_for(lambda: vram_free_ratio() >= 0.9, 60, "VRAM release")
    checks += 1
    print("ok: engine stopped, daemon exited, VRAM released; nothing resident")

    print(f"kvmem-hipfire acceptance: {checks} check groups passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"ACCEPTANCE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)