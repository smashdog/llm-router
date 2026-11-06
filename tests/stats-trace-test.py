#!/usr/bin/env python3
"""Unit checks for ProxyTrace, which feeds the console statistics card.

Runs directly against router.py with no engine or HTTP server: streaming SSE
and non-streaming bodies are parsed in-memory and the recorded stat fields are
compared with the reference numbers (25,038 prompt / 23,387 cached, etc.).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import router  # noqa: E402


class FakeGateway:
    def __init__(self) -> None:
        self.stats = router.RequestStats()


entry = SimpleNamespace(id="ornith", engine="kvmem", context_length=262144)
checks = 0


def new_trace() -> tuple[FakeGateway, router.ProxyTrace]:
    gateway = FakeGateway()
    return gateway, router.ProxyTrace(gateway, entry, time.monotonic())


def feed_sse(trace: router.ProxyTrace, events: list[dict], split: bool = True) -> None:
    payload = b"".join(
        b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n"
        for event in events
    ) + b"data: [DONE]\n\n"
    if not split:
        trace.feed(payload)
        return
    # Split into irregular chunks, including mid-line boundaries.
    sizes = [7, 41, 3, 128, 19]
    index = 0
    step = 0
    while index < len(payload):
        size = sizes[step % len(sizes)]
        trace.feed(payload[index:index + size])
        index += size
        step += 1


llama_usage = {
    "prompt_tokens": 25038,
    "completion_tokens": 1827,
    "total_tokens": 26865,
    "prompt_tokens_details": {"cached_tokens": 23387},
}
llama_timings = {
    "prompt_n": 1651,
    "prompt_ms": 1220.0,
    "prompt_per_second": 1356.4,
    "predicted_n": 1827,
    "predicted_ms": 24000.0,
    "predicted_per_second": 79.9,
    "cache_n": 23387,
}

# --- streaming llama.cpp / KVMem SSE -------------------------------------
gateway, trace = new_trace()
feed_sse(trace, [
    {"choices": [{"index": 0, "delta": {"content": "你"}}]},
    {"choices": [{"index": 0, "delta": {"content": "好"}}]},
    {"choices": [{"index": 0, "delta": {}}], "usage": llama_usage, "timings": llama_timings},
])
trace.flush()
trace.commit()
last = gateway.stats.view()["last"]
assert last["model"] == "ornith" and last["engine"] == "kvmem", last
assert last["prompt_tokens"] == 25038, last
assert last["completion_tokens"] == 1827, last
assert last["total_tokens"] == 26865, last
assert last["cached_tokens"] == 23387, last
assert last["context_length"] == 262144, last
assert last["gen_per_second"] == 79.9, last
assert last["prefill_per_second"] == 1356.4, last
assert last["prefill_ms"] == 1220.0, last
assert last["first_token_ms"] is not None and last["first_token_ms"] >= 0, last
assert last["draft_n"] is None, last
checks += 1

# --- timings only (KVMem SSE without usage) ------------------------------
gateway, trace = new_trace()
feed_sse(trace, [
    {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
    {"choices": [{"index": 0, "delta": {}}], "timings": llama_timings},
])
trace.flush()
trace.commit()
last = gateway.stats.view()["last"]
assert last["prompt_tokens"] == 25038 and last["completion_tokens"] == 1827, last
assert last["cached_tokens"] == 23387, last
assert last["gen_per_second"] == 79.9, last
checks += 1

# --- non-streaming Anthropic message --------------------------------------
gateway, trace = new_trace()
trace.feed_json(json.dumps({
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "hello"}],
    "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 60},
}).encode())
trace.commit()
last = gateway.stats.view()["last"]
assert last["prompt_tokens"] == 100 and last["completion_tokens"] == 10, last
assert last["cached_tokens"] == 60, last
assert last["gen_per_second"] is None and last["prefill_per_second"] is None, last
checks += 1

# --- non-streaming OpenAI completion, no usage ----------------------------
gateway, trace = new_trace()
trace.feed_json(json.dumps({
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
}).encode())
trace.commit()
assert gateway.stats.view() == {}, gateway.stats.view()
checks += 1

# --- rate fallback from counters when *_per_second is absent --------------
gateway, trace = new_trace()
feed_sse(trace, [
    {"choices": [{"index": 0, "delta": {"content": "x"}}]},
    {"choices": [{"index": 0, "delta": {}}], "timings": {
        "prompt_n": 1000, "prompt_ms": 500.0,
        "predicted_n": 500, "predicted_ms": 2500.0,
    }},
], split=False)
trace.flush()
trace.commit()
last = gateway.stats.view()["last"]
assert last["prefill_per_second"] == 2000.0, last
assert last["gen_per_second"] == 200.0, last
checks += 1

print(f"PASS {checks} checks")
