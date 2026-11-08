#!/usr/bin/env python3
"""PolyStrata answer-behavior fields in the router parameter form.

Checks that the serve-level keys (tool_call_recovery, reasoning_close_retry,
reasoning_loop_recovery, ...) are in the KVMem engine schema, are read from the
top level of strata-*.json, are validated, and are written back there instead
of argv.  No engine is started and no real config is touched.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "router.py"
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("llm_router", MODULE)
router = importlib.util.module_from_spec(spec)
sys.modules["llm_router"] = router
spec.loader.exec_module(router)

BASE = {
    "exe": "bin/kvmem/strata-poly",
    "cwd": "/opt/kvmem/backends/polystrata",
    "model_name": "test-model",
    "tokenizer": "/tmp/tokenizer",
    "args": [
        "--model", "/models/test.gguf",
        "--max-context", "32768",
        "--kvmem-budget", "8192",
        "--kvmem-gen-reserve", "4096",
        "--vram-reserve-mib", "256",
    ],
    "vision": {"exe": "bin/strata-vision", "threads": 8, "max_tokens": 2048},
    "tool_call_recovery": True,
    "reasoning_close_retry": False,
    "reasoning_loop_recovery": "stop",
    "reasoning_budget_tokens": 4096,
    "preserve_thinking": True,
    "client_reasoning_history": False,
    "reasoning_effort": "medium",
    "anthropic_thinking": "on_request",
    "effort_position": "end",
    "fit_max_tokens": True,
    "drop_oldest": False,
    "repeat_stop_tokens": 0,
    "warm_start": True,
    "codex_thread_titles": True,
    "codex_compaction_cache": False,
    "answer_room": False,
    "kvmem_default_max_tokens": 65536,
    "api_monitor": True,
    "engine_silence_s": 120,
    "memory_debug": False,
    "env_to_user": True,
}

BEHAVIOR_KEYS = [
    "tool_call_recovery", "reasoning_close_retry", "reasoning_loop_recovery",
    "reasoning_budget_tokens", "preserve_thinking", "client_reasoning_history",
    "reasoning_effort", "anthropic_thinking", "effort_position", "fit_max_tokens",
    "drop_oldest", "repeat_stop_tokens", "warm_start", "codex_thread_titles",
    "codex_compaction_cache", "answer_room", "kvmem_default_max_tokens",
    "api_monitor", "engine_silence_s", "memory_debug", "env_to_user",
]

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("ok   " if condition else "FAIL ") + name + (f" :: {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# -- schema ---------------------------------------------------------------
schema = {row["key"]: row for row in router.kvmem_poly_engine_schema()}
for key in BEHAVIOR_KEYS:
    check(f"schema has {key}", key in schema)
    check(f"{key} is a config field", schema[key].get("config_path") == [key])
    check(f"{key} is in the behavior group", schema[key].get("group") == "behavior")
check("behavior group has a label", schema["tool_call_recovery"].get("group_label") == "回答行为")
check("tool_call_recovery is boolean", schema["tool_call_recovery"].get("type") == "boolean")
check("reasoning_loop_recovery has choices",
      [choice["value"] for choice in schema["reasoning_loop_recovery"].get("choices") or []]
      == ["stop", "recover"])
check("reasoning_budget_tokens allows 0", schema["reasoning_budget_tokens"].get("min") == 0)
check("kvmem_default_max_tokens needs at least 1", schema["kvmem_default_max_tokens"].get("min") == 1)

# -- read ----------------------------------------------------------------
values = router.kvmem_poly_engine_values(list(BASE["args"]), BASE)
for key in BEHAVIOR_KEYS:
    check(f"read {key}", values.get(key) == BASE[key], f"got {values.get(key)!r}")
check("explicit false is kept", values.get("reasoning_close_retry") is False)
check("no behavior key leaks into args", all(flag not in BASE["args"] for flag in BEHAVIOR_KEYS))

empty = copy.deepcopy(BASE)
for key in BEHAVIOR_KEYS:
    empty.pop(key)
absent = router.kvmem_poly_engine_values(list(empty["args"]), empty)
check("absent keys stay empty", all(key not in absent for key in BEHAVIOR_KEYS), repr(absent))

legacy = copy.deepcopy(BASE)
legacy["reasoning_loop_recovery"] = True
check("legacy true reads as recover",
      router.kvmem_poly_engine_values(list(legacy["args"]), legacy).get("reasoning_loop_recovery") == "recover")

# -- validation ----------------------------------------------------------
normalized = router.normalize_kvmem_poly_engine(
    {
        "tool_call_recovery": False,
        "reasoning_close_retry": True,
        "reasoning_loop_recovery": "recover",
        "reasoning_budget_tokens": 0,
        "reasoning_effort": "high",
        "anthropic_thinking": "model",
        "effort_position": "start",
        "fit_max_tokens": True,
        "kvmem_default_max_tokens": 4096,
        "engine_silence_s": 0,
    }
)
check("false normalized", normalized.get("tool_call_recovery") is False)
check("enum normalized", normalized.get("reasoning_loop_recovery") == "recover")
check("zero budget normalized", normalized.get("reasoning_budget_tokens") == 0)
check("cleared value normalized", router.normalize_kvmem_poly_engine({"drop_oldest": None}).get("drop_oldest") is None)

for bad in (
    {"reasoning_loop_recovery": "loop"},
    {"reasoning_effort": "max"},
    {"anthropic_thinking": "always"},
    {"effort_position": "middle"},
    {"reasoning_budget_tokens": -1},
    {"kvmem_default_max_tokens": 0},
    {"engine_silence_s": -5},
    {"tool_call_recovery": "yes"},
):
    try:
        router.normalize_kvmem_poly_engine(bad)
    except router.BadRequest:
        check(f"rejects {bad}", True)
    else:
        check(f"rejects {bad}", False, "accepted")

# -- write back ----------------------------------------------------------
config = copy.deepcopy(BASE)
changes = {
    "tool_call_recovery": False,
    "reasoning_close_retry": True,
    "reasoning_loop_recovery": "recover",
    "reasoning_budget_tokens": 0,
    "preserve_thinking": None,
    "reasoning_effort": "low",
    "anthropic_thinking": "model",
    "effort_position": "start",
    "fit_max_tokens": False,
    "drop_oldest": True,
    "repeat_stop_tokens": 12,
    "warm_start": False,
    "codex_thread_titles": False,
    "codex_compaction_cache": True,
    "answer_room": True,
    "kvmem_default_max_tokens": 8192,
    "api_monitor": False,
    "engine_silence_s": 30,
    "memory_debug": True,
    "env_to_user": False,
}
router.kvmem_poly_apply_config_values(config, changes)
check("tool_call_recovery written false", config.get("tool_call_recovery") is False)
check("reasoning_close_retry written true", config.get("reasoning_close_retry") is True)
check("enum written", config.get("reasoning_loop_recovery") == "recover")
check("preserve_thinking cleared", "preserve_thinking" not in config)
check("repeat_stop_tokens written", config.get("repeat_stop_tokens") == 12)
check("kvmem_default_max_tokens written", config.get("kvmem_default_max_tokens") == 8192)
check("engine_silence_s written", config.get("engine_silence_s") == 30)
check("vision untouched", config.get("vision") == BASE["vision"])
check("args untouched", config.get("args") == BASE["args"])

again = router.kvmem_poly_engine_values(config["args"], config)
for key, want in changes.items():
    if want is None:
        check(f"round trip cleared {key}", key not in again, f"got {again.get(key)!r}")
    else:
        check(f"round trip {key}", again.get(key) == want, f"got {again.get(key)!r}")

# apply_engine_values must leave the top-level keys to apply_config_values.
args_after = router.kvmem_poly_apply_engine_values(list(BASE["args"]), changes)
check("behavior fields never reach argv",
      all(f"--{key.replace('_', '-')}" not in args_after for key in changes), repr(args_after))
check("ordinary engine flags survive",
      router._config_arg_value(args_after, "--kvmem-budget") == "8192")

check("base template unchanged",
      BASE["tool_call_recovery"] is True and BASE["reasoning_close_retry"] is False)

print()
if failures:
    print(f"{len(failures)} failure(s)")
    sys.exit(1)
print("all checks passed")
