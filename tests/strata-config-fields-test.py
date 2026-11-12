#!/usr/bin/env python3
"""Strata model-config fields in the router parameter form.

Checks, against a copy of the local gyro config, that the new config-level
fields are read (drafting effective value, list fields), validated, and written
back: config keys (drafting/arch/layer_split/replicas/vision.mmproj), the
launcher-owned drafter flags in user_args, and ordinary engine flags in args.
No engine is started and no real config is touched.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODULE = ROOT / "router.py"

spec = importlib.util.spec_from_file_location("llm_router", MODULE)
router = importlib.util.module_from_spec(spec)
sys.modules["llm_router"] = router
spec.loader.exec_module(router)

BASE = {
    "exe": "build-hip/strata",
    "args": [
        "--pack", "packs/gyro-l",
        "--native", "/mnt/ai/models/model.gguf",
        "--expert-profile", "data/expert-profile.bin",
        "--expert-cache", "auto",
        "--prefill", "4096",
        "--spec", "4",
        "--spec-min-p", "0.5",
        "--mtp", "/mnt/ai/models/mtp-draft.rt",
        "--max-context", "131072",
        "--kv", "int8",
        "--kv-resident", "32768",
        "--pool-workers", "7",
        "--vram-reserve-mib", "512",
        "--vision",
    ],
    "args_written": [],
    "tokenizer": "packs/gyro-l/tokenizer",
    "arch": "qwen4exp",
    "model_name": "gyro-l-abliterated",
    "drafting": "mtp",
    "mtp": True,
    "vision": {
        "exe": "/mnt/ai/llm/Strata/build-vision/bin/strata-vision",
        "mmproj": "/mnt/ai/models/mmproj-F16.gguf",
        "model": "/mnt/ai/models/model.gguf",
        "gpu": False,
        "max_tokens": 300,
        "threads": 8,
    },
}
BASE["args_written"] = list(BASE["args"])

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("ok   " if condition else "FAIL ") + name + (f" :: {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# -- schema ---------------------------------------------------------------
schema = {row["key"]: row for row in router.strata_engine_schema()}
for key in (
    "drafting", "mtp_path", "mtp_draft_vocab", "lookup_sizes", "coupled_draft",
    "arch", "ple_io", "ple_row_cache", "pcie_frac", "adapt_every", "adapt_swaps",
    "pool_tasks", "pool_affinity", "aux_cpus", "pipeline_windows",
    "prefill_head_alias", "session_min_free_mib", "session_save_reclaim",
    "vram_segment_mib", "layer_split", "split_skip_if_fits", "replicas",
    "user_args", "vision_mmproj", "rope_freq_base", "rope_freq_scale",
    "yarn_orig_ctx", "peer_device", "expert_cache_device1",
):
    check(f"schema has {key}", key in schema)
check("drafting is a config field", schema["drafting"].get("config_path") == ["drafting"])
check("spec is stored in user_args", schema["spec"].get("store") == "user_args")
check("draft_conf is stored in user_args", schema["draft_conf"].get("store") == "user_args")
check("vision_mmproj is vision-only", bool(schema["vision_mmproj"].get("vision_only")))

# -- read ----------------------------------------------------------------
values = router.strata_engine_values(list(BASE["args"]), BASE)
check("drafting read", values.get("drafting") == "mtp", repr(values.get("drafting")))
check("arch read", values.get("arch") == "qwen4exp")
check("spec read from args", values.get("spec") == 4)
check("mtp path read", values.get("mtp_path") == "/mnt/ai/models/mtp-draft.rt")
check("vision mmproj read", values.get("vision_mmproj") == "/mnt/ai/models/mmproj-F16.gguf")
check("absent replicas stays empty", "replicas" not in values)

legacy = copy.deepcopy(BASE)
legacy.pop("drafting")
legacy["mtp"] = False
check("mtp false without drafting reads off",
      router.strata_engine_values(legacy["args"], legacy).get("drafting") == "off")
legacy2 = copy.deepcopy(BASE)
legacy2.pop("drafting")
legacy2.pop("mtp")
check("no drafting/mtp reads auto",
      router.strata_engine_values(legacy2["args"], legacy2).get("drafting") == "auto")

# user_args list is shown as a command line
with_user = copy.deepcopy(BASE)
with_user["user_args"] = ["--control-vector-scaled", "/x.gguf:1.2", "--cvec-mode", "add"]
check("user_args list rendered as text",
      router.strata_engine_values(with_user["args"], with_user).get("user_args")
      == "--control-vector-scaled /x.gguf:1.2 --cvec-mode add")

# -- validation ----------------------------------------------------------
normalized = router.normalize_strata_engine(
    {
        "drafting": "dflash2",
        "arch": "deepseek41",
        "replicas": 2,
        "layer_split": "24,36",
        "user_args": "--foo 1 --bar on",
        "lookup_sizes": "5,6,12",
        "pcie_frac": 0.35,
        "adapt_every": 0,
    }
)
check("drafting normalized", normalized.get("drafting") == "dflash2")
check("replicas normalized", normalized.get("replicas") == 2)
for bad in (
    {"drafting": "df2"},
    {"arch": "llama"},
    {"layer_split": "24,xx"},
    {"lookup_sizes": "5,0"},
    {"user_args": "'unterminated"},
    {"replicas": 0},
):
    try:
        router.normalize_strata_engine(bad)
    except router.BadRequest:
        check(f"rejects {bad}", True)
    else:
        check(f"rejects {bad}", False, "accepted")

# -- write back ----------------------------------------------------------
config = copy.deepcopy(BASE)
router.strata_apply_engine_values(
    config,
    {
        "drafting": "dflash2",
        "arch": "qwen4exp",
        "spec": 6,
        "draft_conf": 0.35,
        "draft_cap": 6,
        "mtp_path": "/mnt/ai/models/mtp-draft.rt",
        "pcie_frac": 0.4,
        "adapt_every": 0,
        "adapt_swaps": 80,
        "layer_split": "auto",
        "replicas": 2,
        "split_skip_if_fits": True,
        "user_args": "--foo 1",
        "vision_mmproj": "/mnt/ai/models/mmproj-F16.gguf",
        "pool_affinity": "auto",
        "coupled_draft": False,
        "session_save_reclaim": True,
    },
)
check("drafting written", config.get("drafting") == "dflash2")
check("mtp kept true with DFlash2", config.get("mtp") is True)
check("arch untouched value kept", config.get("arch") == "qwen4exp")
check("replicas written int", config.get("replicas") == 2)
check("layer_split flag written",
      router._config_arg_value(config["args"], "--layer-split") == "auto")
check("split_skip_if_fits written", config.get("split_skip_if_fits") is True)
check("vision mmproj written", config["vision"]["mmproj"] == "/mnt/ai/models/mmproj-F16.gguf")
check("args spec removed", "--spec" not in config["args"], repr(config["args"]))
check("user_args carries spec", "--spec" in config["user_args"])
check("user_args carries draft-conf", "--draft-conf" in config["user_args"])
check("user_args carries draft-cap", "--draft-cap" in config["user_args"])
check("user_args keeps the typed option", "--foo" in config["user_args"])
check("pcie flag written", "--pcie-frac" in config["args"])
check("adapt-every written", "--adapt-every" in config["args"])
check("adapt-swaps written", "--adapt-swaps" in config["args"])
check("pool-affinity written", "--pool-affinity" in config["args"])
check("coupled-draft off writes neither flag",
      "--coupled-draft" not in config["args"] and "--no-coupled-draft" not in config["args"])
check("session-save-reclaim written", "--session-save-reclaim" in config["args"])
check("args_written refreshed", config.get("args_written") == config["args"])

# Reading the written config back gives what the form should show.
again = router.strata_engine_values(config["args"], config)
for key, want in (
    ("drafting", "dflash2"),
    ("spec", 6),
    ("draft_conf", 0.35),
    ("draft_cap", 6),
    ("pcie_frac", 0.4),
    ("adapt_every", 0),
    ("adapt_swaps", 80),
    ("replicas", 2),
    ("split_skip_if_fits", True),
    ("pool_affinity", "auto"),
    ("session_save_reclaim", True),
):
    check(f"round trip {key}", again.get(key) == want, f"got {again.get(key)!r}")

# Turning drafting off sets the legacy switch; turning it back to auto
# overrides an older "mtp": false.
router.strata_apply_engine_values(config, {"drafting": "off"})
check("drafting off sets mtp false", config.get("drafting") == "off" and config.get("mtp") is False)
router.strata_apply_engine_values(config, {"drafting": "auto"})
check("drafting auto clears legacy off", config.get("drafting") == "auto" and config.get("mtp") is True)

# Clearing a config field removes the key.
router.strata_apply_engine_values(config, {"replicas": None, "layer_split": None, "user_args": None})
check("replicas cleared", "replicas" not in config)
check("layer_split flag cleared",
      router._config_arg_value(config["args"], "--layer-split") is None
      and "layer_split" not in config)
check("user_args cleared", "user_args" not in config)

# The original config file is never touched by these calls.
check("base template unchanged", BASE["drafting"] == "mtp" and "--pcie-frac" not in BASE["args"])

print()
if failures:
    print(f"{len(failures)} failure(s)")
    sys.exit(1)
print("all checks passed")
