#!/usr/bin/env python3
"""DFlash2-compatible model naming when a Strata model is added.

strata_control/drafting.py starts DFlash2 only for a config whose model_name
matches ^qwen3.8-flash-next(-gyro-…|-unsloth-…|-q/iq/tq…)?$; the router names
models after their file, so a Flash-Next GGUF added under another name silently
kept the MTP.  These checks pin the pure naming helpers that the add worker
uses: family detection from the GGUF header (arch qwen4exp / general.name),
the supported-name gate, and the (name, alias) the worker writes to the config.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("llm_router", ROOT / "router.py")
router = importlib.util.module_from_spec(spec)
sys.modules["llm_router"] = router
spec.loader.exec_module(router)

is_flash_next = router.strata_is_flash_next
dflash2_name = router.strata_dflash2_model_name
add_name = router.strata_add_model_name
gate = router.STRATA_DFLASH2_NAME

# The real local GGUF header (arch qwen4exp) and file layout.
LOCAL = {
    "architecture": "qwen4exp",
    "name": "Qwen3.8-Flash-Next Gyro-L abliterated",
    "block_count": 48,
    "context_length": 262144,
}
STEM = "Qwen3.8-Flash-Next-Gyro-L-abliterated-TQ3_0"
FOLDER = "Qwen3.8-Flash-Next"
CANONICAL = "qwen3.8-flash-next-gyro-l-abliterated-tq3_0"

# 1. Family detection: arch qwen4exp, or the family name in general.name.
assert is_flash_next(LOCAL)
assert is_flash_next({"name": "Qwen3.8 Flash Next Gyro-L"})
assert is_flash_next({"name": "qwen3.8_flash_next"})
assert not is_flash_next({"architecture": "llama", "name": "DeepSeek-V3.1"})
assert not is_flash_next({"architecture": "qwen4exp-hybrid"})
assert not is_flash_next({})
assert not is_flash_next(None)

# 2. The gate mirror matches Strata's _SUPPORTED_NAME semantics.
for supported in (
    "qwen3.8-flash-next",
    "qwen3.8-flash-next-gyro-l-abliterated-tq3_0",
    "qwen3.8-flash-next-unsloth-bf16",
    "qwen3.8-flash-next-q4_k_m",
    "qwen3.8-flash-next-iq2_xxs",
    "QWEN3.8-FLASH-NEXT-GYRO-L",
):
    assert gate.match(supported), supported
for rejected in (
    "gyro-l-abliterated",
    "qwen3.8-flash-next-heretic",
    "qwen3.8-flash-next-gyro",
    "qwen3.8-flash-next-",
    "qwen4exp",
):
    assert not gate.match(rejected), rejected

# 3. The real file stem wins over the folder: it carries the quantization tag.
assert dflash2_name(LOCAL, STEM, FOLDER) == CANONICAL
assert add_name(LOCAL, STEM, STEM, FOLDER) == (CANONICAL, None, CANONICAL)
# A directory add is named from the file inside it, not the bare folder; the
# folder name stays routable as an alias.
assert add_name(LOCAL, FOLDER, None, STEM, FOLDER) == (CANONICAL, FOLDER, CANONICAL)

# 4. general.name fallback when the file name says nothing about the family.
assert dflash2_name(LOCAL, "gyro-l-abliterated", "models") == (
    "qwen3.8-flash-next-gyro-l-abliterated"
)

# 5. A name that already satisfies the gate is kept; the alias is the name the
#    add form/file gave it.
assert add_name(
    LOCAL, "Qwen3.8-Flash-Next-Gyro-L", "Qwen3.8-Flash-Next-Gyro-L", "weights.gguf"
) == ("qwen3.8-flash-next-gyro-l", None, "qwen3.8-flash-next-gyro-l")
assert add_name(
    LOCAL,
    "gyro-l-abliterated",
    "gyro-l-abliterated",
    taken=lambda value: False,
) == (
    "qwen3.8-flash-next-gyro-l-abliterated",
    "gyro-l-abliterated",
    "qwen3.8-flash-next-gyro-l-abliterated",
)
# ...but an identifier that already belongs to another model is not aliased.
assert add_name(
    LOCAL, "gyro-l-abliterated", taken={"gyro-l-abliterated"}.__contains__
)[1] is None

# 6. No rename outside the family, even when the file name merely looks close.
DEEPSEEK = {"architecture": "deepseek41", "name": "DeepSeek-V3.1-Terminus"}
assert dflash2_name(DEEPSEEK, "DeepSeek-V3.1-Terminus", "DeepSeek-V3.1") is None
assert add_name(DEEPSEEK, "DeepSeek-V3.1-Terminus", "DeepSeek-V3.1") == (
    "DeepSeek-V3.1-Terminus", None, None
)

# 7. Flash-Next by header but no mappable name: keep the person's name (the
#    worker then warns that DFlash2 will fall back to MTP).
HERETIC = {"architecture": "qwen4exp", "name": "Qwen3.8-Flash-Next-Heretic"}
assert dflash2_name(HERETIC, "Qwen3.8-Flash-Next-Heretic") is None
assert add_name(HERETIC, "Qwen3.8-Flash-Next-Heretic") == (
    "Qwen3.8-Flash-Next-Heretic", None, None
)

# 8. Other quant spellings of the same family.
assert dflash2_name(LOCAL, "Qwen3.8-Flash-Next-Q4_K_M") == "qwen3.8-flash-next-q4_k_m"
assert dflash2_name(LOCAL, "Qwen3.8-Flash-Next-Unsloth-BF16") == (
    "qwen3.8-flash-next-unsloth-bf16"
)
assert dflash2_name(LOCAL, "Qwen3.8-Flash-Next") == "qwen3.8-flash-next"

# 9. A plain name (no Flash-Next header) keeps the slugified identifier and no
#    alias.
assert add_name({"architecture": "llama", "name": "My Model"}, "My Model") == (
    "My-Model", None, None
)

print("strata add naming: all checks passed")
