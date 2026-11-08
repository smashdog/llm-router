#!/usr/bin/env python3
"""Unit test: KVMem template discovery and q8 KV support detection."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import router  # noqa: E402


def make_exe(path: Path, help_text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\ncat <<'EOF'\n{help_text}\nEOF\n")
    path.chmod(0o755)
    return path


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="kvmem-ref-"))
    try:
        # 1. The managed reference wins and is reported as the template file.
        base = root / "a"
        managed = base / "config" / "polystrata" / "strata-reference.json"
        exe = make_exe(base / "bin" / "kvmem" / "strata-poly", "--kv fmt")
        cwd = base / "engine" / "kvmem" / "backends" / "polystrata"
        cwd.mkdir(parents=True)
        managed.parent.mkdir(parents=True)
        managed.write_text(json.dumps({
            "exe": str(exe), "cwd": str(cwd),
            "args": ["--model", "", "--kv", "q8_0"],
        }), encoding="utf-8")
        cfg = {"models": {"dir": str(root / "a-models")}, "cwd": str(root / "legacy")}
        path, template = router.kvmem_reference_config(cfg, base)
        assert path == managed and template["exe"] == str(exe), (path, template)

        # 2. No template anywhere: synthesize one from the managed binaries.
        base = root / "b"
        exe = make_exe(base / "bin" / "kvmem" / "cuda" / "strata-poly")
        cwd = base / "engine" / "kvmem" / "backends" / "polystrata"
        cwd.mkdir(parents=True)
        models = root / "b-models"
        models.mkdir()
        cfg = {"models": {"dir": str(models)}, "cwd": str(root / "legacy"),
               "host": "127.0.0.1", "port": 18081}
        path, template = router.kvmem_reference_config(cfg, base)
        assert path is None and template is not None, (path, template)
        assert template["exe"] == str(exe) and template["cwd"] == str(cwd), template
        assert "--kv" in template["args"] and "q8_0" in template["args"], template["args"]
        assert template["port"] == 18081 and template["host"] == "127.0.0.1", template
        assert template["log"] == str(base / "log" / "polystrata-generated.log"), template
        assert router.kvmem_cache_supported(template, cfg) is True

        # 3. A stale .bak still supplies exe/cwd/vision for the synthesized config.
        base = root / "c"
        models = root / "c-models"
        models.mkdir()
        legacy_exe = make_exe(root / "build" / "strata-poly")
        legacy_cwd = root / "legacy-backends"
        legacy_cwd.mkdir()
        vision = make_exe(root / "build-vision" / "strata-vision")
        (models / "strata-old.json.bak-1").write_text(json.dumps({
            "exe": str(legacy_exe), "cwd": str(legacy_cwd),
            "vision": {"exe": str(vision)},
            "args": ["--model", "/old", "--kv", "q8_0"],
        }), encoding="utf-8")
        cfg = {"models": {"dir": str(models)}, "cwd": str(legacy_cwd)}
        path, template = router.kvmem_reference_config(cfg, base)
        assert path is None and template is not None, (path, template)
        assert template["exe"] == str(legacy_exe) and template["cwd"] == str(legacy_cwd), template
        assert template["vision"]["exe"] == str(vision), template
        assert "--kv" in template["args"], template["args"]

        # 4. Nothing usable: the caller reports a clear error.
        cfg = {"models": {"dir": str(root / "none-models")}, "cwd": str(root / "none")}
        assert router.kvmem_reference_config(cfg, root / "d") == (None, None)

        # 5. --kv is on by default; only a readable help without it disables q8.
        with_kv = make_exe(root / "help" / "with-kv", "--kv fmt\n--kvmem-budget B")
        without_kv = make_exe(root / "help" / "without-kv", "--kvmem-budget B")
        silent = make_exe(root / "help" / "silent", "")
        assert router.kvmem_cache_supported({"exe": str(with_kv)}, {}) is True
        assert router.kvmem_cache_supported({"exe": str(without_kv)}, {}) is False
        assert router.kvmem_cache_supported({"exe": str(silent)}, {}) is True
        assert router.kvmem_cache_supported({"exe": str(root / "missing")}, {}) is True
        assert router.kvmem_cache_supported(None, {}) is False
    finally:
        shutil.rmtree(root)
    print("kvmem-reference-test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
