#!/usr/bin/env python3
"""Tune a PolyStrata (KVMem) model against this machine.

Independent entry point: the router runs it after the smoke test, with the
gateway engine stopped.  It probes VRAM/RAM/physical cores, reads the model's
GGUF header, then starts one strata-poly process per candidate on a spare
port and measures prefill/decode.  The winning argv is written to --out as a
JSON suggestion; the router applies it only after the user confirms.

Official basis (engine tree):
  engine/kvmem/docs/recommended-config-performance.md
  engine/kvmem/backends/polystrata/docs/Q8_KV.md
  engine/kvmem/backends/polystrata/docs/VRAM_ELASTIC.md
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_common as common  # noqa: E402

KIND = "kvmem"
SCRIPT = "tune_kvmem_polystrata.py"


def parse_int(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def variant(
    base: list[str],
    *,
    budget: int,
    reserve: int,
    batch: int,
    window: int,
    mtp: bool,
) -> list[str]:
    args = common.args_set(base, "--kvmem-budget", budget)
    args = common.args_set(args, "--kvmem-gen-reserve", reserve)
    args = common.args_set(args, "--prompt-batch", batch)
    if mtp:
        # A config saved without MTP carries --no-draft; strata-poly refuses
        # --kvmem-mtp while it is present, so the switch replaces both flags.
        args = common.args_set_switch(args, "--kvmem-mtp", True, "--no-draft")
        args = common.args_set(args, "--window", window)
    else:
        args = common.args_set_switch(args, "--kvmem-mtp", False, "--no-draft")
        args = common.args_set(args, "--window", 1)
    args = common.args_set(args, "--lookup", 0)
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="per-model strata-*.json")
    parser.add_argument("--out", required=True, help="result JSON path")
    parser.add_argument("--workdir", default="")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--bench-timeout", type=float, default=600.0)
    parser.add_argument("--max-seconds", type=float, default=1500.0)
    cli = parser.parse_args()

    out_path = Path(cli.out)
    workdir = Path(cli.workdir) if cli.workdir else out_path.parent / "polystrata-work"
    workdir.mkdir(parents=True, exist_ok=True)

    started_at = common.iso_now()
    result: dict = {
        "engine": KIND,
        "script": SCRIPT,
        "started_at": started_at,
        "status": "failed",
        "hardware": None,
        "model": None,
        "candidates": [],
        "suggestions": None,
        "notes": [],
    }

    def finish(status: str, error: str | None = None) -> int:
        result["status"] = status
        result["finished_at"] = common.iso_now()
        if error:
            result["error"] = error
        common.write_json_atomic(out_path, result)
        common.emit("result", status=status, error=error or "", summary=(result.get("suggestions") or {}).get("summary", ""))
        return 0 if status == "ok" else 1

    config_path = Path(cli.config)
    config = common.read_json(config_path, None)
    if not isinstance(config, dict) or not isinstance(config.get("args"), list):
        return finish("failed", f"无法读取模型配置：{config_path}")
    original_args = [str(item) for item in config["args"]]
    result["config_path"] = str(config_path)

    slot_info = common.find_engine_slot(common.REPO_ROOT, KIND)
    if slot_info is None:
        return finish("failed", "config.json 中没有 kind=kvmem 的引擎槽位")
    slot_name, slot = slot_info
    result["engine_slot"] = slot_name
    result["engine_port"] = slot.get("port")

    common.progress("读取硬件信息（显存 / 内存 / 物理核心）…", 3)
    hardware = common.probe_hardware()
    result["hardware"] = hardware
    common.emit("hardware", hardware=hardware)
    if not hardware.get("vram_total_mib"):
        result["notes"].append("未检测到 GPU 显存信息；按保守值继续。")
    if not hardware.get("ram_total_mib"):
        result["notes"].append("未读取到 /proc/meminfo；Host KV 预算按 8 GiB 处理。")

    model_raw = common.args_value(original_args, "--model")
    model_path = Path(model_raw) if model_raw else None
    if model_path is None or not model_path.is_file():
        return finish("failed", f"模型配置里没有可用的 --model 路径：{model_raw}")
    common.progress(f"流式读取 GGUF 头：{model_path.name}", 6)
    info = common.read_gguf_header(model_path)
    if info is None:
        return finish("failed", f"无法读取 GGUF 头：{model_path}")
    result["model"] = info

    current_ctx = parse_int(common.args_value(original_args, "--max-context", "--ctx"))
    model_max = info.get("context_length") or current_ctx or 32768
    context = common.clamp(int(model_max), 256, 1048576)
    current_mtp = common.args_switch(original_args, "--kvmem-mtp")
    mtp_supported = bool(info.get("has_nextn")) or current_mtp
    if not mtp_supported:
        result["notes"].append("GGUF 没有内嵌 nextn 草稿块，候选全部关闭 MTP。")

    cores = int(hardware.get("cpu_physical_cores") or 4)
    threads = common.clamp(cores, 1, 64)
    ram_mib = int(hardware.get("ram_available_mib") or hardware.get("ram_total_mib") or 8192)
    host_mib = common.clamp((ram_mib // 2) // 1024 * 1024, 4096, 24576)
    vram_total = int(hardware.get("vram_total_mib") or 0)
    vram_reserve = 512 if vram_total <= 12288 else 768

    common.progress("按引擎文档生成候选参数…", 10)
    base = [str(item) for item in original_args]
    base = common.args_set(base, "--max-context", context)
    base = common.args_set(base, "--kv", "q8_0")
    base = common.args_set(base, "--kvmem-block-tokens", 32)
    base = common.args_set(base, "--vram-reserve-mib", vram_reserve)
    base = common.args_set(base, "--threads", threads)
    base = common.args_set(base, "--kvmem-host-mib", host_mib)
    if common.args_value(base, "--kvmem-sink-tokens") is None:
        base = common.args_set(base, "--kvmem-sink-tokens", 64)
    if common.args_value(base, "--kvmem-recent-tokens") is None:
        base = common.args_set(base, "--kvmem-recent-tokens", 64)
    # A profile copied from another host must not abort the run on this one.
    profile = common.args_value(base, "--expert-profile")
    if profile:
        profile_path = Path(profile)
        if not profile_path.exists() and not profile_path.parent.is_dir():
            base = common.args_remove(base, "--expert-profile")
            result["notes"].append(
                f"忽略不可用的专家统计文件 {profile}（调优时由引擎自动选择专家）。"
            )

    # B + R must stay inside the context; the official 36864/16384 pair is
    # for a 262144-token window.
    max_pair = max(4096, context - 1024)
    docs_budget = common.align(min(36864, max_pair - 4096), 32)
    docs_reserve = common.align(min(16384, max_pair - docs_budget), 32)
    docs_budget = max(128, docs_budget)
    docs_reserve = max(32, docs_reserve)
    if docs_budget + docs_reserve > context:
        docs_budget = common.align(max(128, context - docs_reserve), 32)
    low_budget = common.align(min(32768, docs_budget), 32)
    low_reserve = common.align(min(8192, max(32, context - low_budget)), 32)

    candidates: list[dict] = [{
        "id": "docs",
        "label": "文档推荐（B=36864 / R=16384 / batch=512 / MTP3）",
        "args": variant(
            base, budget=docs_budget, reserve=docs_reserve,
            batch=min(512, docs_reserve), window=3 if mtp_supported else 1,
            mtp=mtp_supported,
        ),
    }, {
        "id": "compact",
        "label": "显存保守（B=32768 / R=8192 / batch=512 / MTP2）",
        "args": variant(
            base, budget=low_budget, reserve=min(low_reserve, max(512, low_reserve)),
            batch=min(512, low_reserve), window=2 if mtp_supported else 1,
            mtp=mtp_supported,
        ),
    }]
    if mtp_supported:
        candidates.append({
            "id": "no-mtp",
            "label": "关闭 MTP（window=1 / --no-draft）",
            "args": variant(
                base, budget=docs_budget, reserve=docs_reserve,
                batch=min(512, docs_reserve), window=1, mtp=False,
            ),
        })

    common.emit(
        "plan",
        context=context,
        model_max=model_max,
        threads=threads,
        host_mib=host_mib,
        vram_reserve_mib=vram_reserve,
        mtp_supported=mtp_supported,
        candidates=[{"id": item["id"], "label": item["label"]} for item in candidates],
    )

    def launch(candidate: dict, port: int, case_dir: Path):
        case_dir = case_dir.resolve()
        case_dir.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(config)
        cfg["args"] = candidate["args"]
        cfg["host"] = "127.0.0.1"
        cfg["port"] = port
        cfg["log"] = str(case_dir / "engine.log")
        # The engine runs with cwd set to the backend tree, so every path in
        # its config and on its command line must be absolute.
        cfg_path = case_dir / "model.json"
        common.write_json_atomic(cfg_path, cfg)
        cmd = common.expand_slot_command(slot, cfg_path, "127.0.0.1", port, common.REPO_ROOT)
        env = common.engine_env(common.REPO_ROOT, KIND, slot, cfg)
        engine = common.EngineProcess(
            cmd, slot.get("cwd") or common.REPO_ROOT, env,
            case_dir / "process.log", label=str(candidate["id"]),
        )
        engine.start()
        model_id = str(cfg.get("model_name") or cfg.get("model_id") or "model")
        return engine, f"http://127.0.0.1:{port}", model_id

    common.progress("开始逐组基准测试（每组会重新加载模型）…", 15)
    cands = common.run_candidates(
        candidates, launch, workdir,
        startup_timeout=cli.startup_timeout,
        bench_timeout=cli.bench_timeout,
        max_seconds=cli.max_seconds,
    )
    result["candidates"] = cands
    best = common.best_candidate(cands)
    if best is None:
        errors = "; ".join(
            f"{item['label']}: {item.get('error', '')[:120]}" for item in cands
        )
        return finish("failed", f"所有候选都未能完成基准测试。{errors}")

    best_args = [str(item) for item in best["args"]]
    specs = [
        ("max_context", "上下文长度", ("--max-context", "--ctx"), "value"),
        ("kv", "KV 缓存", "--kv", "value"),
        ("kvmem_budget", "KV 预算 B", "--kvmem-budget", "value"),
        ("kvmem_gen_reserve", "生成预留 R", "--kvmem-gen-reserve", "value"),
        ("prompt_batch", "Prompt batch", "--prompt-batch", "value"),
        ("threads", "CPU 线程（物理核心）", ("--threads", "-t"), "value"),
        ("kvmem_host_mib", "Host KV 预算（MiB）", "--kvmem-host-mib", "value"),
        ("vram_reserve_mib", "VRAM 预留（MiB）", "--vram-reserve-mib", "value"),
        ("window", "推测窗口", "--window", "value"),
        ("kvmem_mtp", "启用 MTP", "--kvmem-mtp", "switch"),
        ("lookup", "历史查表", "--lookup", "value"),
        ("kvmem_block_tokens", "检索块 block", "--kvmem-block-tokens", "value"),
    ]
    changes = common.diff_params(original_args, best_args, specs)
    summary = (
        f"在 {hardware.get('vram_total_mib') or '?'} MiB 显存、"
        f"{hardware.get('ram_total_mib') or '?'} MiB 内存、"
        f"{cores} 个物理核心上测得最佳：decode {best.get('decode_tps', 0)} tok/s，"
        f"prefill {best.get('prefill_tps', 0)} tok/s（{best['label']}）。"
    )
    result["suggestions"] = {
        "args": best_args,
        "params": changes,
        "summary": summary,
        "best_candidate": best["id"],
        "context_length": context,
        "kv": "q8_0",
        "apply_note": "应用后会写入该模型的 strata-*.json 并重启引擎。",
    }
    result["notes"].extend([
        "Q8 KV（--kv q8_0）按引擎文档减半 KV 显存；上下文取模型 GGUF 的最大值。",
        "B/R/batch/window 取自 recommended-config-performance.md，并结合本机显存与内存收敛。",
        f"CPU 线程使用物理核心数 {cores}（而不是逻辑线程数 {hardware.get('cpu_logical_cores')}）。",
    ])
    return finish("ok")


if __name__ == "__main__":
    raise SystemExit(main())
