#!/usr/bin/env python3
"""Tune an NInfer (KVMem) model against this machine.

Independent entry point, run by the router with the gateway engine stopped.
It probes VRAM/RAM/physical cores, reads the .ninfer header (the artifact
embeds its model config), then benchmarks KV formats and the drafter on a
spare port.  The winning argv is returned as a suggestion the user may apply.

Official basis (engine tree):
  engine/kvmem/backends/ninfer/docs/performance.md
  engine/kvmem/backends/ninfer/docs/recipes.md
  engine/kvmem/backends/ninfer/docs/cli.md  (KV cache formats)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_common as common  # noqa: E402

KIND = "ninfer"
SCRIPT = "tune_kvmem_ninfer.py"


def parse_int(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def memory_failure(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in (
        "out of memory", "oom", "failed to allocate", "cannot allocate",
        "not enough memory", "insufficient memory", "cuda error",
        "hip error", "内存不足", "allocation failed",
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="per-model ninfer-*.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--bench-timeout", type=float, default=600.0)
    parser.add_argument("--max-seconds", type=float, default=1500.0)
    cli = parser.parse_args()

    out_path = Path(cli.out)
    workdir = Path(cli.workdir) if cli.workdir else out_path.parent / "ninfer-work"
    workdir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "engine": KIND,
        "script": SCRIPT,
        "started_at": common.iso_now(),
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
        common.emit(
            "result", status=status, error=error or "",
            summary=(result.get("suggestions") or {}).get("summary", ""),
        )
        return 0 if status == "ok" else 1

    config_path = Path(cli.config)
    config = common.read_json(config_path, None)
    if not isinstance(config, dict) or not isinstance(config.get("args"), list):
        return finish("failed", f"无法读取模型配置：{config_path}")
    original_args = [str(item) for item in config["args"]]
    result["config_path"] = str(config_path)

    slot_info = common.find_engine_slot(common.REPO_ROOT, KIND)
    if slot_info is None:
        return finish("failed", "config.json 中没有 kind=ninfer 的引擎槽位")
    slot_name, slot = slot_info
    result["engine_slot"] = slot_name
    result["engine_port"] = slot.get("port")

    common.progress("读取硬件信息（显存 / 内存 / 物理核心）…", 3)
    hardware = common.probe_hardware()
    result["hardware"] = hardware
    common.emit("hardware", hardware=hardware)
    if not hardware.get("vram_total_mib"):
        result["notes"].append("未检测到 GPU 显存信息；按引擎自动策略继续。")

    model_raw = str(config.get("model") or config.get("artifact") or "").strip()
    if not model_raw:
        return finish("failed", "模型配置里没有 model/artifact 路径")
    model_path = Path(model_raw)
    if not model_path.is_file():
        return finish("failed", f"模型文件不存在：{model_path}")
    common.progress(f"流式读取 .ninfer 头：{model_path.name}", 6)
    info = common.read_ninfer_header(model_path)
    if info is None:
        return finish("failed", f"无法读取 .ninfer 头：{model_path}")
    result["model"] = info

    current_ctx = parse_int(common.args_value(original_args, "--max-context"))
    model_max = info.get("context_length") or current_ctx or 32768
    context = common.clamp(int(model_max), 256, 1048576)
    current_dtype = (common.args_value(original_args, "--kv-dtype") or "").strip()
    components = [str(item).lower() for item in (info.get("components") or [])]
    has_mtp = "mtp" in components or (
        bool(info.get("has_nextn")) and any("nextn" in item for item in components)
    )
    if "mtp" not in components and "dflash" not in " ".join(components):
        # Some artifacts advertise MTP only through their tensor directory.
        has_mtp = has_mtp or bool(info.get("has_nextn"))

    ram_mib = int(hardware.get("ram_available_mib") or hardware.get("ram_total_mib") or 8192)
    host_mib = common.clamp((ram_mib // 2) // 256 * 256, 4096, 24576)

    common.progress("按 NInfer 文档生成候选参数…", 10)
    base = [str(item) for item in original_args]
    base = common.args_remove(base, "--host", "--port", "--api-key")
    base = common.args_set(base, "--max-context", context)
    base = common.args_set_switch(base, "--kvmem", True, "--no-kvmem")
    budget = parse_int(common.args_value(base, "--kvmem-budget")) or 36864
    reserve = parse_int(common.args_value(base, "--kvmem-gen-reserve")) or 4096
    budget = common.clamp(budget // 64 * 64, 64, max(64, context - 64))
    reserve = common.clamp(reserve // 64 * 64, 64, max(64, context - budget))
    base = common.args_set(base, "--kvmem-budget", budget)
    base = common.args_set(base, "--kvmem-gen-reserve", reserve)
    base = common.args_set(base, "--kvmem-host-mib", host_mib)
    arch = str(info.get("architecture") or "").lower()
    if any(token in arch for token in ("qwen3_5", "qwen3_6", "qwen3.5", "qwen3.6")):
        base = common.args_set_switch(base, "--gdn-state-fp16", True)

    def variant(dtype: str, spec: bool) -> list[str]:
        args = common.args_set(base, "--kv-dtype", dtype)
        if spec:
            args = common.args_set(args, "--spec", "mtp")
            args = common.args_set(args, "--draft-tokens", 3)
        else:
            # Dropping --spec lets some artifacts auto-pick MTP/DFlash; an
            # explicit "none" is the only way to benchmark without drafting.
            args = common.args_set(args, "--spec", "none")
            args = common.args_remove(args, "--adaptive-mtp")
            args = common.args_remove(args, "--draft-tokens")
        return args

    candidates: list[dict] = []
    if has_mtp:
        candidates.append({
            "id": "int8-mtp",
            "label": "Q8 · int8 + MTP3（默认）",
            "args": variant("int8", True),
        })
        candidates.append({
            "id": "rk8v4-mtp",
            "label": "官方全能 rk8v4 + MTP3",
            "args": variant("rk8v4", True),
        })
        candidates.append({
            "id": "int8-no-spec",
            "label": "Q8 · int8（关闭推测解码）",
            "args": variant("int8", False),
        })
    else:
        candidates.append({
            "id": "int8",
            "label": "Q8 · int8（默认）",
            "args": variant("int8", False),
        })
        candidates.append({
            "id": "rk8v4",
            "label": "官方全能 rk8v4",
            "args": variant("rk8v4", False),
        })

    common.emit(
        "plan", context=context, model_max=model_max,
        kv_dtypes=[item["id"] for item in candidates],
        has_mtp=has_mtp, host_mib=host_mib,
        candidates=[{"id": item["id"], "label": item["label"]} for item in candidates],
    )

    exe = str(config.get("exe") or (slot.get("command") or [""])[0])
    cwd = str(config.get("cwd") or slot.get("cwd") or common.REPO_ROOT)

    def launch(candidate: dict, port: int, case_dir: Path):
        case_dir.mkdir(parents=True, exist_ok=True)
        model_id = str(config.get("model_id") or config.get("model_name") or "model")
        attempts: list[int] = []
        for value in (context, context // 2, context // 4):
            value = common.clamp(int(value), 256, 1048576)
            if value not in attempts:
                attempts.append(value)
        last_error: Exception | None = None
        for index, ctx_try in enumerate(attempts):
            actual_port = common.free_port()
            args = common.args_set(candidate["args"], "--max-context", ctx_try)
            cmd = [exe, model_raw, *args, "--host", "127.0.0.1", "--port", str(actual_port)]
            env = common.engine_env(common.REPO_ROOT, KIND, slot, config)
            engine = common.EngineProcess(
                cmd, cwd, env,
                case_dir / f"process-{index + 1}.log", label=str(candidate["id"]),
            )
            engine.start()
            base_url = f"http://127.0.0.1:{actual_port}"
            try:
                common.wait_ready(
                    base_url, engine,
                    timeout=min(cli.startup_timeout, 240.0),
                )
            except common.EngineError as exc:
                engine.stop()
                last_error = exc
                if memory_failure(str(exc)) and ctx_try > 256:
                    common.progress(
                        f"{candidate['label']}：{ctx_try} 上下文放不下，降到 {max(256, ctx_try // 2)} 再试…"
                    )
                    continue
                raise
            candidate["args"] = args
            candidate["context_used"] = ctx_try
            return engine, base_url, model_id
        raise last_error or common.EngineError("engine did not start")

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
        ("max_context", "上下文长度", "--max-context", "value"),
        ("kv_dtype", "KV 精度", "--kv-dtype", "value"),
        ("spec", "推测解码", "--spec", "value"),
        ("draft_tokens", "草稿长度", "--draft-tokens", "value"),
        ("kvmem_budget", "KV 预算 B", "--kvmem-budget", "value"),
        ("kvmem_gen_reserve", "生成预留 R", "--kvmem-gen-reserve", "value"),
        ("kvmem_host_mib", "Host KV 预算（MiB）", "--kvmem-host-mib", "value"),
        ("gdn_state_fp16", "GDN 状态 FP16", "--gdn-state-fp16", "switch"),
    ]
    changes = common.diff_params(original_args, best_args, specs)
    summary = (
        f"在 {hardware.get('vram_total_mib') or '?'} MiB 显存、"
        f"{hardware.get('ram_total_mib') or '?'} MiB 内存、"
        f"{hardware.get('cpu_physical_cores') or '?'} 个物理核心上测得最佳："
        f"decode {best.get('decode_tps', 0)} tok/s，"
        f"prefill {best.get('prefill_tps', 0)} tok/s（{best['label']}）。"
    )
    result["suggestions"] = {
        "args": best_args,
        "params": changes,
        "summary": summary,
        "best_candidate": best["id"],
        "context_length": int(best.get("context_used") or context),
        "kv": best_args[best_args.index("--kv-dtype") + 1]
        if "--kv-dtype" in best_args[:-1] else "int8",
        "apply_note": "应用后会写入该模型的 ninfer-*.json 并重启引擎。",
    }
    result["notes"].extend([
        "NInfer 没有 q8_0 格式；默认用字面 8-bit 的 int8，同时对照官方全能的 rk8v4。",
        "上下文优先取 .ninfer 内嵌的模型最大值；显存放不下时自动减半重试。",
        "kvmem 预算与生成预留按 64 token 对齐；host 预算按可用内存收敛。",
    ])
    if not has_mtp:
        result["notes"].append("该构件没有 MTP/DFlash 组件，未启用推测解码。")
    return finish("ok")


if __name__ == "__main__":
    raise SystemExit(main())
