#!/usr/bin/env python3
"""Tune a KVMem + llama.cpp model against this machine.

Independent entry point, run by the router with the gateway engine stopped.
It probes VRAM/RAM/physical cores, streams the GGUF header, then benchmarks
offload depth and batch sizes on a spare port.  Q8 KV + Flash Attention are
the baseline; the winning argv is returned as a suggestion.

Official basis (engine tree):
  engine/kvmem/backends/llamacpp/docs/development/token_generation_performance_tips.md
  engine/kvmem/backends/llamacpp/docs/multi-gpu.md
  engine/kvmem/docs/llamacpp-v090-integration.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_common as common  # noqa: E402

KIND = "kvmem_llama"
SCRIPT = "tune_kvmem_llama.py"


def parse_int(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def estimate_ngl(info: dict, free_mib: int, kv_tokens: int, reserve_mib: int) -> tuple[int, bool]:
    size_mib = float(info.get("size_mib") or 0.0)
    block_count = int(info.get("block_count") or 32)
    kv_heads = int(info.get("head_count_kv") or info.get("head_count") or 8)
    head_dim = int(info.get("head_dim") or 128)
    # q8_0 stores 32 values in 34 bytes; K and V each.
    row_bytes = 2 * kv_heads * head_dim * (34.0 / 32.0)
    kv_mib = kv_tokens * row_bytes / (1024 * 1024)
    overhead_mib = 1024 + reserve_mib
    budget_mib = free_mib - overhead_mib - kv_mib
    if size_mib > 0 and size_mib + kv_mib + overhead_mib <= free_mib * 0.95:
        return 999, True
    per_layer = max(8.0, (size_mib - 1024.0) / max(1, block_count))
    ngl = int(budget_mib / per_layer)
    return common.clamp(ngl, 0, block_count), False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="per-model kvmem-llama-*.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--bench-timeout", type=float, default=600.0)
    parser.add_argument("--max-seconds", type=float, default=1500.0)
    cli = parser.parse_args()

    out_path = Path(cli.out)
    workdir = Path(cli.workdir) if cli.workdir else out_path.parent / "kvmem-llama-work"
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
        return finish("failed", "config.json 中没有 kind=kvmem_llama 的引擎槽位")
    slot_name, slot = slot_info
    result["engine_slot"] = slot_name
    result["engine_port"] = slot.get("port")

    common.progress("读取硬件信息（显存 / 内存 / 物理核心）…", 3)
    hardware = common.probe_hardware()
    result["hardware"] = hardware
    common.emit("hardware", hardware=hardware)
    if not hardware.get("vram_total_mib"):
        return finish("failed", "未检测到 GPU；KVMem + llama.cpp 调优需要显存信息")

    model_raw = str(config.get("model") or config.get("artifact") or "").strip()
    if not model_raw:
        return finish("failed", "模型配置里没有 model/artifact 路径")
    model_path = Path(model_raw)
    if not model_path.is_file():
        return finish("failed", f"模型文件不存在：{model_path}")
    common.progress(f"流式读取 GGUF 头：{model_path.name}", 6)
    info = common.read_gguf_header(model_path)
    if info is None:
        return finish("failed", f"无法读取 GGUF 头：{model_path}")
    result["model"] = info

    current_ctx = parse_int(common.args_value(original_args, "-c", "--ctx-size"))
    model_max = info.get("context_length") or current_ctx or 32768
    context = common.clamp(int(model_max), 256, 1048576)
    cores = common.clamp(int(hardware.get("cpu_physical_cores") or 4), 1, 64)
    ram_mib = int(hardware.get("ram_available_mib") or hardware.get("ram_total_mib") or 8192)
    host_mib = common.clamp((ram_mib // 2) // 256 * 256, 4096, 24576)
    free_mib = int(hardware.get("vram_free_mib") or hardware.get("vram_total_mib") or 0)
    vram_reserve = 512 if int(hardware.get("vram_total_mib") or 0) <= 12288 else 768

    common.progress("按 llama.cpp 性能文档生成候选参数…", 10)
    base = [str(item) for item in original_args]
    base = common.args_remove(
        base, "-m", "--model", "-a", "--alias", "--host", "--port",
        "--api-key", "--api-key-file", "--mmproj", "-mmdev",
        "--mmproj-device", "--mmproj-offload", "--no-mmproj-offload",
        "-ctk", "--cache-type-k", "-ctv", "--cache-type-v",
        "-fa", "--flash-attn",
    )
    base = common.args_set(base, "-c", context)
    base = common.args_set(base, "--kv-dtype", "q8_0")
    base = common.args_set(base, "-fa", "on")
    # llama.cpp caches the prompt in its slot by default; an inherited
    # --no-cache-prompt would re-prefill every conversation turn.
    base = common.args_remove(base, "--no-cache-prompt")
    base = common.args_set(base, "--kvmem-block-tokens", 32)
    base = common.args_set(base, "--kvmem-host-mib", host_mib)
    base = common.args_set_switch(base, "--kvmem", True, "--no-kvmem")
    budget = parse_int(common.args_value(base, "--kvmem-budget")) or 32768
    reserve = parse_int(common.args_value(base, "--kvmem-gen-reserve")) or 1024
    budget = common.clamp(budget // 32 * 32, 128, max(128, context - 32))
    reserve = common.clamp(reserve // 32 * 32, 32, max(32, context - budget))
    base = common.args_set(base, "--kvmem-budget", budget)
    base = common.args_set(base, "--kvmem-gen-reserve", reserve)
    base = common.args_set(base, "-t", cores)
    base = common.args_set(base, "-tb", cores)

    ngl_auto, full_fit = estimate_ngl(info, free_mib, budget + reserve, vram_reserve)
    if full_fit:
        ngl_primary = 999
        result["notes"].append(
            "模型权重与 KV 预算可全部放进显存，候选使用 -ngl 999（全部卸载）。"
        )
    else:
        ngl_primary = max(1, ngl_auto)
        result["notes"].append(
            f"权重 {info.get('size_mib') or '?'} MiB 超过可用显存 {free_mib} MiB，"
            f"按层大小估算 -ngl {ngl_primary}（block_count {info.get('block_count')}）。"
        )
    ngl_conservative = max(1, ngl_primary - 4) if ngl_primary < 999 else max(1, (info.get("block_count") or 32) - 4)

    def variant(ngl: int, batch: int, ubatch: int) -> list[str]:
        args = common.args_set(base, "-ngl", ngl)
        args = common.args_set(args, "-b", batch)
        args = common.args_set(args, "-ub", ubatch)
        return args

    candidates: list[dict] = [{
        "id": "offload-primary",
        "label": f"推荐卸载（ngl={ngl_primary} / batch=2048 / ubatch=512）",
        "args": variant(ngl_primary, 2048, 512),
    }, {
        "id": "offload-conservative",
        "label": f"保守卸载（ngl={ngl_conservative} / batch=2048 / ubatch=512）",
        "args": variant(ngl_conservative, 2048, 512),
    }, {
        "id": "batch-512",
        "label": f"稳妥 batch（ngl={ngl_primary} / batch=512 / ubatch=512）",
        "args": variant(ngl_primary, 512, 512),
    }]

    common.emit(
        "plan", context=context, model_max=model_max, threads=cores,
        ngl_primary=ngl_primary, ngl_conservative=ngl_conservative,
        full_fit=full_fit, host_mib=host_mib,
        candidates=[{"id": item["id"], "label": item["label"]} for item in candidates],
    )

    exe = str(config.get("exe") or (slot.get("command") or [""])[0])
    cwd = str(config.get("cwd") or slot.get("cwd") or common.REPO_ROOT)
    alias = str(config.get("model_name") or config.get("model_id") or "tune-model")
    vision = config.get("vision") if isinstance(config.get("vision"), dict) else {}
    mmproj = str(config.get("mmproj") or vision.get("mmproj") or "").strip()
    mmproj_gpu = vision.get("gpu")
    if not isinstance(mmproj_gpu, bool):
        mmproj_gpu = True

    def launch(candidate: dict, port: int, case_dir: Path):
        case_dir.mkdir(parents=True, exist_ok=True)
        cmd = [exe, "-m", model_raw, "-a", alias, *candidate["args"]]
        if mmproj:
            cmd += ["--mmproj", mmproj]
            cmd.append("--mmproj-offload" if mmproj_gpu else "--no-mmproj-offload")
        cmd += ["--host", "127.0.0.1", "--port", str(port)]
        env = common.engine_env(common.REPO_ROOT, KIND, slot, config)
        engine = common.EngineProcess(
            cmd, cwd, env,
            case_dir / "process.log", label=str(candidate["id"]),
        )
        engine.start()
        return engine, f"http://127.0.0.1:{port}", alias

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
        ("ngl", "GPU 层数", ("-ngl", "--n-gpu-layers", "--gpu-layers"), "value"),
        ("threads", "CPU 线程（物理核心）", ("-t", "--threads"), "value"),
        ("threads_batch", "CPU batch 线程", ("-tb", "--threads-batch"), "value"),
        ("batch_size", "逻辑 batch", ("-b", "--batch-size"), "value"),
        ("ubatch_size", "物理 batch", ("-ub", "--ubatch-size"), "value"),
        ("max_context", "上下文长度", ("-c", "--ctx-size"), "value"),
        ("kv_dtype", "KV 精度", "--kv-dtype", "value"),
        ("flash_attn", "Flash Attention", ("-fa", "--flash-attn"), "value"),
        ("kvmem_budget", "KV 预算 B", "--kvmem-budget", "value"),
        ("kvmem_gen_reserve", "生成预留 R", "--kvmem-gen-reserve", "value"),
        ("kvmem_host_mib", "Host KV 预算（MiB）", "--kvmem-host-mib", "value"),
    ]
    changes = common.diff_params(original_args, best_args, specs)
    reuse_text = ""
    if best.get("reuse_prompt_tokens"):
        reuse_text = f"，KV 复用 {best.get('reuse_ratio', 0) * 100:.1f}%"
    summary = (
        f"在 {hardware.get('vram_total_mib') or '?'} MiB 显存、"
        f"{hardware.get('ram_total_mib') or '?'} MiB 内存、{cores} 个物理核心上测得最佳："
        f"decode {best.get('decode_tps', 0)} tok/s，"
        f"prefill {best.get('prefill_tps', 0)} tok/s{reuse_text}（{best['label']}）。"
    )
    result["suggestions"] = {
        "args": best_args,
        "params": changes,
        "summary": summary,
        "best_candidate": best["id"],
        "context_length": context,
        "kv": "q8_0",
        "apply_note": "应用后会写入该模型的 kvmem-llama-*.json 并重启引擎。",
    }
    result["notes"].extend([
        "Q8 KV 通过 --kv-dtype q8_0 + Flash Attention 开启，上下文取 GGUF 最大值。",
        "线程数按 llama.cpp 文档取物理核心数，-ngl 由模型层数与可用显存估算。",
        "batch/ubatch 取文档建议的 2048/512，并与 512/512 做了实测对照。",
        "保留引擎默认的前缀复用；从旧配置继承的 --no-cache-prompt 会被移除，避免每轮对话重新 prefill。",
    ])
    note = common.reuse_note(best)
    if note:
        result["notes"].append(note)
    return finish("ok")


if __name__ == "__main__":
    raise SystemExit(main())
