#!/usr/bin/env python3
"""Tune a KVMem + hipfire model against this machine.

Independent entry point, run by the router with the gateway engine stopped.
It probes VRAM/RAM, streams the HFQM header for the trained context ceiling
and the embedded MTP head, then benchmarks speculation modes against prefill
chunk sizes on a spare port. K/V stays q8 and the KVMem budget/slots keep the
model config's values; the winning spec/prefill_chunk pair is returned as
engine_params (spec → --spec, prefill_chunk → the multi-slot env variable).

Official basis (engine tree):
  hipfire CHANGELOG.md (speculation auto/mtp defaults; mq4-xts MTP numbers)
  hipfire docs/CONFIG.md (serve multi-slot and HIPFIRE_* environment)
  llm-router router.py KVMEM_HIPFIRE_ENGINE_SPECS (parameter targets)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_common as common  # noqa: E402

KIND = "kvmem_hipfire"
SCRIPT = "tune_kvmem_hipfire.py"


def parse_int(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="per-model kvmem-hipfire-*.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--bench-timeout", type=float, default=600.0)
    parser.add_argument("--max-seconds", type=float, default=2400.0)
    cli = parser.parse_args()

    out_path = Path(cli.out)
    workdir = Path(cli.workdir) if cli.workdir else out_path.parent / "kvmem-hipfire-work"
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
    original_env = config.get("env") if isinstance(config.get("env"), dict) else {}
    result["config_path"] = str(config_path)

    slot_info = common.find_engine_slot(common.REPO_ROOT, KIND)
    if slot_info is None:
        return finish("failed", "config.json 中没有 kind=kvmem_hipfire 的引擎槽位")
    slot_name, slot = slot_info
    result["engine_slot"] = slot_name
    result["engine_port"] = slot.get("port")

    common.progress("读取硬件信息（显存 / 内存 / 物理核心）…", 3)
    hardware = common.probe_hardware()
    result["hardware"] = hardware
    common.emit("hardware", hardware=hardware)
    if not hardware.get("vram_total_mib"):
        return finish("failed", "未检测到 GPU；KVMem + hipfire 调优需要显存信息")

    model_raw = str(config.get("model") or config.get("artifact") or "").strip()
    if not model_raw:
        return finish("failed", "模型配置里没有 model/artifact 路径")
    cwd = str(config.get("cwd") or slot.get("cwd") or common.REPO_ROOT)
    model_path = Path(model_raw)
    if not model_path.is_absolute():
        model_path = Path(cwd) / model_path
    if not model_path.is_file():
        return finish("failed", f"模型文件不存在：{model_path}")
    common.progress(f"流式读取 HFQM 头：{model_path.name}", 6)
    info = common.read_hfqm_header(model_path)
    if info is None:
        return finish("failed", f"无法读取 HFQM 头：{model_path}")
    result["model"] = info

    # Same admission order as the router: explicit max_seq, then the config
    # context_length, then the trained ceiling from the artifact metadata.
    context = (
        parse_int(config.get("max_seq"))
        or parse_int(config.get("context_length"))
        or info.get("context_length")
    )
    if not context:
        return finish("failed", "HFQM 元数据缺少 max_position_embeddings，无法确定上下文上限")
    context = common.clamp(int(context), 256, 1048576)

    exe_raw = str(config.get("exe") or (slot.get("command") or [""])[0])
    exe_path = Path(exe_raw)
    if not exe_path.is_absolute():
        exe_path = Path(cwd) / exe_path
    daemon = exe_path.parent / "kvmem-hipfire-daemon"
    if not exe_path.is_file():
        return finish("failed", f"hipfire CLI 不存在：{exe_path}")
    if not daemon.is_file():
        return finish("failed", f"缺少 hipfire daemon 二进制：{daemon}")

    # Product defaults mirror router._environment; the slot/model env that
    # engine_env already merged keeps overriding each of them.
    env = common.engine_env(common.REPO_ROOT, KIND, slot, config)
    env.setdefault("HIPFIRE_KVMEM", "1")
    env.setdefault("HIPFIRE_SERVE_MULTI_SLOT", "true")
    env.setdefault("HIPFIRE_KV_MODE", "q8")
    env.setdefault("HIPFIRE_KVMEM_BUDGET_TOKENS", "32768")
    env.setdefault("HIPFIRE_SERVE_MULTI_SLOT_SLOTS", "4")
    env.setdefault("HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK", "256")
    env.setdefault("HIPFIRE_FLASH_PARTIALS_BATCH", "4")
    env.setdefault("HIPFIRE_DAEMON_BIN", str(daemon))
    budget = parse_int(env.get("HIPFIRE_KVMEM_BUDGET_TOKENS")) or 32768
    slots = parse_int(env.get("HIPFIRE_SERVE_MULTI_SLOT_SLOTS")) or 4

    # Vision placement mirrors router._environment so the benchmark carries
    # the same tower VRAM as production: an explicit sidecar wires the tower
    # (CPU = host-mapped weights in RAM, GPU = streamed/resident in VRAM);
    # without one the gate is forced off so hipfire's sibling-file
    # auto-discovery cannot load a tower behind the benchmark's back.
    vision_cfg = config.get("vision") if isinstance(config.get("vision"), dict) else {}
    sidecar_raw = str(vision_cfg.get("sidecar") or "").strip()
    vision_info: dict | None = None
    if sidecar_raw:
        sidecar_path = Path(sidecar_raw)
        if not sidecar_path.is_absolute():
            sidecar_path = Path(cwd) / sidecar_path
        if not sidecar_path.is_file():
            return finish("failed", f"视觉 sidecar 不存在：{sidecar_path}")
        vision_gpu = bool(vision_cfg.get("gpu", False))
        env["HIPFIRE_VISION_SIDECAR"] = str(sidecar_path)
        env["HIPFIRE_VISION_HOST_MAPPED"] = "0" if vision_gpu else "1"
        mode = str(env.get("HIPFIRE_VISION_MODE") or "auto").strip().lower()
        if not vision_gpu or mode not in ("auto", "resident", "on"):
            # resident is a GPU-only placement; CPU pins host-mapped auto.
            mode = "auto"
        env["HIPFIRE_VISION_MODE"] = mode
        if not vision_gpu:
            placement, overhead_mib = "HostMapped（内存零拷贝）", 330
        elif mode == "resident":
            placement, overhead_mib = "resident（完整驻显存）", 1280
        else:
            placement, overhead_mib = "Streamed（按块流式）", 470
        sidecar_info = common.read_hfqm_header(sidecar_path) or {}
        vision_info = {
            "sidecar": str(sidecar_path),
            "sidecar_mib": sidecar_info.get("size_mib"),
            "gpu": vision_gpu,
            "mode": mode,
            "placement": placement,
            "vram_overhead_mib": overhead_mib,
        }
        result["vision"] = vision_info
        result["notes"].append(
            f"视觉 sidecar 计入显存：{placement}，推理峰值约 +{overhead_mib} MiB"
            f"（包体 {sidecar_info.get('size_mib', '?')} MiB"
            + ("，驻内存不占显存" if not vision_gpu else "")
            + "）；候选基准带着视觉塔一起加载。"
        )
    else:
        env["HIPFIRE_VISION_MODE"] = "off"
        env["HIPFIRE_VISION_SIDECAR"] = ""
        env.pop("HIPFIRE_VISION_HOST_MAPPED", None)
        result["notes"].append("未配置视觉路径：HIPFIRE_VISION_MODE=off，纯文本基准，无视觉显存开销。")

    mtp_layers = int(info.get("mtp_layers") or 0)
    spec_modes = ["auto"] + (["mtp"] if mtp_layers > 0 else []) + ["ngram", "off"]
    # Scratch rows = prefill_chunk × slots; 128/256 keep a 27B MQ4 model
    # plus its DFlash2 draft scratch within a 20 GiB card beside the paged
    # KV arenas and the context-ceiling-scaled flash partials.
    chunks = (128, 256)

    common.progress("按 HFQM 元数据与 hipfire 文档生成候选参数…", 10)
    # Router-owned flags never belong to the per-candidate argv; --spec is
    # set per candidate below.
    base = common.args_remove(
        original_args, "--model", "--max-seq", "--host", "--port",
        "--api-key", "--api-key-file", "--detach", "--spec",
    )
    candidates: list[dict] = []
    for spec in spec_modes:
        for chunk in chunks:
            candidates.append({
                "id": f"spec-{spec}-chunk-{chunk}",
                "label": f"spec={spec} / prefill_chunk={chunk}",
                "args": common.args_set(base, "--spec", spec),
                "spec": spec,
                "prefill_chunk": chunk,
            })
    common.emit(
        "plan", context=context, mtp_layers=mtp_layers, spec_modes=spec_modes,
        chunks=list(chunks), budget=budget, slots=slots,
        candidates=[{"id": item["id"], "label": item["label"]} for item in candidates],
    )
    result["notes"].append(
        f"投机候选按模型自动检测：mtp_num_hidden_layers={mtp_layers}"
        + ("，包含 mtp" if mtp_layers else "，不含 mtp")
        + "；auto/ngram/off 始终参测。dflash/dspark 需要外部 draft 文件，不参与自动调优。"
    )

    def launch(candidate: dict, port: int, case_dir: Path):
        case_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(exe_path), "serve", "--model", str(model_path),
            "--max-seq", str(context), *candidate["args"],
            f"127.0.0.1:{port}",
        ]
        case_env = dict(env)
        case_env["HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK"] = str(candidate["prefill_chunk"])
        engine = common.EngineProcess(
            cmd, cwd, case_env,
            case_dir / "process.log", label=str(candidate["id"]),
        )
        engine.start()
        # hipfire echoes the exact --model string as its /v1/models id.
        return engine, f"http://127.0.0.1:{port}", str(model_path)

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

    after_spec = str(best["spec"])
    after_chunk = int(best["prefill_chunk"])
    before_spec = common.args_value(original_args, "--spec")
    before_chunk = parse_int(original_env.get("HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK"))
    changes: list[dict] = []
    if (before_spec or "auto") != after_spec:
        changes.append({
            "key": "spec", "label": "投机解码",
            "before": before_spec, "after": after_spec,
        })
    if (before_chunk if before_chunk is not None else 256) != after_chunk:
        changes.append({
            "key": "prefill_chunk", "label": "Prefill 分块行数",
            "before": before_chunk, "after": after_chunk,
        })
    reuse_text = ""
    if best.get("reuse_prompt_tokens"):
        reuse_text = f"，KV 复用 {best.get('reuse_ratio', 0) * 100:.1f}%"
    summary = (
        f"在 {hardware.get('vram_total_mib') or '?'} MiB 显存、"
        f"{hardware.get('ram_total_mib') or '?'} MiB 内存上测得最佳："
        f"decode {best.get('decode_tps', 0)} tok/s，"
        f"prefill {best.get('prefill_tps', 0)} tok/s{reuse_text}（{best['label']}）。"
    )
    result["suggestions"] = {
        "args": [str(item) for item in best["args"]],
        "engine_params": {"spec": after_spec, "prefill_chunk": after_chunk},
        "params": changes,
        "summary": summary,
        "best_candidate": best["id"],
        "context_length": context,
        "kv": "q8",
        "apply_note": "应用后会写入该模型配置的 --spec 与 HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK 并重启引擎。",
    }
    result["notes"].extend([
        f"上下文取 {context:,}（配置 max_seq 优先，否则 HFQM 训练上限）；"
        f"K/V 固定 q8，KVMem 预算 {budget:,} tokens、{slots} 槽位沿用模型配置，不参与候选。",
        "prefill 分块候选 128/256（HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK）；scratch 行数 = 分块 × 槽位。",
        "评分以 decode 优先，prefill 在 2% 内破平；下一轮 KV 复用未命中的候选会被降权。",
    ])
    note = common.reuse_note(best)
    if note:
        result["notes"].append(note)
    return finish("ok")


if __name__ == "__main__":
    raise SystemExit(main())