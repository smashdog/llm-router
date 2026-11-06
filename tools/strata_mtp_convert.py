#!/usr/bin/env python3
"""Convert a llama.cpp-layout Qwen3.8-Flash-Next MTP draft GGUF into the
runtime files Strata's engine loads for its draft layer
(dense.txt / dense.bin / experts.bin / draft_vocab.bin).

Strata's own path is `tools/mtp_rt.py`, which expects a GGUF holding the
checkpoint's `mtp.layers.0.*` tensors in BF16.  llama.cpp / Unsloth ship the
same draft block as a standalone GGUF with `blk.<N>.*` names and quantized
tensors (Q4_K / Q6_K / IQ4_NL / Q8_0).  This tool maps that layout onto the
names and kinds the engine reads.

Two conventions differ from `mtp_rt.py` and are handled here:

* the draft GGUF's F32 norm weights already include the Gemma "+1" (checked
  against the checkpoint's raw `mtp.*` tensors: the GGUF value is exactly
  raw + 1), so they are copied verbatim - adding 1 again would shift every
  norm by one;
* `nextn.eh_proj` fuses the checkpoint's `fc_embedding` and `fc_hidden`
  along the input axis (embedding half first, matching llama.cpp's
  `ggml_concat(e_norm, h_norm, dim=0)`), so its Q8_0 rows are split into two
  half-row tensors without requantizing.

Run it with the Strata virtualenv interpreter (it needs numpy and gguf-py):

    .venv/bin/python strata_mtp_convert.py --strata /path/to/Strata \
        --gguf /path/to/mtp-...-draft.gguf --out /path/to/rt

The output is re-used when it already holds experts.bin, dense.txt and
dense.bin unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# name suffix after "blk.<N>." -> (runtime tensor name, kind)
#   q8   engine Q8_0 blob (requantized with mtp_rt's ggml-reference quantizer)
#   bf16 engine BF16 blob
#   norm engine F32 blob, values copied as stored (the draft GGUF already has 1 + w)
DENSE_MAP = (
    ("attn_q.weight", "self_attn.q_proj.weight", "q8"),
    ("attn_k.weight", "self_attn.k_proj.weight", "q8"),
    ("attn_v.weight", "self_attn.v_proj.weight", "q8"),
    ("attn_output.weight", "self_attn.o_proj.weight", "q8"),
    ("ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight", "q8"),
    ("ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight", "q8"),
    ("ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight", "q8"),
    ("ffn_gate_inp.weight", "mlp.gate.weight", "bf16"),
    ("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight", "bf16"),
    ("hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight", "bf16"),
    ("hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight", "bf16"),
    ("hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight", "bf16"),
    ("hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight", "bf16"),
    ("hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight", "bf16"),
    ("hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight", "bf16"),
    ("nextn.hc_head_down.weight", "hyper_connection_mixer.input_mix_weight_down.weight", "bf16"),
    ("nextn.hc_head_up.weight", "hyper_connection_mixer.input_mix_weight_up.weight", "bf16"),
    ("attn_q_norm.weight", "self_attn.q_norm.weight", "norm"),
    ("attn_k_norm.weight", "self_attn.k_norm.weight", "norm"),
    ("hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight", "norm"),
    ("hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight", "norm"),
    ("nextn.hc_head_norm.weight", "hyper_connection_mixer.hc_norm.weight", "norm"),
    ("nextn.enorm.weight", "pre_fc_norm_embedding.weight", "norm"),
    ("nextn.hnorm.weight", "pre_fc_norm_hidden.weight", "norm"),
)

EXPERT_NAMES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


def to_bf16(values: np.ndarray) -> bytes:
    """float32 -> bfloat16 bytes (round to nearest even)."""
    f = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    u = f.view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return rounded.astype(np.uint16).tobytes()


def fail(message: str) -> "SystemExit":
    print(f"strata-mtp-convert: ERROR: {message}", file=sys.stderr)
    return SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="llama.cpp-layout MTP draft GGUF (blk.<N>.*)")
    ap.add_argument("--out", required=True, help="output runtime directory (dense.txt, dense.bin, experts.bin)")
    ap.add_argument("--strata", required=True, help="Strata checkout (tools/ and data/draft_vocab.bin)")
    ap.add_argument("--force", action="store_true", help="rebuild even when the output is complete")
    args = ap.parse_args()

    strata = Path(args.strata).expanduser().resolve()
    tools_dir = strata / "tools"
    gguf_py = strata / "third_party" / "llama.cpp" / "gguf-py"
    if not (tools_dir / "mtp_rt.py").is_file():
        raise fail(f"no tools/mtp_rt.py under {strata}")
    if not (gguf_py / "gguf").is_dir():
        raise fail(f"no gguf-py under {gguf_py}")
    os.environ.setdefault("STRATA_GGUF_PY", str(gguf_py))
    sys.path.insert(0, str(tools_dir))
    sys.path.insert(0, str(gguf_py))
    import gguf  # noqa: E402
    from gguf import quants  # noqa: E402
    import mtp_pack as pack  # noqa: E402  (the Q2_0 expert quantizer)
    import mtp_rt as rt  # noqa: E402  (the Q8_0 quantizer and the expert blob layout)

    source = Path(args.gguf).expanduser()
    if not source.is_file():
        raise fail(f"draft GGUF not found: {source}")
    out = Path(args.out).expanduser()
    complete = ("experts.bin", "dense.bin", "dense.txt")
    if not args.force and all((out / name).is_file() for name in complete):
        print(f"strata-mtp-convert: runtime already complete: {out}")
        return 0

    reader = gguf.GGUFReader(str(source))
    tensors = {t.name: t for t in reader.tensors}
    blocks = sorted({int(m.group(1)) for name in tensors if (m := re.match(r"blk\.(\d+)\.", name))})
    if len(blocks) != 1:
        raise fail("not a llama.cpp nextn MTP GGUF: expected one blk.<N>.* block")
    block = blocks[0]
    prefix = f"blk.{block}."

    def get(suffix: str):
        tensor = tensors.get(prefix + suffix)
        if tensor is None:
            raise fail(f"missing tensor {prefix}{suffix} in {source.name}")
        return tensor

    meta = reader.fields
    arch = meta["general.architecture"].contents()
    if arch != "qwen4exp":
        raise fail(f"unsupported architecture {arch!r}; expected qwen4exp")
    embd = int(meta["qwen4exp.embedding_length"].contents())
    ffn = int(meta["qwen4exp.expert_feed_forward_length"].contents())
    n_expert = int(meta["qwen4exp.expert_count"].contents())
    if f"blk.{block}.nextn.eh_proj.weight" not in tensors:
        raise fail(f"blk.{block} has no nextn.eh_proj.weight; not an MTP draft block")
    print(f"strata-mtp-convert: {source.name}: blk.{block}, {embd} hidden, {ffn} ffn, {n_expert} experts")

    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    def values_of(tensor, index: int | None = None) -> np.ndarray:
        data = tensor.data if index is None else tensor.data[index]
        return quants.dequantize(np.ascontiguousarray(data), tensor.tensor_type).astype(np.float32)

    started = time.time()
    lines: list[str] = []
    offset = 0
    dense_path = tmp / "dense.bin"
    with open(dense_path, "wb") as dense:
        def emit(name: str, kind: str, payload: bytes, rows: int, cols: int) -> None:
            nonlocal offset
            pad = (-offset) % 256
            if pad:
                dense.write(b"\0" * pad)
                offset += pad
            dense.write(payload)
            lines.append(f"{name} {kind} {rows} {cols} {offset} {len(payload)}")
            offset += len(payload)

        # fc_embedding / fc_hidden: llama.cpp fuses the checkpoint's two input
        # projections into nextn.eh_proj (embedding half first, hidden half
        # second, along the input axis).  Q8_0 blocks split on the row boundary
        # without requantizing: each row holds 2 * embd values / 32 blocks.
        eh = get("nextn.eh_proj.weight")
        if eh.tensor_type != gguf.GGMLQuantizationType.Q8_0:
            raise fail("nextn.eh_proj.weight is not Q8_0; unsupported layout")
        raw = np.ascontiguousarray(eh.data)
        rows, row_bytes = raw.shape
        if rows != embd or row_bytes * 32 // 34 != 2 * embd:
            raise fail(f"nextn.eh_proj.weight has shape {list(eh.shape)}, expected [{2 * embd}, {embd}]")
        half = row_bytes // 2
        emit("fc_embedding.weight", "q8_0", np.ascontiguousarray(raw[:, :half]).tobytes(), rows, embd)
        emit("fc_hidden.weight", "q8_0", np.ascontiguousarray(raw[:, half:]).tobytes(), rows, embd)

        for suffix, name, kind in DENSE_MAP:
            tensor = get(suffix)
            if kind == "norm":
                values = values_of(tensor).reshape(-1)
                emit(name, "f32", np.ascontiguousarray(values, dtype=np.float32).tobytes(), 1, values.size)
            elif kind == "bf16":
                values = values_of(tensor)
                if values.ndim == 1:
                    rows_, cols_ = 1, values.size
                else:
                    rows_, cols_ = values.shape
                emit(name, "bf16", to_bf16(values), rows_, cols_)
            else:
                if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
                    raw = np.ascontiguousarray(tensor.data)
                    rows_, row_bytes_ = raw.shape
                    if row_bytes_ * 32 // 34 != tensor.shape[0]:
                        raise fail(f"{prefix}{suffix}: unexpected Q8_0 row bytes")
                    emit(name, "q8_0", raw.tobytes(), rows_, tensor.shape[0])
                else:
                    values = values_of(tensor)
                    if values.ndim != 2:
                        raise fail(f"{prefix}{suffix}: expected a 2-D projection, got {list(tensor.shape)}")
                    rows_, cols_ = values.shape
                    if cols_ % 32:
                        raise fail(f"{prefix}{suffix}: {cols_} columns is not a multiple of 32")
                    emit(name, "q8_0", rt.q8_0(values), rows_, cols_)

    with open(tmp / "dense.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    gate_t, up_t, down_t = (get(name) for name in EXPERT_NAMES)
    for tensor in (gate_t, up_t, down_t):
        if len(tensor.shape) != 3 or int(tensor.shape[2]) != n_expert:
            raise fail(f"{prefix}{tensor.name}: shape {list(tensor.shape)} is not [.., .., {n_expert}]")
    per = 3 * (embd * ffn * 18 // 64)
    experts_path = tmp / "experts.bin"
    wrote = 0
    with open(experts_path, "wb") as fh:
        for e in range(n_expert):
            # GGUF layout [ne1, ne0]: gate/up are [ffn, embd], down is [embd, ffn]
            gate = values_of(gate_t, e).reshape(ffn, embd)
            up = values_of(up_t, e).reshape(ffn, embd)
            down = values_of(down_t, e).reshape(embd, ffn)
            gu = np.vstack([gate, up])
            gu_q = pack.q2_0(gu).reshape(2 * ffn, embd // 64 * 18)
            down_q = pack.q2_0(down).reshape(embd, ffn // 64 * 18)
            blob = rt.blob_of(gu_q, down_q)
            if len(blob) != per:
                raise fail(f"expert {e}: blob {len(blob)} bytes, expected {per}")
            fh.write(blob)
            wrote += 1
            if wrote % 32 == 0 or wrote == n_expert:
                print(
                    f"strata-mtp-convert: experts {wrote}/{n_expert} "
                    f"({time.time() - started:.0f} s)",
                    flush=True,
                )

    vocab = strata / "data" / "draft_vocab.bin"
    if vocab.is_file():
        shutil.copyfile(vocab, tmp / "draft_vocab.bin")
    else:
        print(
            "strata-mtp-convert: WARNING: data/draft_vocab.bin is missing; "
            "the engine may fall back to the whole head",
            file=sys.stderr,
        )

    (tmp / "convert.json").write_text(
        json.dumps(
            {
                "tool": "llm-router/tools/strata_mtp_convert.py",
                "source": str(source),
                "source_bytes": source.stat().st_size,
                "block": block,
                "hidden": embd,
                "ffn": ffn,
                "experts": n_expert,
                "expert_format": "q2_0",
                "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(
        f"strata-mtp-convert: wrote {out} (dense {offset >> 20} MiB, "
        f"experts {per * n_expert >> 20} MiB, {time.time() - started:.0f} s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
