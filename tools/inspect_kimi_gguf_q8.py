#!/usr/bin/env python3
"""Inspect a local Kimi-VL Q8_0 GGUF for SSD expert-streaming compatibility.

This does not run llama.cpp and does not claim model-quality equivalence. It answers
whether the GGUF physical layout can feed this project's sparse ExpertStore design:
- DeepSeek2/Kimi metadata matches the expected 27-layer / 64-expert / top-6 model.
- Routed MoE weights are Q8_0 tensors with expert as the outermost dimension.
- Each expert slice is physically contiguous and can be addressed by file offset.
- 4096-byte no-buffering reads can cover each slice with small bounded padding.
- A tiny Q8_0 dequantization sample is finite/non-zero.

The current project Q8 format is deliberately different (per-row FP32 scale + int8),
so a positive result means "direct-streamable with a Q8_0 kernel/reader", not
"drop-in compatible with KVL_DTYPE_Q8_ROW".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize

ALIGN = 4096
EXPECTED_LAYERS = 27
EXPECTED_EXPERTS = 64
EXPECTED_USED = 6
EXPECTED_EXPERT_FF = 1408

ROUTED_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<part>down|gate|up|gate_up)_exps\.weight$"
)


def field_value(reader: GGUFReader, key: str):
    f = reader.get_field(key)
    return None if f is None else f.contents()


def align_down(x: int, a: int = ALIGN) -> int:
    return (x // a) * a


def align_up(x: int, a: int = ALIGN) -> int:
    return ((x + a - 1) // a) * a


def qname(t) -> str:
    try:
        return t.tensor_type.name
    except Exception:
        return str(t.tensor_type)


def inspect(path: Path) -> dict:
    reader = GGUFReader(path, mode="r")
    arch = field_value(reader, "general.architecture")
    block_count = field_value(reader, "deepseek2.block_count")
    expert_count = field_value(reader, "deepseek2.expert_count")
    expert_used = field_value(reader, "deepseek2.expert_used_count")
    expert_ff = field_value(reader, "deepseek2.expert_feed_forward_length")
    leading_dense = field_value(reader, "deepseek2.leading_dense_block_count")

    metadata_checks = {
        "architecture_deepseek2": arch == "deepseek2",
        "block_count_27": block_count == EXPECTED_LAYERS,
        "expert_count_64": expert_count == EXPECTED_EXPERTS,
        "expert_used_count_6": expert_used == EXPECTED_USED,
        "expert_ff_1408": expert_ff == EXPECTED_EXPERT_FF,
        "leading_dense_block_count_1": leading_dense in (None, 1),
    }

    by_layer: dict[int, dict[str, object]] = {}
    routed = []
    for t in reader.tensors:
        m = ROUTED_RE.match(t.name)
        if not m:
            continue
        layer = int(m.group("layer"))
        part = m.group("part")
        logical_shape = [int(x) for x in t.shape.tolist()]
        numpy_shape = [int(x) for x in t.data.shape]
        outer_experts = bool(numpy_shape and numpy_shape[0] == EXPECTED_EXPERTS)
        divisible = t.n_bytes % EXPECTED_EXPERTS == 0
        slice_bytes = t.n_bytes // EXPECTED_EXPERTS if divisible else None
        contiguous = False
        if outer_experts and divisible:
            contiguous = bool(t.data[0].flags.c_contiguous and t.data[-1].flags.c_contiguous)

        entry = {
            "name": t.name,
            "layer": layer,
            "part": part,
            "type": qname(t),
            "logical_shape_ggml": logical_shape,
            "numpy_byte_shape": numpy_shape,
            "tensor_bytes": int(t.n_bytes),
            "absolute_data_offset": int(t.data_offset),
            "outer_expert_axis": outer_experts,
            "expert_slice_bytes": int(slice_bytes) if slice_bytes is not None else None,
            "expert_slice_mib": (float(slice_bytes) / 2**20) if slice_bytes is not None else None,
            "expert_slice_contiguous": contiguous,
        }
        routed.append(entry)
        by_layer.setdefault(layer, {})[part] = t

    expected_moe_layers = set(range(1, EXPECTED_LAYERS))
    actual_moe_layers = set(by_layer)
    layer_layout = {}
    all_q8 = True
    all_contiguous = True
    all_complete = True
    io_overheads = []
    expert_total_bytes_by_layer = {}

    for layer in sorted(actual_moe_layers):
        parts = by_layer[layer]
        split = {"down", "gate", "up"}.issubset(parts)
        fused = {"down", "gate_up"}.issubset(parts)
        complete = split or fused
        all_complete &= complete
        names = sorted(parts)
        layer_total = 0
        layer_q8 = True
        layer_contiguous = True
        for part, t in parts.items():
            layer_q8 &= t.tensor_type == GGMLQuantizationType.Q8_0
            layer_contiguous &= bool(
                t.n_bytes % EXPECTED_EXPERTS == 0
                and t.data.shape[0] == EXPECTED_EXPERTS
                and t.data[0].flags.c_contiguous
            )
            if t.n_bytes % EXPECTED_EXPERTS != 0:
                continue
            sb = int(t.n_bytes // EXPECTED_EXPERTS)
            layer_total += sb
            for e in (0, EXPECTED_EXPERTS // 2, EXPECTED_EXPERTS - 1):
                start = int(t.data_offset) + e * sb
                end = start + sb
                envelope = align_up(end) - align_down(start)
                io_overheads.append(envelope - sb)
        all_q8 &= layer_q8
        all_contiguous &= layer_contiguous
        expert_total_bytes_by_layer[str(layer)] = layer_total
        layer_layout[str(layer)] = {
            "parts": names,
            "split_gate_up": split,
            "fused_gate_up": fused,
            "complete": complete,
            "all_q8_0": layer_q8,
            "expert_slices_contiguous": layer_contiguous,
            "expert_total_bytes": layer_total,
            "expert_total_mib": layer_total / 2**20,
        }

    sample = None
    if actual_moe_layers:
        layer = min(actual_moe_layers)
        # Prefer down because its shape is unambiguous in both split/fused layouts.
        t = by_layer[layer].get("down") or next(iter(by_layer[layer].values()))
        if t.tensor_type == GGMLQuantizationType.Q8_0 and t.data.shape[0] == EXPECTED_EXPERTS:
            # Dequantize only two rows from expert 0; this touches kilobytes, not a whole tensor.
            raw = np.asarray(t.data[0, :2])
            dq = dequantize(raw, t.tensor_type)
            sample = {
                "tensor": t.name,
                "raw_bytes": int(raw.nbytes),
                "dequant_shape": [int(x) for x in dq.shape],
                "finite": bool(np.isfinite(dq).all()),
                "nonzero": bool(np.count_nonzero(dq) > 0),
                "min": float(np.min(dq)),
                "max": float(np.max(dq)),
                "mean_abs": float(np.mean(np.abs(dq))),
            }

    io_stats = None
    if io_overheads:
        io_stats = {
            "alignment": ALIGN,
            "sampled_slice_reads": len(io_overheads),
            "min_padding_bytes": int(min(io_overheads)),
            "max_padding_bytes": int(max(io_overheads)),
            "mean_padding_bytes": float(sum(io_overheads) / len(io_overheads)),
        }

    routed_bytes = sum(int(x["tensor_bytes"]) for x in routed)
    direct_streamable = bool(
        all(metadata_checks.values())
        and actual_moe_layers == expected_moe_layers
        and all_complete
        and all_q8
        and all_contiguous
        and sample is not None
        and sample["finite"]
        and sample["nonzero"]
    )

    # Current KVL Q8_ROW uses one FP32 scale per output row. GGUF Q8_0 uses 32-value blocks
    # with a per-block scale, so binary records are intentionally not interchangeable.
    result = {
        "schema": "kimi-gguf-q8-compat-v1",
        "file": str(path),
        "file_bytes": int(path.stat().st_size),
        "gguf_data_offset": int(reader.data_offset),
        "tensor_count": len(reader.tensors),
        "metadata": {
            "architecture": arch,
            "block_count": block_count,
            "expert_count": expert_count,
            "expert_used_count": expert_used,
            "expert_feed_forward_length": expert_ff,
            "leading_dense_block_count": leading_dense,
        },
        "metadata_checks": metadata_checks,
        "moe_layers_found": sorted(actual_moe_layers),
        "moe_layer_count": len(actual_moe_layers),
        "expected_moe_layer_count": EXPECTED_LAYERS - 1,
        "routed_tensor_count": len(routed),
        "routed_tensor_bytes": routed_bytes,
        "routed_tensor_gib": routed_bytes / 2**30,
        "routed_tensors": routed,
        "layer_layout": layer_layout,
        "expert_total_bytes_by_layer": expert_total_bytes_by_layer,
        "direct_io_alignment_probe": io_stats,
        "q8_0_dequant_sample": sample,
        "verdict": {
            "direct_streamable_from_gguf": direct_streamable,
            "drop_in_current_kvl_q8_row": False,
            "needs_new_q8_0_kernel_or_offline_repack": True,
            "recommended_path": (
                "add GGUF tensor index + Q8_0 expert GEMV and read aligned expert slices directly"
                if direct_streamable
                else "inspect failed checks before integration"
            ),
        },
        "claim_boundary": (
            "Physical-layout compatibility only. This probe does not establish end-to-end logits, "
            "quality, speed, or Windows no-buffering performance."
        ),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    result = inspect(args.gguf.resolve())
    v = result["verdict"]
    print(
        "KIMI_GGUF_Q8_COMPAT "
        f"streamable={int(v['direct_streamable_from_gguf'])} "
        f"dropin={int(v['drop_in_current_kvl_q8_row'])} "
        f"layers={result['moe_layer_count']}/{result['expected_moe_layer_count']} "
        f"routed_gib={result['routed_tensor_gib']:.6f}"
    )
    if result["q8_0_dequant_sample"]:
        s = result["q8_0_dequant_sample"]
        print(
            "KIMI_GGUF_Q8_SAMPLE "
            f"tensor={s['tensor']} shape={s['dequant_shape']} finite={int(s['finite'])} "
            f"nonzero={int(s['nonzero'])} mean_abs={s['mean_abs']:.8g}"
        )
    if result["direct_io_alignment_probe"]:
        io = result["direct_io_alignment_probe"]
        print(
            "KIMI_GGUF_Q8_DIRECTIO "
            f"align={io['alignment']} max_padding={io['max_padding_bytes']} "
            f"mean_padding={io['mean_padding_bytes']:.2f}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if v["direct_streamable_from_gguf"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
