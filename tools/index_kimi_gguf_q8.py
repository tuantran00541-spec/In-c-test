#!/usr/bin/env python3
"""Create a tiny KVL sidecar index for direct expert streaming from Kimi Q8_0 GGUF.

No model tensor bytes are copied. The KVL records describe the in-cache aligned
layout, while appended KvlGgufQ8Source records describe three aligned positioned
reads (gate/up/down) from the original GGUF file.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct

from gguf import GGMLQuantizationType, GGUFReader

ALIGN = 4096
MAGIC = b"KVLXPRT1"
VERSION = 1
DTYPE_GGUF_Q8_0 = 5
HDR = struct.Struct("<8sIIIIIIQQ")
REC = struct.Struct("<IIQQQQQQQQQ")
SRC = struct.Struct("<QQQQQQQQQ")


def align_down(x: int, a: int = ALIGN) -> int:
    return x // a * a


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def part_info(t, expert: int, n_experts: int) -> dict[str, int]:
    if t.tensor_type != GGMLQuantizationType.Q8_0:
        raise SystemExit(f"{t.name}: expected Q8_0, got {t.tensor_type.name}")
    if t.n_bytes % n_experts:
        raise SystemExit(f"{t.name}: tensor bytes not divisible by experts")
    slice_bytes = int(t.n_bytes // n_experts)
    # Quantized ReaderTensor data shape is [expert, rows, quantized-row-bytes].
    if len(t.data.shape) != 3 or int(t.data.shape[0]) != n_experts:
        raise SystemExit(f"{t.name}: expert is not outer physical axis: {t.data.shape}")
    raw = int(t.data_offset) + expert * slice_bytes
    src = align_down(raw)
    sub = raw - src
    read = align_up(sub + slice_bytes)
    return {"raw": raw, "src": src, "sub": sub, "read": read, "bytes": slice_bytes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=pathlib.Path)
    ap.add_argument("out_idx", type=pathlib.Path)
    ap.add_argument("--summary-json", type=pathlib.Path)
    args = ap.parse_args()

    r = GGUFReader(args.gguf)
    by_name = {t.name: t for t in r.tensors}

    # Kimi-VL-A3B / DeepSeek2-lite architecture invariants.
    n_layers = 27
    n_experts = 64
    first_moe = 1
    last_moe = 26

    tensors: dict[tuple[int, str], object] = {}
    for layer in range(first_moe, last_moe + 1):
        for part in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{part}_exps.weight"
            t = by_name.get(name)
            if t is None:
                raise SystemExit(f"missing {name}")
            tensors[(layer, part)] = t

    records = []
    sources = []
    sample = None
    for layer in range(first_moe, last_moe + 1):
        for expert in range(n_experts):
            p = {part: part_info(tensors[(layer, part)], expert, n_experts)
                 for part in ("gate", "up", "down")}
            dst = 0
            dst_start = {}
            for part in ("gate", "up", "down"):
                dst = align_up(dst)
                dst_start[part] = dst
                dst += p[part]["read"]
            read_bytes = dst
            payload_bytes = sum(p[part]["bytes"] for part in p)
            gate_off = dst_start["gate"] + p["gate"]["sub"]
            up_off = dst_start["up"] + p["up"]["sub"]
            down_off = dst_start["down"] + p["down"]["sub"]
            file_sort = min(p[part]["src"] for part in p)
            records.append((
                layer, expert, file_sort, read_bytes, payload_bytes,
                gate_off, p["gate"]["bytes"],
                up_off, p["up"]["bytes"],
                down_off, p["down"]["bytes"],
            ))
            sources.append((
                p["gate"]["src"], p["gate"]["read"], dst_start["gate"],
                p["up"]["src"], p["up"]["read"], dst_start["up"],
                p["down"]["src"], p["down"]["read"], dst_start["down"],
            ))
            if layer == 1 and expert == 0:
                sample = {part: p[part] for part in p}
                sample["record_read_bytes"] = read_bytes
                sample["payload_bytes"] = payload_bytes
                sample["gate_off"] = gate_off
                sample["up_off"] = up_off
                sample["down_off"] = down_off

    args.out_idx.parent.mkdir(parents=True, exist_ok=True)
    data_bytes = os.path.getsize(args.gguf)
    with open(args.out_idx, "wb") as f:
        f.write(HDR.pack(MAGIC, VERSION, ALIGN, n_layers, n_experts,
                         len(records), DTYPE_GGUF_Q8_0, HDR.size, data_bytes))
        for rec in records:
            f.write(REC.pack(*rec))
        for src in sources:
            f.write(SRC.pack(*src))

    summary = {
        "gguf": str(args.gguf),
        "gguf_bytes": data_bytes,
        "index": str(args.out_idx),
        "index_bytes": args.out_idx.stat().st_size,
        "dtype": DTYPE_GGUF_Q8_0,
        "layers": n_layers,
        "moe_layers": last_moe - first_moe + 1,
        "experts": n_experts,
        "records": len(records),
        "sample_L1E0": sample,
        "claim_boundary": "Index only; no model tensor bytes are copied or requantized.",
    }
    print(
        f"KIMI_GGUF_Q8_INDEX records={len(records)} idx_bytes={summary['index_bytes']} "
        f"slot_mib={records[0][3] / 1048576.0:.6f} payload_mib={records[0][4] / 1048576.0:.6f}"
    )
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
