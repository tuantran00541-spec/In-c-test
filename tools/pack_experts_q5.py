#!/usr/bin/env python3
"""Experimental routed-expert packer: symmetric Q5, group size 128, FP32 scales.

Each matrix blob is stored as:
  float32 scales[out_rows, ceil(in_cols / 128)]
  uint5   weights[out_rows, in_cols] packed row-major, LSB-first

Quantization is symmetric RTN with q in [-15, 15]. Negative values are encoded
as their 5-bit two's-complement representation. Expert records remain 4096-byte
aligned for direct I/O. Router/shared/attention/embeddings/LM-head/vision remain
unchanged BF16.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import struct

import numpy as np

ALIGN = 4096
GROUP = 128
MAGIC = b"KVLXPRT1"
VERSION = 1
DTYPE_Q5_G128 = 4
HDR = struct.Struct("<8sIIIIIIQQ")
REC = struct.Struct("<IIQQQQQQQQQ")
PAT = re.compile(
    r"language_model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def read_st_header(path: pathlib.Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def bf16_bytes_to_f32(raw: bytes, shape) -> np.ndarray:
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    bits = u << np.uint32(16)
    return bits.view(np.float32).reshape(shape)


def load_tensor(model_dir: pathlib.Path, headers, name: str, shard: str) -> np.ndarray:
    base, h = headers[shard]
    meta = h[name]
    if meta["dtype"] != "BF16":
        raise SystemExit(f"{name}: expected BF16, got {meta['dtype']}")
    a, b = meta["data_offsets"]
    with (model_dir / shard).open("rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    if len(raw) != b - a:
        raise IOError(f"short read {name}")
    return bf16_bytes_to_f32(raw, meta["shape"])


def pack_signed_q5(q: np.ndarray) -> bytes:
    flat = np.asarray(q, dtype=np.int8).reshape(-1)
    n8 = (flat.size // 8) * 8
    parts: list[bytes] = []
    if n8:
        u = (flat[:n8].astype(np.int16) & 31).astype(np.uint64).reshape(-1, 8)
        word = (
            u[:, 0]
            | (u[:, 1] << np.uint64(5))
            | (u[:, 2] << np.uint64(10))
            | (u[:, 3] << np.uint64(15))
            | (u[:, 4] << np.uint64(20))
            | (u[:, 5] << np.uint64(25))
            | (u[:, 6] << np.uint64(30))
            | (u[:, 7] << np.uint64(35))
        )
        out = np.empty((word.size, 5), dtype=np.uint8)
        for b in range(5):
            out[:, b] = ((word >> np.uint64(8 * b)) & np.uint64(255)).astype(np.uint8)
        parts.append(out.tobytes(order="C"))
    if n8 != flat.size:
        acc = 0
        bits = 0
        tail = bytearray()
        for value in flat[n8:]:
            acc |= (int(value) & 31) << bits
            bits += 5
            while bits >= 8:
                tail.append(acc & 255)
                acc >>= 8
                bits -= 8
        if bits:
            tail.append(acc & 255)
        parts.append(bytes(tail))
    packed = b"".join(parts)
    expected = (flat.size * 5 + 7) // 8
    if len(packed) != expected:
        raise AssertionError((len(packed), expected, flat.size))
    return packed


def quantize_g128(w: np.ndarray) -> bytes:
    if w.ndim != 2:
        raise ValueError(w.shape)
    rows, cols = map(int, w.shape)
    groups = (cols + GROUP - 1) // GROUP
    scales = np.empty((rows, groups), dtype="<f4")
    q = np.empty((rows, cols), dtype=np.int8)
    for g in range(groups):
        a = g * GROUP
        b = min(cols, a + GROUP)
        block = w[:, a:b]
        maxabs = np.max(np.abs(block), axis=1)
        s = np.where(maxabs > 0, maxabs / 15.0, 1.0).astype(np.float32)
        scales[:, g] = s
        q[:, a:b] = np.rint(block / s[:, None]).clip(-15, 15).astype(np.int8)
    return scales.tobytes(order="C") + pack_signed_q5(q)


def load_existing(idx_path: pathlib.Path):
    if not idx_path.exists():
        return None, []
    raw = idx_path.read_bytes()
    if len(raw) < HDR.size:
        raise SystemExit("bad existing experts.idx")
    h = HDR.unpack_from(raw, 0)
    if h[0] != MAGIC or h[1] != VERSION or h[2] != ALIGN or h[6] != DTYPE_Q5_G128 or h[7] != HDR.size:
        raise SystemExit("incompatible existing Q5 experts.idx")
    recs = []
    off = h[7]
    for _ in range(h[5]):
        recs.append(REC.unpack_from(raw, off))
        off += REC.size
    return h, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=pathlib.Path)
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--layer", type=int, action="append")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    idx = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    wanted = set(args.layer) if args.layer is not None else None
    entries = {}
    for name, shard in wm.items():
        m = PAT.match(name)
        if not m:
            continue
        layer, expert = int(m.group(1)), int(m.group(2))
        part = m.group(3)
        if wanted is not None and layer not in wanted:
            continue
        entries.setdefault((layer, expert), {})[part] = (name, shard)
    if not entries:
        raise SystemExit("No routed experts matched")
    missing = [k for k, v in entries.items() if set(v) != {"gate_proj", "up_proj", "down_proj"}]
    if missing:
        raise SystemExit(f"Incomplete experts, first: {missing[:3]}")

    cfg = json.loads((args.model_dir / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    n_layers = int(tc["num_hidden_layers"])
    n_experts = int(tc["n_routed_experts"])
    binp = args.out_dir / "experts.bin"
    idxp = args.out_dir / "experts.idx"
    old_h, recs = load_existing(idxp) if args.append else (None, [])
    if old_h is not None and (old_h[3] != n_layers or old_h[4] != n_experts):
        raise SystemExit("existing Q5 expert store dimensions mismatch")
    existing = {(r[0], r[1]) for r in recs}
    entries = {k: v for k, v in entries.items() if k not in existing}
    if not entries:
        print("no new Q5 expert records to append")
        return

    headers = {}
    for shard in sorted(set(s for v in entries.values() for _, s in v.values())):
        p = args.model_dir / shard
        if not p.exists():
            raise FileNotFoundError(p)
        headers[shard] = read_st_header(p)

    mode = "r+b" if args.append and binp.exists() else "wb"
    with binp.open(mode) as out:
        if mode == "r+b":
            out.seek(0, os.SEEK_END)
        for (layer, expert), parts in sorted(entries.items()):
            start = align_up(out.tell())
            out.write(b"\0" * (start - out.tell()))
            offsets = {}
            sizes = {}
            for part in ("gate_proj", "up_proj", "down_proj"):
                name, shard = parts[part]
                w = load_tensor(args.model_dir, headers, name, shard)
                blob = quantize_g128(w)
                offsets[part] = out.tell() - start
                sizes[part] = len(blob)
                out.write(blob)
                del w, blob
            payload = out.tell() - start
            end = align_up(out.tell())
            out.write(b"\0" * (end - out.tell()))
            recs.append(
                (
                    layer,
                    expert,
                    start,
                    end - start,
                    payload,
                    offsets["gate_proj"],
                    sizes["gate_proj"],
                    offsets["up_proj"],
                    sizes["up_proj"],
                    offsets["down_proj"],
                    sizes["down_proj"],
                )
            )

    with idxp.open("wb") as f:
        f.write(
            HDR.pack(
                MAGIC,
                VERSION,
                ALIGN,
                n_layers,
                n_experts,
                len(recs),
                DTYPE_Q5_G128,
                HDR.size,
                os.path.getsize(binp),
            )
        )
        for r in recs:
            f.write(REC.pack(*r))
    print(
        f"Q5_G128 expert records={len(recs)} data={os.path.getsize(binp)/1024**3:.3f} GiB "
        f"index={os.path.getsize(idxp)} bytes scales=fp32"
    )


if __name__ == "__main__":
    main()
