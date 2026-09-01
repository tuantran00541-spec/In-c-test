#!/usr/bin/env python3
"""Build a Q8 shared-expert sidecar directly from an existing packed trunk.

No original safetensors are required. The input trunk must contain BF16 shared
expert gate/up/down records. The output deliberately reuses KVLXPRT1 with one
expert id (0) per MoE layer so the established direct-I/O expert store/cache can
be reused by research runtimes.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import numpy as np

ALIGN = 4096
TRUNK_MAGIC = b"KVLTRNK1"
TRUNK_VERSION = 1
TRUNK_DTYPE_BF16 = 1
TRUNK_HDR = struct.Struct("<8s4I2Q")
TRUNK_REC = struct.Struct("<8I3Q")
SHARED_GATE = 32
SHARED_UP = 33
SHARED_DOWN = 34

EXPERT_MAGIC = b"KVLXPRT1"
EXPERT_VERSION = 1
EXPERT_DTYPE_Q8_ROW = 3
EXPERT_HDR = struct.Struct("<8sIIIIIIQQ")
EXPERT_REC = struct.Struct("<IIQQQQQQQQQ")


def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def read_trunk_index(path: pathlib.Path):
    raw = path.read_bytes()
    if len(raw) < TRUNK_HDR.size:
        raise SystemExit("bad trunk.idx: short header")
    magic, version, align, nrec, _reserved, roff, data_bytes = TRUNK_HDR.unpack_from(raw, 0)
    if magic != TRUNK_MAGIC or version != TRUNK_VERSION or align != ALIGN or roff != TRUNK_HDR.size:
        raise SystemExit("incompatible trunk.idx")
    need = roff + nrec * TRUNK_REC.size
    if len(raw) < need:
        raise SystemExit("bad trunk.idx: truncated records")
    recs = []
    off = roff
    for _ in range(nrec):
        recs.append(TRUNK_REC.unpack_from(raw, off))
        off += TRUNK_REC.size
    return recs, data_bytes


def bf16_raw_to_f32(raw: bytes, rows: int, cols: int) -> np.ndarray:
    expected = rows * cols * 2
    if len(raw) != expected:
        raise ValueError(f"BF16 payload mismatch: got {len(raw)}, expected {expected}")
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32, copy=False)
    return (u16 << np.uint32(16)).view(np.float32).reshape(rows, cols)


def quantize_rows(w: np.ndarray) -> bytes:
    if w.ndim != 2:
        raise ValueError(f"expected matrix, got shape={w.shape}")
    maxabs = np.max(np.abs(w), axis=1)
    scales = np.where(maxabs > 0, maxabs / 127.0, 1.0).astype("<f4")
    q = np.rint(w / scales[:, None]).clip(-127, 127).astype(np.int8)
    return scales.tobytes(order="C") + q.tobytes(order="C")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_dir", type=pathlib.Path,
                    help="directory containing existing trunk.bin/trunk.idx")
    ap.add_argument("out_dir", type=pathlib.Path,
                    help="output directory for shared_q8.bin/shared_q8.idx")
    args = ap.parse_args()

    trunk_bin = args.packed_dir / "trunk.bin"
    trunk_idx = args.packed_dir / "trunk.idx"
    if not trunk_bin.is_file() or not trunk_idx.is_file():
        raise SystemExit("missing trunk.bin/trunk.idx")

    recs, declared_bytes = read_trunk_index(trunk_idx)
    actual_bytes = trunk_bin.stat().st_size
    if actual_bytes < declared_bytes:
        raise SystemExit(f"trunk.bin shorter than index: {actual_bytes} < {declared_bytes}")

    by_layer = {}
    max_layer = -1
    for r in recs:
        layer, kind, dtype, ndim, d0, d1, d2, d3, file_off, read_bytes, payload_bytes = r
        if kind not in (SHARED_GATE, SHARED_UP, SHARED_DOWN):
            continue
        if layer == 0xFFFFFFFF:
            raise SystemExit("shared tensor unexpectedly marked global")
        if dtype != TRUNK_DTYPE_BF16 or ndim != 2 or d0 <= 0 or d1 <= 0:
            raise SystemExit(f"L{layer} kind={kind}: expected 2D BF16 matrix")
        if payload_bytes != d0 * d1 * 2:
            raise SystemExit(f"L{layer} kind={kind}: payload/shape mismatch")
        by_layer.setdefault(layer, {})[kind] = r
        max_layer = max(max_layer, layer)

    complete = []
    for layer, parts in sorted(by_layer.items()):
        missing = {SHARED_GATE, SHARED_UP, SHARED_DOWN} - set(parts)
        if missing:
            raise SystemExit(f"L{layer}: incomplete shared expert, missing kinds={sorted(missing)}")
        g, u, d = parts[SHARED_GATE], parts[SHARED_UP], parts[SHARED_DOWN]
        if (g[4], g[5]) != (u[4], u[5]):
            raise SystemExit(f"L{layer}: shared gate/up shapes differ")
        # gate/up are [I,H], down must be [H,I].
        if (d[4], d[5]) != (g[5], g[4]):
            raise SystemExit(f"L{layer}: shared down shape is not transpose-compatible")
        complete.append((layer, parts))
    if not complete:
        raise SystemExit("no shared expert records found in trunk")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_bin = args.out_dir / "shared_q8.bin"
    out_idx = args.out_dir / "shared_q8.idx"
    out_recs = []
    bf16_payload_total = 0
    q8_payload_total = 0

    with trunk_bin.open("rb") as src, out_bin.open("wb") as dst:
        for layer, parts in complete:
            start = align_up(dst.tell())
            if start > dst.tell():
                dst.write(b"\0" * (start - dst.tell()))
            offsets = {}
            sizes = {}
            for kind, name in ((SHARED_GATE, "gate"), (SHARED_UP, "up"), (SHARED_DOWN, "down")):
                r = parts[kind]
                rows, cols = r[4], r[5]
                src.seek(r[8])
                raw = src.read(r[10])
                if len(raw) != r[10]:
                    raise SystemExit(f"L{layer} {name}: short trunk read")
                w = bf16_raw_to_f32(raw, rows, cols)
                blob = quantize_rows(w)
                offsets[name] = dst.tell() - start
                sizes[name] = len(blob)
                dst.write(blob)
                bf16_payload_total += len(raw)
                q8_payload_total += len(blob)
            payload = dst.tell() - start
            end = align_up(dst.tell())
            if end > dst.tell():
                dst.write(b"\0" * (end - dst.tell()))
            out_recs.append((
                layer, 0, start, end - start, payload,
                offsets["gate"], sizes["gate"],
                offsets["up"], sizes["up"],
                offsets["down"], sizes["down"],
            ))
            print(f"packed shared Q8 L={layer:2d} record={(end-start)/1048576:.2f} MiB")

    data_bytes = out_bin.stat().st_size
    with out_idx.open("wb") as f:
        f.write(EXPERT_HDR.pack(
            EXPERT_MAGIC, EXPERT_VERSION, ALIGN, max_layer + 1, 1,
            len(out_recs), EXPERT_DTYPE_Q8_ROW, EXPERT_HDR.size, data_bytes,
        ))
        for r in out_recs:
            f.write(EXPERT_REC.pack(*r))

    ratio = q8_payload_total / bf16_payload_total if bf16_payload_total else 0.0
    saved = bf16_payload_total - q8_payload_total
    print(
        f"shared_q8 records={len(out_recs)} data={data_bytes/1073741824:.3f} GiB "
        f"payload_ratio={ratio:.4f} saved_payload={saved/1048576:.2f} MiB "
        f"index={out_idx.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
