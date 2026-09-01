#!/usr/bin/env python3
"""Compare a shared-Q8 sidecar against the BF16 shared experts in trunk.bin.

This is a pre-generation quality gate for the real packed model. It reads one
layer at a time, never needs original safetensors, and measures the full shared
MLP output on deterministic random hidden vectors. It does not claim task-level
model quality; it only catches unexpectedly bad quantisation/runtime assets.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import struct
import numpy as np

THDR = struct.Struct("<8s4I2Q")
TREC = struct.Struct("<8I3Q")
EHDR = struct.Struct("<8sIIIIIIQQ")
EREC = struct.Struct("<IIQQQQQQQQQ")
SG, SU, SD = 32, 33, 34


def read_trunk(path: pathlib.Path):
    raw = path.read_bytes()
    h = THDR.unpack_from(raw, 0)
    if h[0] != b"KVLTRNK1" or h[1] != 1:
        raise SystemExit("bad trunk.idx")
    recs = {}
    off = h[5]
    for _ in range(h[3]):
        r = TREC.unpack_from(raw, off)
        off += TREC.size
        if r[1] in (SG, SU, SD):
            recs[(r[0], r[1])] = r
    return recs


def read_sidecar(path: pathlib.Path):
    raw = path.read_bytes()
    h = EHDR.unpack_from(raw, 0)
    if h[0] != b"KVLXPRT1" or h[1] != 1 or h[6] != 3:
        raise SystemExit("bad shared_q8.idx")
    recs = {}
    off = h[7]
    for _ in range(h[5]):
        r = EREC.unpack_from(raw, off)
        off += EREC.size
        if r[1] != 0:
            raise SystemExit("shared sidecar must use expert id 0")
        recs[r[0]] = r
    return recs


def bf16_matrix(f, r) -> np.ndarray:
    rows, cols = r[4], r[5]
    f.seek(r[8])
    raw = f.read(r[10])
    if len(raw) != r[10] or len(raw) != rows * cols * 2:
        raise SystemExit("short/malformed BF16 trunk tensor")
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u << np.uint32(16)).view(np.float32).reshape(rows, cols)


def q8_matrix(blob: bytes, off: int, nbytes: int, rows: int, cols: int) -> np.ndarray:
    scale_bytes = rows * 4
    if nbytes != scale_bytes + rows * cols:
        raise SystemExit("malformed Q8 matrix span")
    scales = np.frombuffer(blob, dtype="<f4", count=rows, offset=off).copy()
    q = np.frombuffer(blob, dtype=np.int8, count=rows * cols,
                      offset=off + scale_bytes).astype(np.float32).reshape(rows, cols)
    return q * scales[:, None]


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rel_rms(got: np.ndarray, ref: np.ndarray) -> float:
    d = got.astype(np.float64) - ref.astype(np.float64)
    rms = np.sqrt(np.mean(d * d))
    rr = ref.astype(np.float64)
    sig = np.sqrt(np.mean(rr * rr)) + 1e-30
    return float(rms / sig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_dir", type=pathlib.Path)
    ap.add_argument("sidecar_dir", type=pathlib.Path)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=260901)
    ap.add_argument("--max-rel", type=float, default=0.05,
                    help="fail if any sampled full shared-MLP output exceeds this relative RMS")
    args = ap.parse_args()
    if args.samples <= 0 or args.max_rel <= 0:
        raise SystemExit("samples and max-rel must be positive")

    trunk_recs = read_trunk(args.packed_dir / "trunk.idx")
    side_recs = read_sidecar(args.sidecar_dir / "shared_q8.idx")
    layers = sorted({layer for layer, kind in trunk_recs if kind == SG})
    if not layers:
        raise SystemExit("no BF16 shared experts in trunk")
    if set(layers) != set(side_recs):
        raise SystemExit("trunk/shared sidecar layer sets differ")

    rng = np.random.default_rng(args.seed)
    all_rel = []
    with (args.packed_dir / "trunk.bin").open("rb") as tf:
        qblob = (args.sidecar_dir / "shared_q8.bin").read_bytes()
        for layer in layers:
            gr, ur, dr = (trunk_recs[(layer, k)] for k in (SG, SU, SD))
            gate = bf16_matrix(tf, gr)
            up = bf16_matrix(tf, ur)
            down = bf16_matrix(tf, dr)
            er = side_recs[layer]
            base = er[2]
            qgate = q8_matrix(qblob, base + er[5], er[6], gate.shape[0], gate.shape[1])
            qup = q8_matrix(qblob, base + er[7], er[8], up.shape[0], up.shape[1])
            qdown = q8_matrix(qblob, base + er[9], er[10], down.shape[0], down.shape[1])

            layer_rel = []
            for _ in range(args.samples):
                # Unit-scale vectors are intentional: relative error is invariant to
                # modest input scaling for the linear pieces while SiLU sees realistic
                # positive/negative activation spread.
                x = rng.normal(0.0, 1.0, size=(gate.shape[1],)).astype(np.float32)
                ref = down @ (silu(gate @ x) * (up @ x))
                got = qdown @ (silu(qgate @ x) * (qup @ x))
                layer_rel.append(rel_rms(got, ref))
            all_rel.extend(layer_rel)
            print(
                f"layer={layer:2d} samples={args.samples} "
                f"mean_rel_rms={statistics.fmean(layer_rel):.6f} "
                f"max_rel_rms={max(layer_rel):.6f}"
            )

    worst = max(all_rel)
    median = statistics.median(all_rel)
    mean = statistics.fmean(all_rel)
    print(
        f"shared_q8_real samples={len(all_rel)} mean_rel_rms={mean:.6f} "
        f"median_rel_rms={median:.6f} worst_rel_rms={worst:.6f} "
        f"threshold={args.max_rel:.6f}"
    )
    if worst >= args.max_rel:
        raise SystemExit("SHARED_Q8_REAL_NUMERICAL_FAIL")
    print("SHARED_Q8_REAL_NUMERICAL_PASS")


if __name__ == "__main__":
    main()
