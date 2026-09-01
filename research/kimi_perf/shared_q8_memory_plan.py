#!/usr/bin/env python3
"""Compute cache-residency tradeoffs for shared-expert Q8 from packed indexes."""
from __future__ import annotations

import argparse
import pathlib
import struct

ALIGN = 4096
MIB = 1024 * 1024
GLOBAL = 0xFFFFFFFF
THDR = struct.Struct("<8s4I2Q")
TREC = struct.Struct("<8I3Q")
EHDR = struct.Struct("<8sIIIIIIQQ")
EREC = struct.Struct("<IIQQQQQQQQQ")
SHARED = {32, 33, 34}


def align_up(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def trunk_records(path: pathlib.Path):
    raw = path.read_bytes()
    h = THDR.unpack_from(raw, 0)
    if h[0] != b"KVLTRNK1" or h[1] != 1 or h[2] != ALIGN:
        raise SystemExit("bad trunk.idx")
    out = []
    off = h[5]
    for _ in range(h[3]):
        out.append(TREC.unpack_from(raw, off))
        off += TREC.size
    return out


def actual_sidecar_read_bytes(path: pathlib.Path) -> int:
    raw = path.read_bytes()
    h = EHDR.unpack_from(raw, 0)
    if h[0] != b"KVLXPRT1" or h[1] != 1 or h[6] != 3:
        raise SystemExit("bad shared_q8.idx")
    total = 0
    off = h[7]
    for _ in range(h[5]):
        r = EREC.unpack_from(raw, off)
        off += EREC.size
        if r[1] != 0:
            raise SystemExit("shared sidecar expert id must be zero")
        total += r[3]
    return total


def projected_q8_bytes(recs) -> int:
    by_layer = {}
    for r in recs:
        if r[0] != GLOBAL and r[1] in SHARED:
            by_layer.setdefault(r[0], {})[r[1]] = r
    total = 0
    for layer, p in sorted(by_layer.items()):
        if set(p) != SHARED:
            raise SystemExit(f"L{layer}: incomplete shared tensors")
        g, u, d = p[32], p[33], p[34]
        if g[2] != 1 or u[2] != 1 or d[2] != 1 or g[3] != 2 or u[3] != 2 or d[3] != 2:
            raise SystemExit(f"L{layer}: shared tensors must be 2D BF16")
        if (g[4], g[5]) != (u[4], u[5]) or (d[4], d[5]) != (g[5], g[4]):
            raise SystemExit(f"L{layer}: incompatible shared shapes")
        gi, gh = g[4], g[5]
        gate_q8 = gi * 4 + gi * gh
        up_q8 = gate_q8
        down_q8 = gh * 4 + gh * gi
        total += align_up(gate_q8 + up_q8 + down_q8)
    return total


def mib(n: int) -> float:
    return n / MIB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_dir", type=pathlib.Path)
    ap.add_argument("--sidecar-dir", type=pathlib.Path)
    ap.add_argument("--baseline-expert-cache-mib", type=int, default=512)
    args = ap.parse_args()
    if args.baseline_expert_cache_mib <= 0:
        raise SystemExit("baseline expert cache must be positive")

    recs = trunk_records(args.packed_dir / "trunk.idx")
    nonglobal = [r for r in recs if r[0] != GLOBAL]
    old_trunk = sum(r[9] for r in nonglobal)
    shared_bf16 = sum(r[9] for r in nonglobal if r[1] in SHARED)
    nonshared = old_trunk - shared_bf16
    projected = projected_q8_bytes(recs)
    sidecar = projected
    source = "projected"
    if args.sidecar_dir is not None:
        sidecar = actual_sidecar_read_bytes(args.sidecar_dir / "shared_q8.idx")
        source = "actual"

    saved = shared_bf16 - sidecar
    safe_extra_mib = max(0, saved // MIB)
    suggested = args.baseline_expert_cache_mib + safe_extra_mib
    old_combined = old_trunk + args.baseline_expert_cache_mib * MIB
    new_combined = nonshared + sidecar + suggested * MIB

    print(f"trunk_non_global_read_mib={mib(old_trunk):.4f}")
    print(f"shared_bf16_read_mib={mib(shared_bf16):.4f}")
    print(f"trunk_without_shared_mib={mib(nonshared):.4f}")
    print(f"shared_q8_{source}_read_mib={mib(sidecar):.4f}")
    print(f"shared_q8_projected_read_mib={mib(projected):.4f}")
    print(f"resident_saving_mib={mib(saved):.4f}")
    print(f"baseline_routed_cache_mib={args.baseline_expert_cache_mib}")
    print(f"suggested_routed_cache_mib={suggested}")
    print(f"old_combined_cache_mib={mib(old_combined):.4f}")
    print(f"new_combined_cache_mib={mib(new_combined):.4f}")
    print(f"combined_headroom_mib={mib(old_combined-new_combined):.4f}")
    if new_combined > old_combined:
        raise SystemExit("rebalance unexpectedly exceeds old combined cache residency")
    print("SHARED_Q8_MEMORY_REBALANCE_PASS")


if __name__ == "__main__":
    main()
