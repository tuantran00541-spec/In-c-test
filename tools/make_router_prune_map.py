#!/usr/bin/env python3
"""Choose a smaller set of unique Kimi routed experts from packed router weights.

This is a structured pruning / weight-sharing experiment. It does not edit the
router. Instead it clusters logical expert IDs in each MoE layer by cosine
similarity of their router rows and emits logical_id -> prototype_id.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
from typing import Dict, Tuple

import numpy as np

TRUNK_MAGIC = b"KVLTRNK1"
TRUNK_VERSION = 1
TRUNK_ALIGN = 4096
DTYPE_BF16 = 1
ROUTER_WEIGHT = 30

HDR = struct.Struct("<8sIIIIQQ")
REC = struct.Struct("<IIII4IQQQ")


def bf16_to_f32(raw: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u << np.uint32(16)).view(np.float32).reshape(shape)


def read_router_rows(bin_path: pathlib.Path, idx_path: pathlib.Path) -> Dict[int, np.ndarray]:
    raw = idx_path.read_bytes()
    if len(raw) < HDR.size:
        raise SystemExit("bad trunk.idx: short header")
    magic, version, align, n_records, _reserved, records_offset, data_bytes = HDR.unpack_from(raw, 0)
    if magic != TRUNK_MAGIC or version != TRUNK_VERSION or align != TRUNK_ALIGN:
        raise SystemExit("bad trunk.idx header")
    if len(raw) < records_offset + n_records * REC.size:
        raise SystemExit("bad trunk.idx: truncated records")
    if bin_path.stat().st_size < data_bytes:
        raise SystemExit("trunk.bin shorter than index declares")

    routers: Dict[int, np.ndarray] = {}
    with bin_path.open("rb") as f:
        for i in range(n_records):
            off = records_offset + i * REC.size
            values = REC.unpack_from(raw, off)
            layer, kind, dtype, ndim = values[:4]
            dims = tuple(int(x) for x in values[4:8])
            file_offset, _read_bytes, payload_bytes = values[8:11]
            if kind != ROUTER_WEIGHT:
                continue
            if dtype != DTYPE_BF16 or ndim != 2:
                raise SystemExit(f"layer {layer}: router must be BF16 rank-2")
            rows, cols = dims[0], dims[1]
            expected = rows * cols * 2
            if payload_bytes != expected:
                raise SystemExit(f"layer {layer}: router payload {payload_bytes} != {expected}")
            f.seek(file_offset)
            blob = f.read(expected)
            if len(blob) != expected:
                raise SystemExit(f"layer {layer}: short router read")
            routers[int(layer)] = bf16_to_f32(blob, (rows, cols))
    if not routers:
        raise SystemExit("no router-weight records found")
    return routers


def choose_prototypes(rows: np.ndarray, keep: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Deterministic farthest-first medoids under cosine distance."""
    if rows.ndim != 2:
        raise ValueError(rows.shape)
    n = rows.shape[0]
    if keep < 1 or keep > n:
        raise ValueError(f"keep={keep}, experts={n}")

    norms = np.linalg.norm(rows, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    unit = rows / safe[:, None]

    first = int(np.argmax(norms))
    chosen = [first]
    best_sim = unit @ unit[first]
    while len(chosen) < keep:
        score = best_sim.copy()
        score[np.asarray(chosen, dtype=np.int64)] = np.inf
        nxt = int(np.argmin(score))
        chosen.append(nxt)
        best_sim = np.maximum(best_sim, unit @ unit[nxt])

    chosen_arr = np.asarray(chosen, dtype=np.int64)
    sims = unit @ unit[chosen_arr].T
    nearest_idx = np.argmax(sims, axis=1)
    mapping = chosen_arr[nearest_idx]
    for p in chosen:
        mapping[p] = p
    assigned_sim = sims[np.arange(n), nearest_idx]
    return sorted(chosen), mapping.astype(np.int64), assigned_sim.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trunk_bin", type=pathlib.Path)
    ap.add_argument("trunk_idx", type=pathlib.Path)
    ap.add_argument("output_json", type=pathlib.Path)
    ap.add_argument("--keep", type=int, required=True, help="unique experts kept per MoE layer")
    args = ap.parse_args()

    routers = read_router_rows(args.trunk_bin, args.trunk_idx)
    out_layers = {}
    total_logical = total_unique = 0
    mean_parts = []
    min_sim = 1.0
    for layer in sorted(routers):
        rows = routers[layer]
        chosen, mapping, assigned_sim = choose_prototypes(rows, args.keep)
        total_logical += rows.shape[0]
        total_unique += len(chosen)
        mean_parts.append(assigned_sim)
        min_sim = min(min_sim, float(np.min(assigned_sim)))
        out_layers[str(layer)] = {
            "experts": int(rows.shape[0]),
            "prototypes": chosen,
            "map": {str(i): int(mapping[i]) for i in range(rows.shape[0])},
            "mean_assigned_cosine": float(np.mean(assigned_sim)),
            "min_assigned_cosine": float(np.min(assigned_sim)),
        }

    all_sims = np.concatenate(mean_parts)
    obj = {
        "version": 1,
        "method": "router-cosine-farthest-first",
        "keep_per_layer": args.keep,
        "logical_expert_records": total_logical,
        "unique_expert_payloads": total_unique,
        "mean_assigned_cosine": float(np.mean(all_sims)),
        "min_assigned_cosine": float(min_sim),
        "layers": out_layers,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(
        f"PRUNE_MAP_PASS keep={args.keep} layers={len(out_layers)} "
        f"logical={total_logical} unique={total_unique} "
        f"mean_cos={obj['mean_assigned_cosine']:.6f} min_cos={obj['min_assigned_cosine']:.6f}"
    )


if __name__ == "__main__":
    main()
