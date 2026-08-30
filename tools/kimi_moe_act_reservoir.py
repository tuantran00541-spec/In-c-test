#!/usr/bin/env python3
"""Build balanced per-expert reservoirs from KVLACT01 binary traces.

Usage example:

  python tools/kimi_moe_act_reservoir.py out-dir \
      --trace text=text.act --trace media=vl.act --capacity-per-kind 8

Each KVLACT01 record stores one MoE input vector for a token/layer plus the
selected expert ids and routing weights.  This builder performs independent
reservoir sampling for every (layer, expert, kind), preventing a high-frequency
text route from crowding all media samples out of calibration.

One NPZ is emitted per observed expert with:
  x               [samples, hidden] float32
  routing_weight  [samples] float32
  event           [samples] uint64
  kind             [samples] uint16 index into meta_kind_names

No model weights are copied.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

MAGIC = b"KVLACT01"
HEADER = struct.Struct("<8sIIII")
VERSION = 1
ENDIAN_MARKER = 0x01020304


@dataclass
class TraceHeader:
    hidden: int
    top_k: int


@dataclass
class TraceRecord:
    event: int
    layer: int
    ids: np.ndarray
    weights: np.ndarray
    x: np.ndarray


def read_header(f, path: Path) -> TraceHeader:
    raw = f.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError(f"{path}: short KVLACT01 header")
    magic, version, hidden, top_k, endian = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad activation-trace magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"{path}: unsupported activation-trace version {version}")
    if endian != ENDIAN_MARKER:
        raise ValueError(f"{path}: unsupported endian marker 0x{endian:08x}")
    if hidden <= 0 or hidden > 65536 or top_k <= 0 or top_k > 64:
        raise ValueError(f"{path}: invalid hidden/top_k {hidden}/{top_k}")
    return TraceHeader(int(hidden), int(top_k))


def iter_trace(path: Path) -> tuple[TraceHeader, Iterator[TraceRecord]]:
    f = path.open("rb")
    try:
        header = read_header(f, path)
    except Exception:
        f.close()
        raise
    fixed = struct.Struct("<Qi")
    ids_struct = struct.Struct("<" + "i" * header.top_k)
    weights_struct = struct.Struct("<" + "f" * header.top_k)
    x_bytes = header.hidden * 4

    def _gen() -> Iterator[TraceRecord]:
        try:
            while True:
                raw = f.read(fixed.size)
                if not raw:
                    break
                if len(raw) != fixed.size:
                    raise ValueError(f"{path}: truncated activation record prefix")
                event, layer = fixed.unpack(raw)
                iraw = f.read(ids_struct.size)
                wraw = f.read(weights_struct.size)
                xraw = f.read(x_bytes)
                if len(iraw) != ids_struct.size or len(wraw) != weights_struct.size or len(xraw) != x_bytes:
                    raise ValueError(f"{path}: truncated activation record payload")
                ids = np.asarray(ids_struct.unpack(iraw), dtype=np.int32)
                weights = np.asarray(weights_struct.unpack(wraw), dtype=np.float32)
                x = np.frombuffer(xraw, dtype="<f4").astype(np.float32, copy=True)
                yield TraceRecord(int(event), int(layer), ids, weights, x)
        finally:
            f.close()

    return header, _gen()


def parse_trace_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--trace must be KIND=PATH")
    kind, raw = value.split("=", 1)
    kind = kind.strip()
    if not kind:
        raise argparse.ArgumentTypeError("trace kind cannot be empty")
    path = Path(raw)
    return kind, path


def build_balanced_reservoir(
    traces: list[tuple[str, Path]],
    out_dir: Path,
    capacity_per_kind: int,
    seed: int,
) -> dict:
    if capacity_per_kind <= 0:
        raise ValueError("capacity_per_kind must be positive")
    if not traces:
        raise ValueError("at least one trace is required")
    kinds = []
    for kind, _ in traces:
        if kind not in kinds:
            kinds.append(kind)
    kind_to_id = {k: i for i, k in enumerate(kinds)}
    rng = random.Random(seed)
    hidden = None
    top_k = None
    seen: Counter[tuple[int, int, str]] = Counter()
    trace_records: Counter[str] = Counter()
    kept: dict[tuple[int, int, str], list[tuple[np.ndarray, float, int]]] = defaultdict(list)

    for kind, path in traces:
        header, records = iter_trace(path)
        if hidden is None:
            hidden, top_k = header.hidden, header.top_k
        elif (header.hidden, header.top_k) != (hidden, top_k):
            raise ValueError(
                f"{path}: hidden/top_k {header.hidden}/{header.top_k} do not match {hidden}/{top_k}"
            )
        for rec in records:
            trace_records[kind] += 1
            if len(set(int(x) for x in rec.ids)) != len(rec.ids):
                raise ValueError(f"{path}: duplicate selected expert id at event {rec.event}")
            for j, expert_raw in enumerate(rec.ids):
                expert = int(expert_raw)
                if expert < 0:
                    raise ValueError(f"{path}: negative expert id at event {rec.event}")
                key = (rec.layer, expert, kind)
                seen[key] += 1
                bucket = kept[key]
                item = (rec.x.copy(), float(rec.weights[j]), int(rec.event))
                n = seen[key]
                if len(bucket) < capacity_per_kind:
                    bucket.append(item)
                else:
                    r = rng.randrange(n)
                    if r < capacity_per_kind:
                        bucket[r] = item

    assert hidden is not None and top_k is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    experts = sorted(set((layer, expert) for layer, expert, _ in kept))
    expert_rows = []
    for layer, expert in experts:
        xs = []
        ws = []
        events = []
        kind_ids = []
        per_kind = {}
        for kind in kinds:
            bucket = kept.get((layer, expert, kind), [])
            per_kind[kind] = {
                "seen": int(seen[(layer, expert, kind)]),
                "kept": len(bucket),
            }
            for x, w, event in bucket:
                xs.append(x)
                ws.append(w)
                events.append(event)
                kind_ids.append(kind_to_id[kind])
        if not xs:
            continue
        x_arr = np.stack(xs).astype(np.float32, copy=False)
        name = f"layer-{layer:02d}-expert-{expert:02d}.npz"
        np.savez(
            out_dir / name,
            x=x_arr,
            routing_weight=np.asarray(ws, dtype=np.float32),
            event=np.asarray(events, dtype=np.uint64),
            kind=np.asarray(kind_ids, dtype=np.uint16),
            meta_kind_names=np.asarray(kinds),
            meta_layer=np.asarray(layer, dtype=np.int32),
            meta_expert=np.asarray(expert, dtype=np.int32),
            meta_kept=np.asarray(len(xs), dtype=np.int32),
            meta_seed=np.asarray(seed, dtype=np.int64),
        )
        expert_rows.append(
            {
                "layer": layer,
                "expert": expert,
                "file": name,
                "kept": len(xs),
                "per_kind": per_kind,
            }
        )

    manifest = {
        "schema": "kimi-expert-activation-reservoir-v2",
        "trace_format": "KVLACT01",
        "traces": [{"kind": k, "path": str(p)} for k, p in traces],
        "hidden": hidden,
        "top_k": top_k,
        "capacity_per_kind": capacity_per_kind,
        "seed": seed,
        "kind_names": kinds,
        "trace_records": dict(trace_records),
        "expert_count": len(expert_rows),
        "experts": expert_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--trace", action="append", type=parse_trace_arg, required=True, help="KIND=PATH; repeatable")
    ap.add_argument("--capacity-per-kind", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    manifest = build_balanced_reservoir(args.trace, args.out_dir, args.capacity_per_kind, args.seed)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
