#!/usr/bin/env python3
"""Build bounded per-expert activation reservoirs from routed token traces.

The intended upstream trace format is JSON Lines with one routed token per
record.  Required fields:

  layer: int
  expert: int
  x: list[float]        # expert-input activation, hidden-size elements

Optional fields such as source/split/kind/prompt_id are preserved only as
counts in the emitted manifest.  Reservoir sampling is deterministic given a
seed and prevents high-frequency experts from monopolizing calibration memory.

Outputs one NPZ per observed expert containing only `x` plus metadata.  Expert
weights are intentionally *not* copied here; the sensitivity runner can attach
pinned BF16 gate/up/down tensors later.  No model weights are written by this
tool.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def build_reservoir(trace: Path, out_dir: Path, capacity: int, hidden: int, seed: int) -> dict:
    if capacity <= 0 or hidden <= 0:
        raise ValueError("capacity and hidden must be positive")
    rng = random.Random(seed)
    seen: Counter[tuple[int, int]] = Counter()
    kept: dict[tuple[int, int], list[np.ndarray]] = {}
    source_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    rows = 0

    with trace.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                layer = int(rec["layer"])
                expert = int(rec["expert"])
                x = np.asarray(rec["x"], dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{trace}:{line_no}: invalid routed record") from exc
            if x.shape != (hidden,):
                raise ValueError(f"{trace}:{line_no}: x shape={x.shape}, expected {(hidden,)}")
            key = (layer, expert)
            seen[key] += 1
            rows += 1
            if "source" in rec:
                source_counts[str(rec["source"])] += 1
            if "kind" in rec:
                kind_counts[str(rec["kind"])] += 1

            bucket = kept.setdefault(key, [])
            n = seen[key]
            if len(bucket) < capacity:
                bucket.append(x.copy())
            else:
                j = rng.randrange(n)
                if j < capacity:
                    bucket[j] = x.copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    experts = []
    for (layer, expert), bucket in sorted(kept.items()):
        arr = np.stack(bucket, axis=0).astype(np.float32, copy=False)
        name = f"layer-{layer:02d}-expert-{expert:02d}.npz"
        np.savez(
            out_dir / name,
            x=arr,
            meta_layer=np.array(layer, dtype=np.int32),
            meta_expert=np.array(expert, dtype=np.int32),
            meta_seen=np.array(seen[(layer, expert)], dtype=np.int64),
            meta_kept=np.array(arr.shape[0], dtype=np.int32),
            meta_seed=np.array(seed, dtype=np.int64),
        )
        experts.append(
            {
                "layer": layer,
                "expert": expert,
                "seen": int(seen[(layer, expert)]),
                "kept": int(arr.shape[0]),
                "file": name,
            }
        )

    manifest = {
        "schema": "kimi-expert-activation-reservoir-v1",
        "trace": str(trace),
        "hidden": hidden,
        "capacity_per_expert": capacity,
        "seed": seed,
        "rows": rows,
        "expert_count": len(experts),
        "source_counts": dict(sorted(source_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "experts": experts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_jsonl", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--capacity", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    manifest = build_reservoir(args.trace_jsonl, args.out_dir, args.capacity, args.hidden, args.seed)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
