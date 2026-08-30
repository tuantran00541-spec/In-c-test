#!/usr/bin/env python3
"""Measure one pinned Kimi-VL BF16 routed expert at Q8/Q6/Q5/Q4.

The tool reads BF16 tensors directly from local safetensors shards using the
checkpoint index, avoiding PyTorch and avoiding a second copy of the model.  It
combines those source weights with a bounded activation-reservoir NPZ produced
by `kimi_expert_reservoir.py`, then calls the offline quantization simulator.

This is measurement-only.  It does not write quantized model weights.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kimi_compression_lab import evaluate_candidate  # noqa: E402

PINNED_REVISION = "398eede0903cd983a2bfa0cc634e9ac1d843f375"
PREFIX = "language_model.model.layers.{layer}.mlp.experts.{expert}.{part}.weight"


def _read_safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: short safetensors header")
        n = struct.unpack("<Q", raw)[0]
        hraw = f.read(n)
        if len(hraw) != n:
            raise ValueError(f"{path}: short safetensors JSON header")
    return 8 + n, json.loads(hraw)


def _bf16_bytes_to_f32(raw: bytes, shape: list[int]) -> np.ndarray:
    expected = int(np.prod(shape, dtype=np.int64))
    u16 = np.frombuffer(raw, dtype="<u2")
    if u16.size != expected:
        raise ValueError(f"BF16 payload elements={u16.size}, expected={expected}")
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32).reshape(tuple(int(x) for x in shape))


def load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> tuple[np.ndarray, str]:
    try:
        shard_name = weight_map[name]
    except KeyError as exc:
        raise KeyError(f"checkpoint index does not contain {name}") from exc
    shard = model_dir / shard_name
    if not shard.is_file():
        raise FileNotFoundError(f"required source shard missing: {shard}")
    base, header = _read_safetensors_header(shard)
    try:
        meta = header[name]
    except KeyError as exc:
        raise KeyError(f"{shard}: header does not contain {name}") from exc
    if meta.get("dtype") != "BF16":
        raise ValueError(f"{name}: expected BF16, got {meta.get('dtype')}")
    a, b = (int(x) for x in meta["data_offsets"])
    with shard.open("rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    if len(raw) != b - a:
        raise IOError(f"{name}: short tensor read")
    return _bf16_bytes_to_f32(raw, meta["shape"]), shard_name


def measure(
    model_dir: Path,
    reservoir_npz: Path,
    bits: list[int],
    group_size: int,
    revision: str,
) -> dict:
    if revision != PINNED_REVISION:
        raise ValueError(f"refusing unpinned revision {revision}; expected {PINNED_REVISION}")
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    with np.load(reservoir_npz, allow_pickle=False) as data:
        if "x" not in data or "meta_layer" not in data or "meta_expert" not in data:
            raise ValueError("reservoir must contain x, meta_layer, and meta_expert")
        x = np.asarray(data["x"], dtype=np.float32)
        layer = int(data["meta_layer"].item())
        expert = int(data["meta_expert"].item())
        seen = int(data["meta_seen"].item()) if "meta_seen" in data else int(x.shape[0])
        kept = int(data["meta_kept"].item()) if "meta_kept" in data else int(x.shape[0])

    weights = {}
    shards = {}
    for short, part in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
        name = PREFIX.format(layer=layer, expert=expert, part=part)
        weights[short], shards[short] = load_tensor(model_dir, weight_map, name)

    gate = weights["gate"]
    up = weights["up"]
    down = weights["down"]
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {gate.shape} vs {up.shape}")
    inter, hidden = gate.shape
    if x.ndim != 2 or x.shape[1] != hidden:
        raise ValueError(f"reservoir x shape={x.shape}, expected [tokens,{hidden}]")
    if down.shape != (hidden, inter):
        raise ValueError(f"down shape={down.shape}, expected {(hidden, inter)}")

    candidates = [
        evaluate_candidate(x, gate, up, down, bits=b, group_size=group_size)
        for b in bits
    ]
    from dataclasses import asdict

    return {
        "schema": "kimi-expert-quant-sensitivity-v1",
        "source": str(reservoir_npz),
        "source_revision": revision,
        "tokens": int(x.shape[0]),
        "hidden": hidden,
        "intermediate": inter,
        "metadata": {
            "layer": layer,
            "expert": expert,
            "seen": seen,
            "kept": kept,
            "source_shards": shards,
        },
        "projection_only": True,
        "note": "BF16 source was measured directly; projected bytes are not a native packed format or physical measurement.",
        "candidates": [asdict(c) for c in candidates],
    }


def _parse_bits(text: str) -> list[int]:
    out = []
    for item in text.split(","):
        b = int(item.strip())
        if b < 2 or b > 8:
            raise argparse.ArgumentTypeError("bits must be in [2,8]")
        if b not in out:
            out.append(b)
    if not out:
        raise argparse.ArgumentTypeError("at least one bit width is required")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("reservoir_npz", type=Path)
    ap.add_argument("--revision", default=PINNED_REVISION)
    ap.add_argument("--bits", type=_parse_bits, default=_parse_bits("8,6,5,4"))
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = measure(args.model_dir, args.reservoir_npz, args.bits, args.group_size, args.revision)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
