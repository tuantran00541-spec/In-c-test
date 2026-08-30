#!/usr/bin/env python3
"""Offline quantization-sensitivity laboratory for Kimi-VL routed experts.

This tool intentionally does *not* change the native runtime format.  It takes
one or more BF16 expert matrices plus calibration activations, simulates
symmetric group-wise low-bit weight quantization, dequantizes back to float,
and measures reconstruction error.  The first phase is meant to identify
which experts/matrices tolerate Q8/Q6/Q5/Q4 before any Q4 C kernel exists.

Input NPZ format
----------------
Required arrays:
  x                 [tokens, hidden]
  gate              [intermediate, hidden]
  up                [intermediate, hidden]
  down              [hidden, intermediate]

Optional metadata scalars/arrays are copied into the JSON result when their
keys start with ``meta_``.

The simulated MLP is SiLU(gate(x)) * up(x), then down-projected.  Metrics are
computed against the BF16-source-as-float reference for the same activation
batch.  Quantization is per output-row, group-wise along the input dimension.

No claimed storage sizes from this simulator are physical measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class MatrixMetrics:
    bits: int
    group_size: int
    weight_mse: float
    weight_max_abs: float
    scale_count: int
    payload_bits: int
    scale_bits: int


@dataclass
class CandidateMetrics:
    bits: int
    group_size: int
    output_mse: float
    output_rmse: float
    output_max_abs: float
    output_cosine: float
    relative_l2: float
    gate: MatrixMetrics
    up: MatrixMetrics
    down: MatrixMetrics
    projected_payload_bytes: int
    projected_scale_bytes_f16: int
    projected_total_bytes_f16_scales: int


def _check_matrix(name: str, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={a.shape}")
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError(f"{name} must be floating point, got dtype={a.dtype}")
    return a.astype(np.float32, copy=False)


def _silu(x: np.ndarray) -> np.ndarray:
    # Stable enough for calibration tensors and keeps the dependency set tiny.
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def mlp_forward(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    g = x @ gate.T
    u = x @ up.T
    h = _silu(g) * u
    return h @ down.T


def quantize_dequant_symmetric(w: np.ndarray, bits: int, group_size: int) -> tuple[np.ndarray, MatrixMetrics]:
    """Simulate signed symmetric RTN quantization and immediate dequantization.

    Each output row is split into contiguous input groups.  A group uses
    max-absolute scaling.  This is a deliberately simple baseline; GPTQ/AWQ
    variants should be judged against it later rather than conflated with it.
    """
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    w = _check_matrix("weight", w)
    rows, cols = w.shape
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    out = np.empty_like(w, dtype=np.float32)
    groups_per_row = (cols + group_size - 1) // group_size
    scale_count = rows * groups_per_row

    for r in range(rows):
        row = w[r]
        dst = out[r]
        for start in range(0, cols, group_size):
            end = min(start + group_size, cols)
            block = row[start:end]
            amax = float(np.max(np.abs(block))) if block.size else 0.0
            scale = amax / qmax if amax > 0.0 else 1.0
            q = np.rint(block / scale)
            q = np.clip(q, qmin, qmax)
            dst[start:end] = q * scale

    err = out - w
    metrics = MatrixMetrics(
        bits=bits,
        group_size=group_size,
        weight_mse=float(np.mean(np.square(err), dtype=np.float64)),
        weight_max_abs=float(np.max(np.abs(err))) if err.size else 0.0,
        scale_count=scale_count,
        payload_bits=int(w.size * bits),
        scale_bits=int(scale_count * 16),  # projection with FP16 scales only
    )
    return out, metrics


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64, copy=False)
    bv = b.reshape(-1).astype(np.float64, copy=False)
    denom = math.sqrt(float(av @ av) * float(bv @ bv))
    if denom == 0.0:
        return 1.0 if np.array_equal(av, bv) else 0.0
    return float((av @ bv) / denom)


def evaluate_candidate(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    bits: int,
    group_size: int,
    reference: np.ndarray | None = None,
) -> CandidateMetrics:
    x = _check_matrix("x", x)
    gate = _check_matrix("gate", gate)
    up = _check_matrix("up", up)
    down = _check_matrix("down", down)
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {gate.shape} vs {up.shape}")
    inter, hidden = gate.shape
    if x.shape[1] != hidden:
        raise ValueError(f"x hidden={x.shape[1]} but expert hidden={hidden}")
    if down.shape != (hidden, inter):
        raise ValueError(f"down must have shape {(hidden, inter)}, got {down.shape}")

    if reference is None:
        reference = mlp_forward(x, gate, up, down)
    else:
        reference = _check_matrix("reference", reference)

    qgate, mg = quantize_dequant_symmetric(gate, bits, group_size)
    qup, mu = quantize_dequant_symmetric(up, bits, group_size)
    qdown, md = quantize_dequant_symmetric(down, bits, group_size)
    got = mlp_forward(x, qgate, qup, qdown)
    err = got - reference
    sq = np.square(err)
    ref_norm = float(np.linalg.norm(reference.astype(np.float64, copy=False)))
    err_norm = float(np.linalg.norm(err.astype(np.float64, copy=False)))
    payload_bits = mg.payload_bits + mu.payload_bits + md.payload_bits
    scale_bits = mg.scale_bits + mu.scale_bits + md.scale_bits
    return CandidateMetrics(
        bits=bits,
        group_size=group_size,
        output_mse=float(np.mean(sq, dtype=np.float64)),
        output_rmse=float(math.sqrt(float(np.mean(sq, dtype=np.float64)))),
        output_max_abs=float(np.max(np.abs(err))) if err.size else 0.0,
        output_cosine=_cosine(reference, got),
        relative_l2=(err_norm / ref_norm) if ref_norm > 0.0 else err_norm,
        gate=mg,
        up=mu,
        down=md,
        projected_payload_bytes=(payload_bits + 7) // 8,
        projected_scale_bytes_f16=(scale_bits + 7) // 8,
        projected_total_bytes_f16_scales=(payload_bits + scale_bits + 7) // 8,
    )


def _parse_bits(value: str) -> list[int]:
    bits = []
    for item in value.split(","):
        b = int(item.strip())
        if b not in bits:
            bits.append(b)
    if not bits:
        raise argparse.ArgumentTypeError("at least one bit width is required")
    if any(b < 2 or b > 8 for b in bits):
        raise argparse.ArgumentTypeError("bit widths must be in [2, 8]")
    return bits


def run_npz(path: Path, bits: Iterable[int], group_size: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = ("x", "gate", "up", "down")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        x = _check_matrix("x", data["x"])
        gate = _check_matrix("gate", data["gate"])
        up = _check_matrix("up", data["up"])
        down = _check_matrix("down", data["down"])
        metadata = {}
        for key in data.files:
            if key.startswith("meta_"):
                v = data[key]
                metadata[key[5:]] = v.item() if v.ndim == 0 else v.tolist()

    reference = mlp_forward(x, gate, up, down)
    candidates = [
        evaluate_candidate(x, gate, up, down, b, group_size, reference=reference)
        for b in bits
    ]
    return {
        "schema": "kimi-expert-quant-sensitivity-v1",
        "source": str(path),
        "tokens": int(x.shape[0]),
        "hidden": int(x.shape[1]),
        "intermediate": int(gate.shape[0]),
        "metadata": metadata,
        "projection_only": True,
        "note": "Projected bytes model packed payload plus FP16 group scales; no native low-bit format was written.",
        "candidates": [asdict(c) for c in candidates],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_npz", type=Path)
    ap.add_argument("--bits", type=_parse_bits, default=_parse_bits("8,6,5,4"))
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = run_npz(args.input_npz, args.bits, args.group_size)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
