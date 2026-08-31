#!/usr/bin/env python3
"""Build a tiny real-weight Q8_0 oracle fixture without downloading the full GGUF.

The offsets below are pinned to the exact GGUF revision that passed the physical-layout
probe. Each requested range is one expert-0 matrix from decoder layer 1 (~2.92 MiB).
The numerical oracle uses llama.cpp's pinned gguf-py Q8_0 dequantizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
from gguf import GGMLQuantizationType, dequantize

REPO = "mradermacher/Kimi-VL-A3B-Instruct-GGUF"
REVISION = "d645665be0a3028dca3ef3d08dddb51ab23ecf31"
FILENAME = "Kimi-VL-A3B-Instruct.Q8_0.gguf"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}?download=true"

H = 2048
I = 1408
Q8_BLOCK = 32
Q8_BYTES = 34
SLICE_BYTES = 3_063_808
# Exact absolute offsets from run 33391055394 / artifact 9757535280.
MATRICES = {
    "down": {"offset": 822_783_264, "in": I, "out": H},
    "gate": {"offset": 1_024_994_592, "in": H, "out": I},
    "up":   {"offset": 1_227_738_400, "in": H, "out": I},
}


def fetch_range(start: int, size: int) -> bytes:
    end = start + size - 1
    with requests.get(
        URL,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=True,
        allow_redirects=True,
        timeout=(30, 180),
    ) as r:
        if r.status_code != 206:
            raise RuntimeError(
                f"server did not honor Range bytes={start}-{end}: HTTP {r.status_code}; "
                "aborting rather than accidentally downloading the full 17 GB file"
            )
        data = r.raw.read(size + 1)
        if len(data) != size:
            raise RuntimeError(f"short/long range read: expected {size}, got {len(data)}")
        return data


def oracle(raw: bytes, in_dim: int, out_dim: int, x: np.ndarray) -> np.ndarray:
    row_bytes = (in_dim // Q8_BLOCK) * Q8_BYTES
    if in_dim % Q8_BLOCK or row_bytes * out_dim != len(raw):
        raise ValueError((in_dim, out_dim, row_bytes, len(raw)))
    qbytes = np.frombuffer(raw, dtype=np.uint8).reshape(out_dim, row_bytes)
    w = dequantize(qbytes, GGMLQuantizationType.Q8_0)
    if w.shape != (out_dim, in_dim):
        raise RuntimeError(f"unexpected dequant shape {w.shape}, wanted {(out_dim, in_dim)}")
    y = w.astype(np.float64) @ x.astype(np.float64)
    return y.astype("<f4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ih = np.arange(H, dtype=np.float64)
    ii = np.arange(I, dtype=np.float64)
    xh = (np.sin(ih * 0.013) + 0.3 * np.cos(ih * 0.031) + 0.05 * np.sin(ih * 0.071)).astype("<f4")
    xi = (0.7 * np.sin(ii * 0.017) - 0.2 * np.cos(ii * 0.043) + 0.1 * np.cos(ii * 0.089)).astype("<f4")
    (args.out_dir / "x_h.f32").write_bytes(xh.tobytes())
    (args.out_dir / "x_i.f32").write_bytes(xi.tobytes())

    evidence = {
        "repo": REPO,
        "revision": REVISION,
        "filename": FILENAME,
        "layer": 1,
        "expert": 0,
        "qtype": "Q8_0",
        "matrices": {},
        "oracle": "llama.cpp gguf-py dequantize(Q8_0) + numpy float64 dot",
    }

    for name in ("gate", "up", "down"):
        spec = MATRICES[name]
        raw = fetch_range(spec["offset"], SLICE_BYTES)
        blob_path = args.out_dir / f"{name}.q8"
        blob_path.write_bytes(raw)
        x = xh if spec["in"] == H else xi
        y = oracle(raw, spec["in"], spec["out"], x)
        (args.out_dir / f"{name}_ref.f32").write_bytes(y.tobytes())
        evidence["matrices"][name] = {
            **spec,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "reference_min": float(y.min()),
            "reference_max": float(y.max()),
            "reference_mean_abs": float(np.mean(np.abs(y), dtype=np.float64)),
        }
        print(
            f"KIMI_GGUF_Q8_0_FIXTURE part={name} offset={spec['offset']} bytes={len(raw)} "
            f"sha256={evidence['matrices'][name]['sha256']}"
        )

    (args.out_dir / "fixture.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
