#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from pack_experts_q5 import pack_signed_q5  # noqa: E402

GROUP = 128


def quantize_g128(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = w.shape
    groups = (cols + GROUP - 1) // GROUP
    scales = np.empty((rows, groups), dtype=np.float32)
    q = np.empty((rows, cols), dtype=np.int8)
    for g in range(groups):
        a = g * GROUP
        b = min(cols, a + GROUP)
        block = w[:, a:b]
        maxabs = np.max(np.abs(block), axis=1)
        s = np.where(maxabs > 0, maxabs / 15.0, 1.0).astype(np.float32)
        scales[:, g] = s
        q[:, a:b] = np.rint(block / s[:, None]).clip(-15, 15).astype(np.int8)
    return scales, q


def dequant(scales: np.ndarray, q: np.ndarray) -> np.ndarray:
    rows, cols = q.shape
    out = np.empty((rows, cols), dtype=np.float32)
    for g in range(scales.shape[1]):
        a = g * GROUP
        b = min(cols, a + GROUP)
        out[:, a:b] = q[:, a:b].astype(np.float32) * scales[:, g:g+1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path("build"))
    args = ap.parse_args()
    exe = args.build_dir / ("kvl_q5_probe.exe" if os.name == "nt" else "kvl_q5_probe")
    if not exe.exists():
        release = args.build_dir / "Release" / "kvl_q5_probe.exe"
        if release.exists():
            exe = release
    if not exe.exists():
        raise SystemExit(f"missing probe: {exe}")

    rng = np.random.default_rng(54321)
    out_dim, in_dim = 257, 513  # exercises non-byte-aligned row starts too
    w = rng.normal(0, 0.045, size=(out_dim, in_dim)).astype(np.float32)
    w[0, 0] = 1.75
    w[17, 211] = -1.25
    w[128, 300] = 0.95
    x = rng.normal(0, 0.7, size=(in_dim,)).astype(np.float32)
    scales, q = quantize_g128(w)
    dq = dequant(scales, q)
    ref_q5 = dq @ x
    ref_f32 = w @ x
    packed = pack_signed_q5(q)
    blob_bytes = scales.astype("<f4").tobytes(order="C") + packed

    expected_bytes = scales.nbytes + (q.size * 5 + 7) // 8
    assert len(blob_bytes) == expected_bytes

    with tempfile.TemporaryDirectory(prefix="kvl-q5-") as td:
        td = pathlib.Path(td)
        blob = td / "w.q5"
        xp = td / "x.f32"
        yp = td / "y.f32"
        blob.write_bytes(blob_bytes)
        xp.write_bytes(x.astype("<f4").tobytes())
        subprocess.run([str(exe), str(blob), str(xp), str(in_dim), str(out_dim), str(yp)], check=True)
        got = np.fromfile(yp, dtype="<f4")

    kernel_max = float(np.max(np.abs(got - ref_q5)))
    quant_rmse = float(np.sqrt(np.mean((ref_q5 - ref_f32) ** 2)))
    signal_rms = float(np.sqrt(np.mean(ref_f32 ** 2)))
    rel_rmse = quant_rmse / max(signal_rms, 1e-12)
    print(f"Q5_KERNEL_MAXABS={kernel_max:.9g}")
    print(f"Q5_QUANT_RMSE={quant_rmse:.9g}")
    print(f"Q5_REL_RMSE={rel_rmse:.9g}")
    print(f"Q5_BYTES={len(blob_bytes)} BF16_BYTES={w.size * 2}")
    assert kernel_max < 5e-4, kernel_max
    assert rel_rmse < 0.15, rel_rmse
    print("PASS: Q5 group128 kernel, packing, and quantization sanity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
