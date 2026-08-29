#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pathlib
import subprocess
import tempfile
import numpy as np


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxabs = np.max(np.abs(w), axis=1)
    scales = np.where(maxabs > 0, maxabs / 127.0, 1.0).astype(np.float32)
    q = np.rint(w / scales[:, None]).clip(-127, 127).astype(np.int8)
    return scales, q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-dir', type=pathlib.Path, default=pathlib.Path('build'))
    args = ap.parse_args()
    exe = args.build_dir / ('kvl_q8_probe.exe' if __import__('os').name == 'nt' else 'kvl_q8_probe')
    if not exe.exists():
        release = args.build_dir / 'Release' / 'kvl_q8_probe.exe'
        if release.exists(): exe = release
    if not exe.exists(): raise SystemExit(f'missing probe: {exe}')

    rng = np.random.default_rng(12345)
    out_dim, in_dim = 257, 513
    w = rng.normal(0, 0.045, size=(out_dim, in_dim)).astype(np.float32)
    # Add a few outliers so the test exercises realistic row scaling.
    w[0, 0] = 1.75; w[17, 211] = -1.25; w[128, 300] = 0.95
    x = rng.normal(0, 0.7, size=(in_dim,)).astype(np.float32)
    scales, q = quantize_rows(w)
    deq = q.astype(np.float32) * scales[:, None]
    ref_q8 = deq @ x
    ref_f32 = w @ x

    with tempfile.TemporaryDirectory(prefix='kvl-q8-') as td:
        td = pathlib.Path(td)
        blob = td / 'w.q8'; xp = td / 'x.f32'; yp = td / 'y.f32'
        blob.write_bytes(scales.astype('<f4').tobytes() + q.tobytes(order='C'))
        xp.write_bytes(x.astype('<f4').tobytes())
        subprocess.run([str(exe), str(blob), str(xp), str(in_dim), str(out_dim), str(yp)], check=True)
        got = np.fromfile(yp, dtype='<f4')

    kernel_max = float(np.max(np.abs(got - ref_q8)))
    quant_rmse = float(np.sqrt(np.mean((ref_q8 - ref_f32) ** 2)))
    signal_rms = float(np.sqrt(np.mean(ref_f32 ** 2)))
    rel_rmse = quant_rmse / max(signal_rms, 1e-12)
    print(f'Q8_KERNEL_MAXABS={kernel_max:.9g}')
    print(f'Q8_QUANT_RMSE={quant_rmse:.9g}')
    print(f'Q8_REL_RMSE={rel_rmse:.9g}')
    print(f'Q8_BYTES={scales.nbytes + q.nbytes} BF16_BYTES={w.size * 2}')
    assert kernel_max < 2e-4, kernel_max
    assert rel_rmse < 0.02, rel_rmse
    print('PASS: Q8 row-wise kernel and quantization sanity')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
