#!/usr/bin/env python3
"""Compare the V9 lightweight image frontend with Moonshot's released image processor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kimi_image import preprocess_image  # noqa: E402

REPO = "moonshotai/Kimi-VL-A3B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runtime_dir")
    ap.add_argument("image")
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    ours, grid = preprocess_image(args.runtime_dir, args.image)
    gh, gw = grid
    ref = AutoImageProcessor.from_pretrained(REPO, trust_remote_code=True)
    expected = ref.preprocess(image, return_tensors="pt")
    ref_patches = expected["pixel_values"].cpu().numpy().astype(np.float32, copy=False)
    ref_grid = expected["image_grid_hws"].cpu().numpy().reshape(-1, 2)[0]

    assert (gh, gw) == (int(ref_grid[0]), int(ref_grid[1])), ((gh, gw), ref_grid.tolist())
    assert ours.shape == ref_patches.reshape(ref_patches.shape[0], -1).shape, (ours.shape, ref_patches.shape)
    ref_flat = np.ascontiguousarray(ref_patches.reshape(ref_patches.shape[0], -1))
    d = np.abs(ours - ref_flat)
    rms = float(np.sqrt(np.mean((ours - ref_flat) ** 2)))
    print(f"processor grid={gh}x{gw} max_abs={float(d.max()):.9g} rms={rms:.9g}")
    assert np.isfinite(ours).all()
    assert float(d.max()) < 2e-6


if __name__ == "__main__":
    main()
