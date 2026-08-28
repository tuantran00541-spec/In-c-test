#!/usr/bin/env python3
"""Pillow/NumPy implementation of the released Kimi-VL image preprocessing contract.

No PyTorch/torchvision is needed at runtime. Parameters are read from the packed model's
preprocessor_config.json so checkpoint overrides (not class defaults) are honored.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def preprocess_image(model_dir: str | Path, image_path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    patch = int(cfg.get("patch_size", 14))
    limit = int(cfg.get("in_token_limit", 4096))
    pad_input = bool(cfg.get("pad_input", False))
    merge = cfg.get("merge_kernel_size", [2, 2])
    # Current released preprocessor config explicitly supplies 0.5/0.5; retain official
    # class defaults only as a compatibility fallback for older snapshots.
    mean = np.asarray(cfg.get("image_mean", [0.48145466, 0.4578275, 0.40821073]), dtype=np.float32)
    std = np.asarray(cfg.get("image_std", [0.26862954, 0.26130258, 0.27577711]), dtype=np.float32)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    coarse = (w // patch) * (h // patch)
    if coarse > limit:
        scale = math.sqrt(limit / coarse)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)

    w, h = image.size
    if pad_input:
        quantum_h, quantum_w = int(merge[0]) * patch, int(merge[1]) * patch
        pad_h = (quantum_h - h % quantum_h) % quantum_h
        pad_w = (quantum_w - w % quantum_w) % quantum_w
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
    else:
        new_w = w - w % patch
        new_h = h - h % patch
        if new_w <= 0 or new_h <= 0:
            raise ValueError("image is smaller than one patch after center crop")
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        image = image.crop((left, top, left + new_w, top + new_h))

    arr = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    h, w, _ = arr.shape
    gh, gw = h // patch, w // patch
    if gh >= 512 or gw >= 512:
        raise ValueError("Exceed pos emb")
    if gh % int(merge[0]) or gw % int(merge[1]):
        raise ValueError(f"patch grid {gh}x{gw} is not divisible by merge kernel {merge}")

    chw = arr.transpose(2, 0, 1)
    patches = chw.reshape(3, gh, patch, gw, patch).transpose(1, 3, 0, 2, 4)
    patches = np.ascontiguousarray(patches.reshape(gh * gw, 3 * patch * patch), dtype=np.float32)
    return patches, (gh, gw)


def write_patches(model_dir: str | Path, image_path: str | Path, out_path: str | Path) -> tuple[int, int]:
    patches, grid = preprocess_image(model_dir, image_path)
    patches.tofile(out_path)
    return grid
