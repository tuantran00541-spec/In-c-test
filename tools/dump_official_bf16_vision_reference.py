#!/usr/bin/env python3
"""Run only the released Kimi-VL vision tower + projector with official BF16 semantics.

This deliberately does not instantiate the 16B text model. It imports the checkpoint's own
remote-code classes, loads only vision_tower.* and multi_modal_projector.* tensors, casts the
preprocessed patches to the vision tower dtype exactly like KimiVLForConditionalGeneration,
and writes the projected media embeddings seen by the text model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

REPO = "moonshotai/Kimi-VL-A3B-Instruct"
PATCH = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("patches_f32", type=Path)
    ap.add_argument("grid_h", type=int)
    ap.add_argument("grid_w", type=int)
    ap.add_argument("output_f32", type=Path)
    ap.add_argument("output_u16", type=Path)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    gh, gw = args.grid_h, args.grid_w
    raw = np.fromfile(args.patches_f32, dtype=np.float32)
    expected = gh * gw * 3 * PATCH * PATCH
    if raw.size != expected:
        raise SystemExit(f"patch file has {raw.size} floats, expected {expected}")

    cfg = AutoConfig.from_pretrained(args.repo, trust_remote_code=True, revision=args.revision)
    # Make the released implementation use its explicit eager path. This avoids depending on
    # optional FlashAttention/SDPA kernels and preserves the checkpoint source's BF16 matmul +
    # FP32-softmax->BF16 behavior on CPU.
    cfg.vision_config._attn_implementation = "eager"

    vision_cls = get_class_from_dynamic_module(
        "modeling_kimi_vl.MoonVitPretrainedModel", args.repo, revision=args.revision
    )
    projector_cls = get_class_from_dynamic_module(
        "modeling_kimi_vl.KimiVLMultiModalProjector", args.repo, revision=args.revision
    )
    vision = vision_cls(cfg.vision_config).eval().to(dtype=torch.bfloat16)
    projector = projector_cls(cfg).eval().to(dtype=torch.bfloat16)

    weight_map = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    handles: dict[Path, object] = {}

    def get_tensor(name: str) -> torch.Tensor:
        shard = weight_map.get(name)
        if shard is None:
            raise KeyError(name)
        path = args.model_dir / shard
        if path not in handles:
            handles[path] = safe_open(path, framework="pt", device="cpu")
        return handles[path].get_tensor(name).contiguous()

    vision_sd = {}
    projector_sd = {}
    for full_name in weight_map:
        if full_name.startswith("vision_tower."):
            vision_sd[full_name[len("vision_tower."):]] = get_tensor(full_name)
        elif full_name.startswith("multi_modal_projector."):
            projector_sd[full_name[len("multi_modal_projector."):]] = get_tensor(full_name)

    vm = vision.load_state_dict(vision_sd, strict=True)
    pm = projector.load_state_dict(projector_sd, strict=True)
    assert not vm.missing_keys and not vm.unexpected_keys, vm
    assert not pm.missing_keys and not pm.unexpected_keys, pm

    pixels = torch.from_numpy(raw.reshape(gh * gw, 3, PATCH, PATCH)).to(torch.bfloat16)
    grid = torch.tensor([[gh, gw]], dtype=torch.long)
    with torch.inference_mode():
        image_features = vision(pixels, grid)
        projected = projector(image_features)

    expected_tokens = (gh // 2) * (gw // 2)
    assert projected.shape == (expected_tokens, int(cfg.text_config.hidden_size)), projected.shape
    assert projected.dtype == torch.bfloat16, projected.dtype

    projected.float().contiguous().numpy().astype(np.float32).tofile(args.output_f32)
    projected.contiguous().view(torch.uint16).numpy().tofile(args.output_u16)
    p = projected.float()
    print(
        "official BF16 vision: "
        f"revision={args.revision} grid={gh}x{gw} media_tokens={projected.shape[0]} dtype={projected.dtype} "
        f"min={p.min().item():.7g} max={p.max().item():.7g} "
        f"rms={p.square().mean().sqrt().item():.7g}"
    )


if __name__ == "__main__":
    main()
