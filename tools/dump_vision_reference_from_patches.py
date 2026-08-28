#!/usr/bin/env python3
"""Run the released MoonViT+projector math on arbitrary preprocessed patch grids.

This intentionally reads only the official vision/projector tensors from safetensors, so a
196x196 user image can be compared against kvl_vision without instantiating the text model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

D, HEADS, HD, I = 1152, 16, 72, 4304
PATCH = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("patches_f32", type=Path)
    ap.add_argument("grid_h", type=int)
    ap.add_argument("grid_w", type=int)
    ap.add_argument("output_f32", type=Path)
    args = ap.parse_args()

    gh, gw = args.grid_h, args.grid_w
    if gh <= 0 or gw <= 0 or gh % 2 or gw % 2:
        raise SystemExit("grid must be positive and divisible by 2x2 merge")

    cfg = json.loads((args.model_dir / "config.json").read_text())["vision_config"]
    wm = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(name: str) -> torch.Tensor:
        path = args.model_dir / wm[name]
        if path not in handles:
            handles[path] = safe_open(path, framework="pt", device="cpu")
        return handles[path].get_tensor(name).float().contiguous()

    raw = np.fromfile(args.patches_f32, dtype=np.float32)
    expected = gh * gw * 3 * PATCH * PATCH
    if raw.size != expected:
        raise SystemExit(f"patch file has {raw.size} floats, expected {expected}")
    patches = torch.from_numpy(raw.reshape(gh * gw, 3, PATCH, PATCH)).float()

    x = F.conv2d(
        patches,
        get("vision_tower.patch_embed.proj.weight"),
        get("vision_tower.patch_embed.proj.bias"),
        stride=PATCH,
    ).reshape(gh * gw, D)
    pos = get("vision_tower.patch_embed.pos_emb.weight")
    pos2 = F.interpolate(pos.permute(2, 0, 1).unsqueeze(0), size=(gh, gw), mode="bicubic")
    x = x + pos2.squeeze(0).permute(1, 2, 0).reshape(gh * gw, D)

    pairs = HD // 2
    freqs = torch.empty((gh * gw, pairs), dtype=torch.complex64)
    for r in range(gh):
        for c in range(gw):
            vals = []
            for p in range(pairs):
                i = p // 2
                inv = 10000.0 ** (-(4.0 * i) / HD)
                angle = (r if p & 1 else c) * inv
                vals.append(complex(math.cos(angle), math.sin(angle)))
            freqs[r * gw + c] = torch.tensor(vals, dtype=torch.complex64)

    seq = gh * gw
    for layer in range(int(cfg["num_hidden_layers"])):
        p = f"vision_tower.encoder.blocks.{layer}"
        n0 = F.layer_norm(x, (D,), get(p + ".norm0.weight"), get(p + ".norm0.bias"), 1e-5)
        qkv = F.linear(n0, get(p + ".wqkv.weight"), get(p + ".wqkv.bias"))
        qkv = qkv.view(seq, 3, HEADS, HD)
        q, k, val = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        fq = freqs.unsqueeze(1)
        qc = torch.view_as_complex(q.float().reshape(seq, HEADS, HD // 2, 2)) * fq
        kc = torch.view_as_complex(k.float().reshape(seq, HEADS, HD // 2, 2)) * fq
        q = torch.view_as_real(qc).flatten(-2)
        k = torch.view_as_real(kc).flatten(-2)
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), val.transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(HD)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attn = torch.matmul(probs, vh).transpose(0, 1).reshape(seq, D)
        x = x + F.linear(attn, get(p + ".wo.weight"), get(p + ".wo.bias"))
        n1 = F.layer_norm(x, (D,), get(p + ".norm1.weight"), get(p + ".norm1.bias"), 1e-5)
        mlp = F.linear(n1, get(p + ".mlp.fc0.weight"), get(p + ".mlp.fc0.bias"))
        mlp = F.gelu(mlp, approximate="tanh")
        x = x + F.linear(mlp, get(p + ".mlp.fc1.weight"), get(p + ".mlp.fc1.bias"))

    x = F.layer_norm(
        x, (D,), get("vision_tower.encoder.final_layernorm.weight"),
        get("vision_tower.encoder.final_layernorm.bias"), 1e-5
    )
    merged = (
        x.view(gh // 2, 2, gw // 2, 2, D)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view((gh // 2) * (gw // 2), 4 * D)
    )
    # Projector pre_norm is applied per 1152-wide patch before concatenation in the released
    # implementation; reshape around LayerNorm so the operation exactly matches that contract.
    merged = F.layer_norm(
        merged.view(-1, D), (D,), get("multi_modal_projector.pre_norm.weight"),
        get("multi_modal_projector.pre_norm.bias"), 1e-5
    ).view((gh // 2) * (gw // 2), 4 * D)
    y = F.linear(merged, get("multi_modal_projector.linear_1.weight"), get("multi_modal_projector.linear_1.bias"))
    y = F.gelu(y, approximate="none")
    y = F.linear(y, get("multi_modal_projector.linear_2.weight"), get("multi_modal_projector.linear_2.bias"))
    y.detach().contiguous().numpy().astype("float32").tofile(args.output_f32)
    print(
        f"vision actual-image oracle: grid={gh}x{gw} media_tokens={y.shape[0]} "
        f"min={y.min().item():.7g} max={y.max().item():.7g} "
        f"rms={y.square().mean().sqrt().item():.7g}"
    )


if __name__ == "__main__":
    main()
