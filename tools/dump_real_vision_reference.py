#!/usr/bin/env python3
"""Create a tiny official-weight MoonViT+projector oracle without instantiating the text model.

A deterministic 2x2 patch grid (28x28 image-equivalent) keeps attention tiny while exercising
patch projection, interpolated learned positions, all 27 MoonViT blocks, 2D RoPE, 2x2 merge and
the multimodal projector. Weights are read directly from the released safetensor shard.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

D, HEADS, HD, I = 1152, 16, 72, 4304
PATCH = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.model_dir / "config.json").read_text())["vision_config"]
    wm = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(name: str) -> torch.Tensor:
        path = args.model_dir / wm[name]
        if path not in handles:
            handles[path] = safe_open(path, framework="pt", device="cpu")
        return handles[path].get_tensor(name).float().contiguous()

    # Deterministic normalized-pixel-like patches. The preprocessing stage is tested separately;
    # this oracle isolates exact vision-model math.
    n = 4 * 3 * PATCH * PATCH
    v = torch.arange(n, dtype=torch.float32)
    patches = (0.55 * torch.sin(v * 0.013) + 0.25 * torch.cos(v * 0.031)).reshape(4, 3, PATCH, PATCH)
    patches.contiguous().numpy().astype("float32").tofile(args.out_dir / "patches.f32")
    grid_h = grid_w = 2

    x = F.conv2d(
        patches,
        get("vision_tower.patch_embed.proj.weight"),
        get("vision_tower.patch_embed.proj.bias"),
        stride=PATCH,
    ).reshape(4, D)
    pos = get("vision_tower.patch_embed.pos_emb.weight")
    pos2 = F.interpolate(pos.permute(2, 0, 1).unsqueeze(0), size=(grid_h, grid_w), mode="bicubic")
    x = x + pos2.squeeze(0).permute(1, 2, 0).reshape(4, D)

    # Official 2D RoPE frequencies: complex pairs alternate column/x and row/y frequencies.
    pairs = HD // 2
    freqs = torch.empty((4, pairs), dtype=torch.complex64)
    for r in range(grid_h):
        for c in range(grid_w):
            vals = []
            for p in range(pairs):
                i = p // 2
                inv = 10000.0 ** (-(4.0 * i) / HD)
                angle = (r if p & 1 else c) * inv
                vals.append(complex(math.cos(angle), math.sin(angle)))
            freqs[r * grid_w + c] = torch.tensor(vals, dtype=torch.complex64)

    for layer in range(int(cfg["num_hidden_layers"])):
        p = f"vision_tower.encoder.blocks.{layer}"
        n0 = F.layer_norm(x, (D,), get(p + ".norm0.weight"), get(p + ".norm0.bias"), 1e-5)
        qkv = F.linear(n0, get(p + ".wqkv.weight"), get(p + ".wqkv.bias"))
        qkv = qkv.view(4, 3, HEADS, HD)
        q, k, val = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        fq = freqs.unsqueeze(1)
        qc = torch.view_as_complex(q.float().reshape(4, HEADS, HD // 2, 2)) * fq
        kc = torch.view_as_complex(k.float().reshape(4, HEADS, HD // 2, 2)) * fq
        q = torch.view_as_real(qc).flatten(-2)
        k = torch.view_as_real(kc).flatten(-2)
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), val.transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(HD)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attn = torch.matmul(probs, vh).transpose(0, 1).reshape(4, D)
        x = x + F.linear(attn, get(p + ".wo.weight"), get(p + ".wo.bias"))
        n1 = F.layer_norm(x, (D,), get(p + ".norm1.weight"), get(p + ".norm1.bias"), 1e-5)
        mlp = F.linear(n1, get(p + ".mlp.fc0.weight"), get(p + ".mlp.fc0.bias"))
        mlp = F.gelu(mlp, approximate="tanh")
        x = x + F.linear(mlp, get(p + ".mlp.fc1.weight"), get(p + ".mlp.fc1.bias"))

    x = F.layer_norm(
        x, (D,), get("vision_tower.encoder.final_layernorm.weight"),
        get("vision_tower.encoder.final_layernorm.bias"), 1e-5
    )
    merged = x.view(1, 2, 1, 2, D).permute(0, 2, 1, 3, 4).contiguous().view(1, 4, D)
    merged = F.layer_norm(
        merged, (D,), get("multi_modal_projector.pre_norm.weight"),
        get("multi_modal_projector.pre_norm.bias"), 1e-5
    ).view(1, 4 * D)
    y = F.linear(merged, get("multi_modal_projector.linear_1.weight"), get("multi_modal_projector.linear_1.bias"))
    y = F.gelu(y, approximate="none")
    y = F.linear(y, get("multi_modal_projector.linear_2.weight"), get("multi_modal_projector.linear_2.bias"))
    y.detach().contiguous().numpy().astype("float32").tofile(args.out_dir / "reference.f32")
    print(f"vision oracle: grid=2x2 media_tokens=1 min={y.min().item():.7g} max={y.max().item():.7g} rms={y.square().mean().sqrt().item():.7g}")


if __name__ == "__main__":
    main()
