#!/usr/bin/env python3
"""Pack official Kimi-VL MoonViT + multimodal projector into vision.bin/vision.idx.

The binary/index layout is intentionally identical to trunk.bin/trunk.idx so the already
validated direct-I/O store is reused. Vision lives in independent files to support phase-based
memory: load/compute image features, close/free vision state, then enter long text decode.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct

import torch
from safetensors import safe_open

ALIGN = 4096
MAGIC = b"KVLTRNK1"
VERSION = 1
GLOBAL = 0xFFFFFFFF
DT_BF16 = 1
HDR = struct.Struct("<8s4I2Q")
REC = struct.Struct("<8I3Q")

K = {
    "patch_w": 100, "patch_b": 101, "pos": 102,
    "final_w": 103, "final_b": 104,
    "norm0_w": 110, "norm0_b": 111,
    "wqkv_w": 112, "wqkv_b": 113,
    "wo_w": 114, "wo_b": 115,
    "norm1_w": 116, "norm1_b": 117,
    "mlp0_w": 118, "mlp0_b": 119,
    "mlp1_w": 120, "mlp1_b": 121,
    "proj_norm_w": 130, "proj_norm_b": 131,
    "proj_l1_w": 132, "proj_l1_b": 133,
    "proj_l2_w": 134, "proj_l2_b": 135,
}


def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def specs(n_layers: int):
    yield GLOBAL, "patch_w", "vision_tower.patch_embed.proj.weight"
    yield GLOBAL, "patch_b", "vision_tower.patch_embed.proj.bias"
    yield GLOBAL, "pos", "vision_tower.patch_embed.pos_emb.weight"
    yield GLOBAL, "final_w", "vision_tower.encoder.final_layernorm.weight"
    yield GLOBAL, "final_b", "vision_tower.encoder.final_layernorm.bias"
    yield GLOBAL, "proj_norm_w", "multi_modal_projector.pre_norm.weight"
    yield GLOBAL, "proj_norm_b", "multi_modal_projector.pre_norm.bias"
    yield GLOBAL, "proj_l1_w", "multi_modal_projector.linear_1.weight"
    yield GLOBAL, "proj_l1_b", "multi_modal_projector.linear_1.bias"
    yield GLOBAL, "proj_l2_w", "multi_modal_projector.linear_2.weight"
    yield GLOBAL, "proj_l2_b", "multi_modal_projector.linear_2.bias"
    for layer in range(n_layers):
        p = f"vision_tower.encoder.blocks.{layer}"
        for key, suffix in (
            ("norm0_w", "norm0.weight"), ("norm0_b", "norm0.bias"),
            ("wqkv_w", "wqkv.weight"), ("wqkv_b", "wqkv.bias"),
            ("wo_w", "wo.weight"), ("wo_b", "wo.bias"),
            ("norm1_w", "norm1.weight"), ("norm1_b", "norm1.bias"),
            ("mlp0_w", "mlp.fc0.weight"), ("mlp0_b", "mlp.fc0.bias"),
            ("mlp1_w", "mlp.fc1.weight"), ("mlp1_b", "mlp.fc1.bias"),
        ):
            yield layer, key, f"{p}.{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=pathlib.Path)
    ap.add_argument("out_dir", type=pathlib.Path)
    args = ap.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text())
    vcfg = cfg["vision_config"]
    n_layers = int(vcfg["num_hidden_layers"])
    wm = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    binp = args.out_dir / "vision.bin"
    idxp = args.out_dir / "vision.idx"

    handles: dict[pathlib.Path, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = wm.get(name)
        if shard is None:
            raise KeyError(name)
        path = args.model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(f"{path} required for {name}")
        # The released model places all vision/projector tensors in one shard, but keep the
        # code generic for future revisions.
        h = handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt", device="cpu")
            handles[path] = h
        return h.get_tensor(name)

    records = []
    with open(binp, "wb") as bf:
        for layer, key, name in specs(n_layers):
            t = get(name).contiguous().cpu()
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            data = t.view(torch.uint16).numpy().tobytes()
            at = align_up(bf.tell())
            if at > bf.tell():
                bf.write(b"\0" * (at - bf.tell()))
            payload = len(data)
            readb = align_up(payload)
            bf.write(data)
            bf.write(b"\0" * (readb - payload))
            dims = (list(t.shape) + [0, 0, 0, 0])[:4]
            records.append((layer, K[key], DT_BF16, len(t.shape), *map(int, dims), at, readb, payload))
            print(f"packed V={layer if layer != GLOBAL else 'global'} {key:12s} {tuple(t.shape)} {payload/1048576:.2f} MiB")
        data_bytes = bf.tell()

    blob = bytearray(HDR.pack(MAGIC, VERSION, ALIGN, len(records), 0, HDR.size, data_bytes))
    for rec in records:
        blob.extend(REC.pack(*rec))
    idxp.write_bytes(blob)
    print(f"vision records={len(records)} data={data_bytes/1073741824:.3f} GiB index={len(blob)} bytes")


if __name__ == "__main__":
    main()
