#!/usr/bin/env python3
"""Metadata-only compression planner for pinned Qwen3.8-27B.

This script deliberately does not download model shards or claim physical sizes.
It validates the released architecture from local config/index metadata and emits
idealized dense-MLP weight-only quantization projections with FP16 group scales.
The projections are a planning aid for later BF16 sensitivity experiments.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.8-27B"
PINNED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HIDDEN = 5120
INTERMEDIATE = 17408
LAYERS = 64
VISION_DEPTH = 27
VISION_HIDDEN = 1152
FULL_ATTN_INTERVAL = 4
MLP_PARTS = ("gate_proj", "up_proj", "down_proj")
OFFICIAL_LAYER_PREFIX = "model.language_model.layers"


def validate_config(cfg: dict) -> dict:
    errors: list[str] = []
    if cfg.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        errors.append(f"architectures={cfg.get('architectures')!r}")
    if cfg.get("model_type") != "qwen3_5":
        errors.append(f"model_type={cfg.get('model_type')!r}")
    tc = cfg.get("text_config") or {}
    expected = {
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "full_attention_interval": FULL_ATTN_INTERVAL,
        "vocab_size": 248320,
    }
    for key, value in expected.items():
        if tc.get(key) != value:
            errors.append(f"text_config.{key}={tc.get(key)!r} expected={value}")
    layer_types = tc.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != LAYERS:
        errors.append(f"layer_types length={len(layer_types) if isinstance(layer_types, list) else None}")
    else:
        for i, kind in enumerate(layer_types):
            want = "full_attention" if i % FULL_ATTN_INTERVAL == FULL_ATTN_INTERVAL - 1 else "linear_attention"
            if kind != want:
                errors.append(f"layer_types[{i}]={kind!r} expected={want!r}")
                break
    vc = cfg.get("vision_config") or {}
    if vc.get("depth") != VISION_DEPTH:
        errors.append(f"vision.depth={vc.get('depth')!r}")
    if vc.get("hidden_size") != VISION_HIDDEN:
        errors.append(f"vision.hidden_size={vc.get('hidden_size')!r}")
    if errors:
        raise ValueError("not pinned Qwen3.8-27B architecture: " + "; ".join(errors))
    return {
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "layers": LAYERS,
        "full_attention_layers": LAYERS // FULL_ATTN_INTERVAL,
        "linear_attention_layers": LAYERS - LAYERS // FULL_ATTN_INTERVAL,
        "vision_depth": VISION_DEPTH,
    }


def validate_index(index: dict) -> dict:
    wm = index.get("weight_map")
    if not isinstance(wm, dict) or not wm:
        raise ValueError("checkpoint index has no weight_map")
    missing: list[str] = []
    for layer in range(LAYERS):
        p = f"{OFFICIAL_LAYER_PREFIX}.{layer}.mlp"
        for part in MLP_PARTS:
            name = f"{p}.{part}.weight"
            if name not in wm:
                missing.append(name)
    if missing:
        raise ValueError(f"missing dense MLP tensors; first={missing[:3]}")
    vision = [k for k in wm if k.startswith("model.visual.")]
    mtp = [k for k in wm if k.startswith("mtp.")]
    if not vision:
        raise ValueError("vision tensor partition is empty")
    if not mtp:
        raise ValueError("MTP tensor partition is empty")
    return {
        "weight_map_entries": len(wm),
        "dense_mlp_tensors": LAYERS * len(MLP_PARTS),
        "vision_tensor_count": len(vision),
        "mtp_tensor_count": len(mtp),
        "source_shards": len(set(wm.values())),
    }


def mlp_params() -> int:
    # SwiGLU: gate [I,H] + up [I,H] + down [H,I].
    return LAYERS * 3 * HIDDEN * INTERMEDIATE


def scale_count(group_size: int) -> int:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    per_layer = (
        2 * INTERMEDIATE * math.ceil(HIDDEN / group_size)
        + HIDDEN * math.ceil(INTERMEDIATE / group_size)
    )
    return LAYERS * per_layer


def projection(bits: int, group_size: int) -> dict:
    if not 2 <= bits <= 8:
        raise ValueError("bits must be in [2,8]")
    params = mlp_params()
    scales = scale_count(group_size)
    payload_bytes = (params * bits + 7) // 8
    scale_bytes = scales * 2  # FP16 scales, projection only
    return {
        "bits": bits,
        "group_size": group_size,
        "mlp_params": params,
        "payload_bytes": payload_bytes,
        "fp16_scale_bytes": scale_bytes,
        "projected_total_bytes": payload_bytes + scale_bytes,
        "projected_total_gib": (payload_bytes + scale_bytes) / (1024 ** 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--bits", default="8,6,5,4,3,2")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = {
        "schema": "qwen38-27b-dense-compression-plan-v1",
        "model_id": MODEL_ID,
        "pinned_revision": PINNED_REVISION,
        "projection_only": True,
        "notes": [
            "Sizes cover only the 64 dense SwiGLU MLPs, not attention, embeddings, LM head, vision, MTP or container metadata.",
            "No native low-bit file is written by this planner.",
        ],
    }
    if args.config:
        result["architecture"] = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    if args.index:
        result["index"] = validate_index(json.loads(args.index.read_text(encoding="utf-8")))
    bits = [int(x) for x in args.bits.split(",") if x.strip()]
    result["dense_mlp_bf16_bytes"] = mlp_params() * 2
    result["dense_mlp_bf16_gib"] = mlp_params() * 2 / (1024 ** 3)
    result["projections"] = [projection(b, args.group_size) for b in bits]
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
