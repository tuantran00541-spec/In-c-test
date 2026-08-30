#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qplan", ROOT / "tools" / "qwen38_dense_compression_plan.py")
qplan = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qplan)


def fake_config():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "vocab_size": 248320,
            "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)],
        },
        "vision_config": {"depth": 27, "hidden_size": 1152},
    }


def fake_index():
    wm = {}
    for layer in range(64):
        for part in ("gate_proj", "up_proj", "down_proj"):
            wm[f"model.language_model.layers.{layer}.mlp.{part}.weight"] = f"model-{layer//4:05d}.safetensors"
    wm["model.visual.patch_embed.proj.weight"] = "vision.safetensors"
    wm["mtp.fc.weight"] = "mtp.safetensors"
    return {"weight_map": wm}


def test_config():
    got = qplan.validate_config(fake_config())
    assert got["layers"] == 64
    assert got["full_attention_layers"] == 16
    assert got["linear_attention_layers"] == 48


def test_index():
    got = qplan.validate_index(fake_index())
    assert got["dense_mlp_tensors"] == 192
    assert got["vision_tensor_count"] == 1
    assert got["mtp_tensor_count"] == 1


def test_exact_mlp_count_and_q4_projection():
    assert qplan.mlp_params() == 17_112_760_320
    q4 = qplan.projection(4, 128)
    assert q4["payload_bytes"] == 8_556_380_160
    assert q4["fp16_scale_bytes"] == 267_386_880
    assert abs(q4["projected_total_gib"] - 8.2177734375) < 1e-12


def test_projection_monotonic():
    sizes = [qplan.projection(b, 128)["projected_total_bytes"] for b in (8, 6, 5, 4, 3, 2)]
    assert sizes == sorted(sizes, reverse=True)


def main():
    test_config()
    test_index()
    test_exact_mlp_count_and_q4_projection()
    test_projection_monotonic()
    print("QWEN38_DENSE_COMPRESSION_PLAN_TEST_PASS")


if __name__ == "__main__":
    main()
