#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kimi_measure_bf16_expert import (  # noqa: E402
    PINNED_REVISION,
    _bf16_bytes_to_f32,
    load_tensor,
    measure,
)


def f32_to_bf16_bytes(a: np.ndarray) -> bytes:
    x = np.asarray(a, dtype=np.float32)
    u = x.view(np.uint32)
    # Test values are chosen exactly representable in BF16, so truncation is exact.
    return (u >> np.uint32(16)).astype("<u2").tobytes()


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    payload = bytearray()
    for name, arr in tensors.items():
        raw = f32_to_bf16_bytes(arr)
        a = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [a, len(payload)],
        }
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(h)) + h + bytes(payload))


class Bf16ReaderTests(unittest.TestCase):
    def test_raw_bf16_conversion(self) -> None:
        src = np.array([[0.0, 1.0, -2.0, 0.5]], dtype=np.float32)
        got = _bf16_bytes_to_f32(f32_to_bf16_bytes(src), list(src.shape))
        np.testing.assert_array_equal(got, src)

    def test_load_tensor_and_measure(self) -> None:
        layer, expert = 1, 2
        prefix = f"language_model.model.layers.{layer}.mlp.experts.{expert}"
        gate = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32)
        up = np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]], dtype=np.float32)
        down = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        tensors = {
            prefix + ".gate_proj.weight": gate,
            prefix + ".up_proj.weight": up,
            prefix + ".down_proj.weight": down,
        }
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shard = td / "model-00001-of-00001.safetensors"
            write_safetensors(shard, tensors)
            index = {"weight_map": {name: shard.name for name in tensors}}
            (td / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            x = np.array([[1.0, 0.0], [0.5, -1.0]], dtype=np.float32)
            res = td / "reservoir.npz"
            np.savez(
                res,
                x=x,
                meta_layer=np.array(layer),
                meta_expert=np.array(expert),
                meta_seen=np.array(7),
                meta_kept=np.array(2),
            )
            got_gate, shard_name = load_tensor(td, index["weight_map"], prefix + ".gate_proj.weight")
            np.testing.assert_array_equal(got_gate, gate)
            self.assertEqual(shard_name, shard.name)
            result = measure(td, res, [8, 4], 128, PINNED_REVISION)
        self.assertEqual(result["source_revision"], PINNED_REVISION)
        self.assertEqual(result["metadata"]["layer"], layer)
        self.assertEqual(result["metadata"]["expert"], expert)
        self.assertEqual(result["metadata"]["seen"], 7)
        self.assertEqual([c["bits"] for c in result["candidates"]], [8, 4])
        self.assertEqual(result["candidates"][1]["output_mse"], 0.0)
        self.assertTrue(result["projection_only"])

    def test_unpinned_revision_refused_before_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with self.assertRaises(ValueError):
                measure(td, td / "missing.npz", [4], 128, "moving-main")


if __name__ == "__main__":
    unittest.main()
