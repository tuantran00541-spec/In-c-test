#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kimi_compression_lab import (  # noqa: E402
    evaluate_candidate,
    mlp_forward,
    quantize_dequant_symmetric,
    run_npz,
)


class QuantizerTests(unittest.TestCase):
    def test_zero_matrix_is_exact(self) -> None:
        w = np.zeros((3, 7), dtype=np.float32)
        got, metrics = quantize_dequant_symmetric(w, 4, 4)
        np.testing.assert_array_equal(got, w)
        self.assertEqual(metrics.weight_mse, 0.0)
        self.assertEqual(metrics.weight_max_abs, 0.0)
        self.assertEqual(metrics.scale_count, 6)
        self.assertEqual(metrics.payload_bits, 3 * 7 * 4)
        self.assertEqual(metrics.scale_bits, 6 * 16)

    def test_simple_symmetric_levels_are_exact(self) -> None:
        # With Q4 and maxabs=7 each value below is exactly representable.
        w = np.array([[0.0, 1.0, -2.0, 7.0, -7.0]], dtype=np.float32)
        got, metrics = quantize_dequant_symmetric(w, 4, 128)
        np.testing.assert_allclose(got, w, rtol=0.0, atol=0.0)
        self.assertEqual(metrics.weight_mse, 0.0)

    def test_payload_projection_decreases_with_bits(self) -> None:
        w = np.arange(35, dtype=np.float32).reshape(5, 7) - 11.0
        totals = []
        for bits in (8, 6, 5, 4):
            _, m = quantize_dequant_symmetric(w, bits, 4)
            totals.append(m.payload_bits + m.scale_bits)
        self.assertGreater(totals[0], totals[1])
        self.assertGreater(totals[1], totals[2])
        self.assertGreater(totals[2], totals[3])

    def test_invalid_bits_rejected(self) -> None:
        w = np.ones((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            quantize_dequant_symmetric(w, 1, 128)
        with self.assertRaises(ValueError):
            quantize_dequant_symmetric(w, 9, 128)


class ExpertTests(unittest.TestCase):
    def _fixture(self):
        x = np.array([[1.0, 0.0], [0.5, -1.0]], dtype=np.float32)
        gate = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32)
        up = np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]], dtype=np.float32)
        down = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        return x, gate, up, down

    def test_forward_shape(self) -> None:
        x, gate, up, down = self._fixture()
        y = mlp_forward(x, gate, up, down)
        self.assertEqual(y.shape, (2, 2))

    def test_candidate_exact_for_integer_q4_fixture(self) -> None:
        x, gate, up, down = self._fixture()
        c = evaluate_candidate(x, gate, up, down, bits=4, group_size=128)
        self.assertEqual(c.output_mse, 0.0)
        self.assertEqual(c.relative_l2, 0.0)
        self.assertAlmostEqual(c.output_cosine, 1.0, places=12)
        self.assertGreater(c.projected_total_bytes_f16_scales, 0)

    def test_shape_mismatch_rejected(self) -> None:
        x, gate, up, down = self._fixture()
        with self.assertRaises(ValueError):
            evaluate_candidate(x[:, :1], gate, up, down, bits=4, group_size=128)

    def test_npz_result_marks_projection_only(self) -> None:
        x, gate, up, down = self._fixture()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "expert.npz"
            np.savez(
                p,
                x=x,
                gate=gate,
                up=up,
                down=down,
                meta_layer=np.array(6),
                meta_expert=np.array(30),
            )
            result = run_npz(p, [8, 4], 128)
        self.assertEqual(result["schema"], "kimi-expert-quant-sensitivity-v1")
        self.assertTrue(result["projection_only"])
        self.assertEqual(result["metadata"]["layer"], 6)
        self.assertEqual(result["metadata"]["expert"], 30)
        self.assertEqual([c["bits"] for c in result["candidates"]], [8, 4])

    def test_cli_writes_json(self) -> None:
        x, gate, up, down = self._fixture()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "expert.npz"
            out = td / "result.json"
            np.savez(src, x=x, gate=gate, up=up, down=down)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "kimi_compression_lab.py"),
                    str(src),
                    "--bits",
                    "8,6,4",
                    "--group-size",
                    "2",
                    "--output",
                    str(out),
                ],
                check=True,
            )
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual([c["bits"] for c in data["candidates"]], [8, 6, 4])
        self.assertTrue(data["projection_only"])


if __name__ == "__main__":
    unittest.main()
