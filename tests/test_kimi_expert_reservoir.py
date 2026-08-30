#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kimi_expert_reservoir import build_reservoir  # noqa: E402


class ExpertReservoirTests(unittest.TestCase):
    def test_bounded_and_deterministic(self) -> None:
        rows = []
        for i in range(10):
            rows.append({"layer": 1, "expert": 2, "x": [float(i), 1.0], "source": "text", "kind": "text"})
        for i in range(3):
            rows.append({"layer": 1, "expert": 3, "x": [float(i), -1.0], "source": "vl", "kind": "media"})
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            trace = td / "trace.jsonl"
            trace.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
            a = td / "a"
            b = td / "b"
            ma = build_reservoir(trace, a, capacity=4, hidden=2, seed=7)
            mb = build_reservoir(trace, b, capacity=4, hidden=2, seed=7)
            xa = np.load(a / "layer-01-expert-02.npz")["x"]
            xb = np.load(b / "layer-01-expert-02.npz")["x"]
            np.testing.assert_array_equal(xa, xb)
        self.assertEqual(ma["rows"], 13)
        self.assertEqual(ma["expert_count"], 2)
        self.assertEqual(ma["source_counts"], {"text": 10, "vl": 3})
        self.assertEqual(ma["kind_counts"], {"media": 3, "text": 10})
        self.assertEqual(mb["experts"][0]["seen"], 10)
        self.assertEqual(mb["experts"][0]["kept"], 4)
        self.assertEqual(mb["experts"][1]["kept"], 3)

    def test_bad_hidden_size_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            trace = td / "trace.jsonl"
            trace.write_text(json.dumps({"layer": 1, "expert": 0, "x": [1.0]}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_reservoir(trace, td / "out", capacity=2, hidden=2, seed=1)


if __name__ == "__main__":
    unittest.main()
