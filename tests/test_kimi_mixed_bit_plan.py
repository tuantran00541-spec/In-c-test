#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kimi_mixed_bit_plan import Choice, load_choices, optimize  # noqa: E402


class MixedBitPlanTests(unittest.TestCase):
    def test_optimizer_uses_high_precision_when_budget_allows(self) -> None:
        experts = [
            ("L01E00", [Choice(8, 100, 0.0), Choice(4, 50, 0.2)]),
            ("L01E01", [Choice(8, 100, 0.0), Choice(4, 50, 0.1)]),
        ]
        plan = optimize(experts, budget_bytes=200, quantum=1)
        self.assertEqual(plan["projected_bytes"], 200)
        self.assertEqual([x["bits"] for x in plan["assignment"]], [8, 8])
        self.assertEqual(plan["total_error"], 0.0)

    def test_optimizer_spends_bits_on_more_sensitive_expert(self) -> None:
        experts = [
            ("A", [Choice(8, 100, 0.0), Choice(4, 50, 10.0)]),
            ("B", [Choice(8, 100, 0.0), Choice(4, 50, 1.0)]),
        ]
        plan = optimize(experts, budget_bytes=150, quantum=1)
        by_name = {x["expert"]: x for x in plan["assignment"]}
        self.assertEqual(by_name["A"]["bits"], 8)
        self.assertEqual(by_name["B"]["bits"], 4)
        self.assertEqual(plan["total_error"], 1.0)

    def test_quantum_rounding_never_breaks_exact_budget(self) -> None:
        experts = [
            ("A", [Choice(8, 65, 0.0), Choice(4, 31, 1.0)]),
            ("B", [Choice(8, 65, 0.0), Choice(4, 31, 1.0)]),
        ]
        plan = optimize(experts, budget_bytes=96, quantum=32)
        self.assertLessEqual(plan["projected_bytes"], 96)
        self.assertEqual(plan["projected_bytes"], 62)

    def test_impossible_budget_rejected(self) -> None:
        experts = [("A", [Choice(4, 100, 1.0)])]
        with self.assertRaises(ValueError):
            optimize(experts, budget_bytes=50, quantum=1)

    def test_loader_uses_layer_expert_metadata(self) -> None:
        doc = {
            "schema": "kimi-expert-quant-sensitivity-v1",
            "metadata": {"layer": 6, "expert": 30},
            "candidates": [
                {"bits": 8, "projected_total_bytes_f16_scales": 100, "relative_l2": 0.01},
                {"bits": 4, "projected_total_bytes_f16_scales": 50, "relative_l2": 0.20},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            got = load_choices([p], "relative_l2")
        self.assertEqual(got[0][0], "L06E30")
        self.assertEqual([c.bits for c in got[0][1]], [8, 4])


if __name__ == "__main__":
    unittest.main()
