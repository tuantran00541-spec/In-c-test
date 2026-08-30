#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CompressionSchemaTests(unittest.TestCase):
    def test_schema_sentinel(self) -> None:
        data = json.loads((ROOT / "tests" / "data" / "kimi-compression-lab-schema.json").read_text(encoding="utf-8"))
        self.assertEqual(data["pinned_revision"], "398eede0903cd983a2bfa0cc634e9ac1d843f375")
        self.assertTrue(data["projection_is_not_physical_measurement"])
        self.assertFalse(data["native_low_bit_format_implemented"])
        self.assertEqual(
            set(data["schemas"]),
            {
                "kimi-expert-activation-reservoir-v1",
                "kimi-expert-quant-sensitivity-v1",
                "kimi-mixed-bit-plan-v1",
            },
        )


if __name__ == "__main__":
    unittest.main()
