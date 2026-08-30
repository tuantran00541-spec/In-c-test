#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CompressionQuickstartTests(unittest.TestCase):
    def test_documented_tools_exist(self) -> None:
        text = (ROOT / "tools" / "README_COMPRESSION_LAB.txt").read_text(encoding="utf-8")
        for name in (
            "kimi_expert_reservoir.py",
            "kimi_compression_lab.py",
            "kimi_mixed_bit_plan.py",
        ):
            self.assertIn(name, text)
            self.assertTrue((ROOT / "tools" / name).is_file())


if __name__ == "__main__":
    unittest.main()
