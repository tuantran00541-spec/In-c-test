#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_kimi_compression_q0_measure import (  # noqa: E402
    build_plan,
    expert_shards,
    validate_manifest,
)


class Q0MeasurePlanTests(unittest.TestCase):
    def test_expert_shards_deduplicates_and_sorts(self) -> None:
        prefix = "language_model.model.layers.6.mlp.experts.30"
        wm = {
            prefix + ".gate_proj.weight": "b.safetensors",
            prefix + ".up_proj.weight": "a.safetensors",
            prefix + ".down_proj.weight": "b.safetensors",
        }
        self.assertEqual(expert_shards(wm, 6, 30), ("a.safetensors", "b.safetensors"))

    def test_build_plan_binds_expected_reservoir(self) -> None:
        prefix = "language_model.model.layers.6.mlp.experts.30"
        index = {"weight_map": {
            prefix + ".gate_proj.weight": "a.safetensors",
            prefix + ".up_proj.weight": "a.safetensors",
            prefix + ".down_proj.weight": "a.safetensors",
        }}
        manifest = {
            "experts": [
                {"rank": 1, "layer": 6, "expert": 30, "c2_score": 0.1, "boundary_role": "test"}
            ]
        }
        plan = build_plan(index, manifest, Path("reservoirs"))
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["shards"], ("a.safetensors",))
        self.assertEqual(plan[0]["reservoir"], Path("reservoirs/layer-06-expert-30.npz"))

    def test_real_pilot_manifest_validates(self) -> None:
        doc = validate_manifest(ROOT / "tests" / "data" / "kimi-compression-q0-pilot.json")
        self.assertEqual(len(doc["experts"]), 14)
        self.assertEqual(doc["bits"], [8, 6, 5, 4])
        self.assertEqual(doc["group_size"], 128)
        self.assertEqual((doc["experts"][6]["layer"], doc["experts"][6]["expert"]), (11, 43))
        self.assertEqual((doc["experts"][7]["layer"], doc["experts"][7]["expert"]), (8, 32))

    def test_duplicate_manifest_expert_rejected(self) -> None:
        doc = {
            "schema": "kimi-compression-q0-pilot-v1",
            "source_revision": "398eede0903cd983a2bfa0cc634e9ac1d843f375",
            "bits": [4],
            "group_size": 128,
            "experts": [
                {"layer": 1, "expert": 2},
                {"layer": 1, "expert": 2},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pilot.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_manifest(p)


if __name__ == "__main__":
    unittest.main()
