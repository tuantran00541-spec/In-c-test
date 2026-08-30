#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "run_kimi_pruning_vl_phase_b", ROOT / "tools" / "run_kimi_pruning_vl_phase_b.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)

assert mod.first_divergence([1, 2, 3], [1, 2, 3]) is None
assert mod.first_divergence([1, 2, 3], [1, 4, 3]) == 1
assert mod.first_divergence([1, 2], [1, 2, 3]) == 2

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    suite = td / "suite.json"
    suite.write_text(json.dumps({
        "version": 1,
        "cases": [
            {"id": "a", "image": "a.png", "prompt": "p1", "max_new": 8},
            {"id": "b", "image": "b.png", "prompt": "p2", "max_new": 12},
            {"id": "c", "image": "c.png", "prompt": "p3", "max_new": 16},
            {"id": "d", "image": "d.png", "prompt": "p4", "max_new": 20}
        ]
    }), encoding="utf-8")
    data = mod.validate_suite(suite)
    assert sum(x["max_new"] for x in data["cases"]) == 56

    rows = [
        {
            "id": "a",
            "full": {
                "generated_ids": [1, 2],
                "cache": {"read_ops": 10, "bytes_read_mib": 20.5},
                "vl_timing": {"text_total_seconds": 3.0, "vision_seconds": 1.0},
            },
            "candidate": {
                "generated_ids": [1, 2],
                "cache": {"read_ops": 9, "bytes_read_mib": 19.5},
                "vl_timing": {"text_total_seconds": 2.8, "vision_seconds": 1.1},
            },
            "comparison": {"token_exact": True, "first_divergence_position": None},
        },
        {
            "id": "b",
            "full": {
                "generated_ids": [3, 4, 5],
                "cache": {"read_ops": 12, "bytes_read_mib": 30.0},
                "vl_timing": {"text_total_seconds": 4.0, "vision_seconds": 1.2},
            },
            "candidate": {
                "generated_ids": [3, 7, 5],
                "cache": {"read_ops": 11, "bytes_read_mib": 29.0},
                "vl_timing": {"text_total_seconds": 4.1, "vision_seconds": 1.2},
            },
            "comparison": {"token_exact": False, "first_divergence_position": 1},
        },
    ]
    a = mod.summarize(rows)
    assert a["cases"] == 2
    assert a["token_exact_cases"] == 1
    assert a["divergent_cases"] == [{"id": "b", "first_divergence_position": 1}]
    assert a["full_generated_tokens_total"] == 5
    assert a["candidate_generated_tokens_total"] == 5
    assert a["full_cache_read_ops_total"] == 22
    assert a["candidate_cache_read_ops_total"] == 20
    assert abs(a["full_cache_bytes_read_mib_total"] - 50.5) < 1e-9

print("KIMI_VL_PHASE_B_TEST_PASS")
