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
    "run_kimi_pruning_vl_guard", ROOT / "tools" / "run_kimi_pruning_vl_guard.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    suite = td / "suite.json"
    suite.write_text(json.dumps({
        "version": 1,
        "cases": [
            {"id": "a", "image": "a.png", "prompt": "p1"},
            {"id": "b", "image": "b.png", "prompt": "p2"},
            {"id": "c", "image": "c.png", "prompt": "p3"},
            {"id": "d", "image": "d.png", "prompt": "p4"}
        ]
    }), encoding="utf-8")
    assert len(mod.validate_suite(suite)["cases"]) == 4

    mask = td / "mask.txt"
    mask.write_text("# KVL_MOE_MASK_V1\n1 7\n2 9\n", encoding="utf-8")
    m = mod.read_mask(mask)
    assert m == {(1, 7), (2, 9)}

    trace = td / "route.tsv"
    trace.write_text(
        "# event layer expert router_weight output_l2 saliency\n"
        "1 1 7 0.2 3.0 0.6\n"
        "1 1 4 0.3 2.0 0.6\n"
        "2 2 9 0.1 5.0 0.5\n",
        encoding="utf-8",
    )
    hit = mod.direct_mask_hits(trace, m)
    assert hit["selections"] == 2
    assert hit["unique_slots"] == 2
    assert abs(sum(x["saliency"] for x in hit["slots"]) - 1.1) < 1e-9

    stderr = (
        "[kvl-vl] generated ids: 12 34\n"
        "[kvl-vl] timing vision=1.250s first_text_token=2.500s "
        "avg_next=0.750s text_total=3.250s generated=2\n"
    )
    assert mod.parse_generated(stderr) == [12, 34]
    timing = mod.parse_vl_runtime(stderr)["vl_timing"]
    assert timing["vision_seconds"] == 1.25
    assert timing["first_text_token_seconds"] == 2.5
    assert timing["generated"] == 2

print("KIMI_VL_PRUNING_GUARD_TEST_PASS")
