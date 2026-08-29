#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_kimi_mask_novelty", ROOT / "tools" / "analyze_kimi_mask_novelty.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    (td / "heldout" / "p1" / "full").mkdir(parents=True)
    (td / "heldout" / "p2" / "full").mkdir(parents=True)
    (td / "comparisons" / "p1" / "adaptive-unseen-cap6").mkdir(parents=True)
    (td / "comparisons" / "p2" / "adaptive-unseen-cap6").mkdir(parents=True)
    mask = td / "mask.txt"
    mask.write_text("# KVL_MOE_MASK_V1\n1 7\n2 9\n", encoding="utf-8")
    (td / "heldout" / "p1" / "full" / "route.tsv").write_text(
        "# event layer expert router_weight output_l2 saliency\n"
        "1 1 7 0.2 3.0 0.6\n"
        "1 1 4 0.3 2.0 0.6\n",
        encoding="utf-8",
    )
    (td / "heldout" / "p2" / "full" / "route.tsv").write_text(
        "1 2 3 0.2 1.0 0.2\n", encoding="utf-8"
    )
    for prompt, subs in (("p1", 2), ("p2", 0)):
        (td / "comparisons" / prompt / "adaptive-unseen-cap6" / "route.json").write_text(
            json.dumps({"summary": {"substitutions": subs}}), encoding="utf-8"
        )
    r = mod.analyze(td, mask, "adaptive-unseen-cap6")
    assert r["mask_entries"] == 2
    assert r["aggregate"]["direct_masked_selections"] == 1
    assert r["aggregate"]["unique_masked_slots_hit"] == 1
    assert r["aggregate"]["masked_slots_never_hit"] == 1
    assert r["aggregate"]["prompts_with_direct_mask_hit"] == 1
    assert r["aggregate"]["route_substitutions"] == 2
    assert r["aggregate"]["cascade_substitutions"] == 1

print("KIMI_MASK_NOVELTY_TEST_PASS")
