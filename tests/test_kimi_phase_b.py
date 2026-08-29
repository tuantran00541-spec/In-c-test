#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "kimi_phase_b", ROOT / "tools" / "run_kimi_pruning_phase_b.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)

assert mod.first_divergence([1, 2, 3], [1, 2, 3]) is None
assert mod.first_divergence([1, 2, 3], [1, 9, 3]) == 1
assert mod.first_divergence([1, 2], [1, 2, 3]) == 2
assert mod.sanity_contains("Tokyo", ["tokyo"]) is True
assert mod.sanity_contains("continue", ["break"]) is False
assert mod.sanity_contains("anything", None) is None


def run(cache_ops, cache_mib, seconds):
    return {
        "cache": {"read_ops": cache_ops, "bytes_read_mib": cache_mib},
        "timing": {"total_seconds": seconds},
    }


rows = [
    {
        "id": "a",
        "variants": {"full": run(10, 20.0, 2.0), "cand": run(8, 18.0, 1.8)},
        "comparison": {
            "token_exact": True,
            "first_divergence_position": None,
            "baseline_sanity": True,
            "candidate_sanity": True,
        },
    },
    {
        "id": "b",
        "variants": {"full": run(11, 21.0, 2.1), "cand": run(9, 19.0, 1.9)},
        "comparison": {
            "token_exact": False,
            "first_divergence_position": 3,
            "baseline_sanity": None,
            "candidate_sanity": None,
        },
    },
]
s = mod.summarize(rows, "cand")
assert s["prompts"] == 2
assert s["token_exact_prompts"] == 1
assert s["divergent_prompts"] == [{"id": "b", "first_divergence_position": 3}]
assert s["baseline_sanity_pass"] == 1
assert s["candidate_sanity_pass"] == 1
assert s["full_cache_read_ops_total"] == 21
assert s["candidate_cache_read_ops_total"] == 17
print("KIMI_PHASE_B_TEST_PASS")
