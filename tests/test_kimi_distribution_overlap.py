#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "compare_kimi_logits.py"
spec = importlib.util.spec_from_file_location("compare_kimi_logits", TOOL)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def main() -> None:
    # These logits encode p=(0.8, 0.2) and q=(0.5, 0.5). Their full-vocabulary
    # overlap is 0.7 and TV is 0.3. This becomes EvoESAP only when both records
    # came from the same teacher-forced prefix.
    p = [math.log(0.8), math.log(0.2)]
    q = [math.log(0.5), math.log(0.5)]
    row = mod.compare_record(p, q, topk=2)
    assert math.isclose(row["probability_overlap"], 0.7, rel_tol=1e-12)
    assert math.isclose(row["total_variation"], 0.3, rel_tol=1e-12)
    assert math.isclose(
        row["probability_overlap"], 1.0 - row["total_variation"], rel_tol=1e-12
    )

    same = mod.compare_record(p, p, topk=2)
    assert math.isclose(same["probability_overlap"], 1.0, rel_tol=1e-12)
    assert math.isclose(same["total_variation"], 0.0, abs_tol=1e-12)
    print("KIMI_DISTRIBUTION_OVERLAP_UNIT_PASS")


if __name__ == "__main__":
    main()
