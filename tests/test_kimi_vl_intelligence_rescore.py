#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from rescore_kimi_dynskip_vl_intelligence import rescore


def row(case_id: str, domain: str, baseline: str, candidate: str, exact: bool = False) -> dict:
    return {
        "id": case_id,
        "domain": domain,
        "baseline": {"text": baseline, "score": {"correct": False}},
        "candidate": {"text": candidate, "score": {"correct": False}},
        "comparison": {"token_exact": exact},
    }


def main() -> None:
    report = {
        "schema": "demo",
        "cases": [
            row("hard-weighted-inventory", "visual_counting_arithmetic", "FINAL=37", "The result is 37."),
            row("hard-size-ordering", "visual_comparison_ordering", "THIRD=red", "THIRD: green"),
        ],
    }
    scored = rescore(report)
    s = scored["summary"]
    assert s["baseline_semantic_correct_cases"] == 1
    assert s["candidate_semantic_correct_cases"] == 2
    assert s["baseline_semantic_retained_cases"] == 1
    assert s["semantic_retention_regressions"] == []
    assert s["semantic_candidate_improvements"] == ["hard-size-ordering"]
    print("KIMI_VL_INTELLIGENCE_RESCORE_UNIT_PASS")


if __name__ == "__main__":
    main()
