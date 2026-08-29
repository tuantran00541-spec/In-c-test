#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "data" / "kimi-functional-pruning-suite.json"


def main() -> int:
    data = json.loads(SUITE.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise AssertionError(data.get("version"))
    calibration = data.get("calibration", [])
    heldout = data.get("heldout", [])
    if len(calibration) < 12 or len(heldout) < 12:
        raise AssertionError((len(calibration), len(heldout)))

    ids = []
    prompts = []
    for split_name, split in (("calibration", calibration), ("heldout", heldout)):
        categories = set()
        for item in split:
            ident = item.get("id")
            prompt = item.get("prompt")
            category = item.get("category")
            max_new = item.get("max_new")
            if not isinstance(ident, str) or not ident.startswith("cal-" if split_name == "calibration" else "eval-"):
                raise AssertionError((split_name, ident))
            if not isinstance(prompt, str) or len(prompt.strip()) < 8:
                raise AssertionError((ident, prompt))
            if not isinstance(category, str) or not category:
                raise AssertionError((ident, category))
            if not isinstance(max_new, int) or max_new < 1 or max_new > 64:
                raise AssertionError((ident, max_new))
            ids.append(ident)
            prompts.append(prompt.strip())
            categories.add(category)
        if len(categories) < 8:
            raise AssertionError((split_name, categories))

    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate suite ids")
    if len(prompts) != len(set(prompts)):
        raise AssertionError("calibration/held-out prompts must be disjoint")

    metrics = set(data.get("planned_metrics", []))
    required = {
        "full_vs_variant_generated_token_exact",
        "first_generated_token_divergence_position",
        "per_layer_route_set_overlap",
        "per_layer_route_substitution_count",
        "expert_cache_reads_and_bytes",
    }
    if not required.issubset(metrics):
        raise AssertionError(required - metrics)

    print(
        "KIMI_FUNCTIONAL_SUITE_PASS "
        f"calibration={len(calibration)} heldout={len(heldout)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
