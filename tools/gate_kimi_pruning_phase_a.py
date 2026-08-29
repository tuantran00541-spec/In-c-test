#!/usr/bin/env python3
"""Screen Kimi functional-pruning Phase-A results for Phase-B eligibility.

This is an engineering sensitivity gate, not a semantic-quality proof.
It consumes phase-a-summary.json and applies configurable conservative guards:
- first-token exactness on every held-out prompt,
- next-token argmax agreement on every prompt,
- minimum top-k overlap,
- maximum Jensen-Shannon divergence,
- minimum route selected-retention,
- calibration route-coverage warning.

Passing means only "worth spending a longer generation A/B run on".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("keep60", "keep58", "adaptive-unseen-cap6")


def disabled_count(summary: dict, variant: str) -> int:
    cal = summary["calibration"]
    if variant.startswith("keep"):
        keep = variant.removeprefix("keep")
        return int(cal["masks"][keep]["disabled_count"])
    if variant == "adaptive-unseen-cap6":
        return int(cal["unseen_cap_masks"]["6"]["disabled_count"])
    raise KeyError(variant)


def screen(summary: dict, *, min_topk: float, max_js: float,
           min_route_retention: float, min_coverage: float) -> dict:
    if summary.get("schema_version") != 1:
        raise ValueError(f"unsupported summary schema: {summary.get('schema_version')}")
    coverage = summary["calibration"]["coverage"]
    slots = int(coverage["layer_expert_slots"])
    seen = int(coverage["seen_slots"])
    seen_fraction = (seen / slots) if slots else 0.0
    coverage_warning = seen_fraction < min_coverage

    results = {}
    eligible = []
    for name in VARIANTS:
        a = summary["aggregate"][name]
        prompts = int(a["prompts"])
        checks = {
            "all_first_tokens_exact": int(a["first_token_exact"]) == prompts,
            "all_argmax_agree": int(a["logit_argmax_agree"]) == prompts,
            "topk_overlap": float(a["logit_min_topk_overlap"]) >= min_topk,
            "js_divergence": float(a["logit_max_js_divergence"]) <= max_js,
            "route_retention": float(a["route_min_selected_retention"]) >= min_route_retention,
        }
        ok = all(checks.values())
        row = {
            "eligible_for_phase_b": ok,
            "checks": checks,
            "disabled_layer_expert_slots": disabled_count(summary, name),
            "prompts": prompts,
            "first_token_exact": int(a["first_token_exact"]),
            "argmax_agree": int(a["logit_argmax_agree"]),
            "min_topk_overlap": float(a["logit_min_topk_overlap"]),
            "max_js_divergence": float(a["logit_max_js_divergence"]),
            "min_route_retention": float(a["route_min_selected_retention"]),
            "route_substitutions": int(a["route_substitutions"]),
            "cache_read_ops_total": int(a["cache_read_ops_total"]),
            "cache_bytes_read_mib_total": float(a["cache_bytes_read_mib_total"]),
        }
        results[name] = row
        if ok:
            eligible.append(name)

    eligible.sort(
        key=lambda n: (
            -results[n]["disabled_layer_expert_slots"],
            results[n]["max_js_divergence"],
            -results[n]["min_topk_overlap"],
        )
    )
    return {
        "schema_version": 1,
        "scope": "Phase-B eligibility screen only; not a semantic-quality or global-pruning safety claim",
        "thresholds": {
            "min_topk_overlap": min_topk,
            "max_js_divergence": max_js,
            "min_route_retention": min_route_retention,
            "min_calibration_seen_fraction": min_coverage,
        },
        "calibration": {
            "seen_slots": seen,
            "layer_expert_slots": slots,
            "seen_fraction": seen_fraction,
            "coverage_warning": coverage_warning,
        },
        "variants": results,
        "phase_b_candidates": eligible,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-topk", type=float, default=0.8)
    ap.add_argument("--max-js", type=float, default=0.01)
    ap.add_argument("--min-route-retention", type=float, default=0.95)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    args = ap.parse_args()
    for name, value in (
        ("--min-topk", args.min_topk),
        ("--max-js", args.max_js),
        ("--min-route-retention", args.min_route_retention),
        ("--min-coverage", args.min_coverage),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0,1]")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = screen(
        summary,
        min_topk=args.min_topk,
        max_js=args.max_js,
        min_route_retention=args.min_route_retention,
        min_coverage=args.min_coverage,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    c = report["calibration"]
    print(
        "KIMI_PHASE_A_GATE "
        f"coverage={c['seen_slots']}/{c['layer_expert_slots']} "
        f"warning={str(c['coverage_warning']).lower()} "
        f"candidates={','.join(report['phase_b_candidates']) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
