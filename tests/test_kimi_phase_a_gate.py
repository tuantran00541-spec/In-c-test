#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kimi_phase_a_gate", ROOT / "tools" / "gate_kimi_pruning_phase_a.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def aggregate(prompts=14, token=14, argmax=14, topk=1.0, js=0.0, retention=1.0,
              substitutions=0):
    return {
        "prompts": prompts,
        "first_token_exact": token,
        "route_substitutions": substitutions,
        "route_min_selected_retention": retention,
        "route_min_set_exact_fraction": 1.0,
        "logit_argmax_agree": argmax,
        "logit_max_abs_delta": 0.0,
        "logit_max_js_divergence": js,
        "logit_min_topk_overlap": topk,
        "cache_read_ops_total": 10,
        "cache_bytes_read_mib_total": 20.0,
        "first_token_seconds_total": 30.0,
    }


summary = {
    "schema_version": 1,
    "calibration": {
        "coverage": {
            "seen_slots": 1600,
            "layer_expert_slots": 1664,
        },
        "masks": {
            "60": {"disabled_count": 104},
            "58": {"disabled_count": 156},
        },
        "unseen_cap_masks": {
            "6": {"disabled_count": 120},
        },
    },
    "aggregate": {
        "keep60": aggregate(js=0.002, topk=0.9, retention=0.98, substitutions=4),
        "keep58": aggregate(token=13, argmax=13, js=0.03, topk=0.7, retention=0.93,
                            substitutions=20),
        "adaptive-unseen-cap6": aggregate(js=0.001, topk=1.0, retention=0.99,
                                          substitutions=2),
    },
}

report = mod.screen(
    summary,
    min_topk=0.8,
    max_js=0.01,
    min_route_retention=0.95,
    min_coverage=0.95,
)
assert report["calibration"]["coverage_warning"] is False
assert report["variants"]["keep60"]["eligible_for_phase_b"] is True
assert report["variants"]["keep58"]["eligible_for_phase_b"] is False
assert report["variants"]["adaptive-unseen-cap6"]["eligible_for_phase_b"] is True
assert report["phase_b_candidates"] == ["adaptive-unseen-cap6", "keep60"]
print("KIMI_PHASE_A_GATE_TEST_PASS")
