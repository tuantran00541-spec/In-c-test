#!/usr/bin/env python3
"""Compare two Kimi Q8 MoE route traces event-by-event.

Trace rows use the functional-pruning format:
  event layer expert router_weight output_l2 saliency

For deterministic A/B runs over the same prompt, (event, layer) is expected to
align exactly. The report measures ordered-route equality, selected-set overlap,
expert substitutions, and router-weight changes for experts common to both runs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_trace(path: Path):
    events: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"{path}:{lineno}: expected 6 columns")
            event, layer, expert = map(int, parts[:3])
            weight = float(parts[3])
            if event <= 0 or layer < 0 or expert < 0 or not math.isfinite(weight):
                raise ValueError(f"{path}:{lineno}: invalid route row")
            events[(event, layer)].append((expert, weight))
    if not events:
        raise ValueError(f"{path}: no route events")
    return events


def compare(base, variant):
    base_keys = set(base)
    variant_keys = set(variant)
    if base_keys != variant_keys:
        missing = sorted(base_keys - variant_keys)[:8]
        extra = sorted(variant_keys - base_keys)[:8]
        raise ValueError(f"event keys differ: missing={missing} extra={extra}")

    per_layer = defaultdict(lambda: {
        "events": 0,
        "ordered_exact_events": 0,
        "set_exact_events": 0,
        "base_selected": 0,
        "common_selected": 0,
        "substitutions": 0,
        "common_weight_l1": 0.0,
        "common_weight_max_abs": 0.0,
    })
    totals = {
        "events": 0,
        "ordered_exact_events": 0,
        "set_exact_events": 0,
        "base_selected": 0,
        "common_selected": 0,
        "substitutions": 0,
        "common_weight_l1": 0.0,
        "common_weight_max_abs": 0.0,
    }
    first_divergence = None

    for event, layer in sorted(base_keys):
        b = base[(event, layer)]
        v = variant[(event, layer)]
        b_ids = [e for e, _ in b]
        v_ids = [e for e, _ in v]
        if len(b_ids) != len(set(b_ids)) or len(v_ids) != len(set(v_ids)):
            raise ValueError(f"duplicate expert in event={event} layer={layer}")
        b_set, v_set = set(b_ids), set(v_ids)
        common = b_set & v_set
        substitutions = len(b_set - common)
        ordered_exact = b_ids == v_ids
        set_exact = b_set == v_set
        b_weight = dict(b)
        v_weight = dict(v)
        weight_l1 = sum(abs(b_weight[e] - v_weight[e]) for e in common)
        weight_max = max((abs(b_weight[e] - v_weight[e]) for e in common), default=0.0)

        if first_divergence is None and not ordered_exact:
            first_divergence = {
                "event": event,
                "layer": layer,
                "base_ids": b_ids,
                "variant_ids": v_ids,
                "common": sorted(common),
            }

        for acc in (totals, per_layer[layer]):
            acc["events"] += 1
            acc["ordered_exact_events"] += int(ordered_exact)
            acc["set_exact_events"] += int(set_exact)
            acc["base_selected"] += len(b_set)
            acc["common_selected"] += len(common)
            acc["substitutions"] += substitutions
            acc["common_weight_l1"] += weight_l1
            acc["common_weight_max_abs"] = max(acc["common_weight_max_abs"], weight_max)

    def finish(d):
        events = d["events"]
        base_selected = d["base_selected"]
        common_selected = d["common_selected"]
        return {
            **d,
            "ordered_exact_fraction": d["ordered_exact_events"] / events if events else 0.0,
            "set_exact_fraction": d["set_exact_events"] / events if events else 0.0,
            "selected_retention_fraction": common_selected / base_selected if base_selected else 0.0,
            "mean_substitutions_per_event": d["substitutions"] / events if events else 0.0,
            "mean_common_weight_l1_per_event": d["common_weight_l1"] / events if events else 0.0,
        }

    return {
        "summary": finish(totals),
        "first_divergence": first_divergence,
        "layers": {str(layer): finish(per_layer[layer]) for layer in sorted(per_layer)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("variant", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = compare(read_trace(args.baseline), read_trace(args.variant))
    report["baseline"] = str(args.baseline)
    report["variant"] = str(args.variant)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    s = report["summary"]
    print(
        "KIMI_ROUTE_COMPARE_PASS "
        f"events={s['events']} set_exact={s['set_exact_events']} "
        f"substitutions={s['substitutions']} retention={s['selected_retention_fraction']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
