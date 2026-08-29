#!/usr/bin/env python3
"""Analyze Kimi MoE functional traces and emit light expert-pruning masks.

Trace rows come from the Q8 runtime when KVL_MOE_TRACE is set:
  event layer expert router_weight output_l2 saliency

Multiple trace files are accepted. Event ids are namespaced by file because each
kvl_generate process starts its local event counter from one.

The first pruning score is deliberately simple and reproducible: total
abs(router_weight) * ||expert_output||_2 over the calibration trace. Experts
never selected on the calibration set therefore receive score zero. Masks are
logical only; no expert bytes are deleted by this tool.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_trace(path: Path, source_id: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"{path}:{lineno}: expected 6 columns, got {len(parts)}")
            local_event, layer, expert = map(int, parts[:3])
            router_weight, output_l2, saliency = map(float, parts[3:])
            if local_event <= 0 or layer < 0 or expert < 0:
                raise ValueError(f"{path}:{lineno}: invalid event/layer/expert")
            event = (source_id << 32) | local_event
            rows.append((event, layer, expert, router_weight, output_l2, saliency))
    if not rows:
        raise ValueError(f"{path}: no trace rows")
    return rows


def analyze(rows, n_experts: int, keep_values: list[int], coactivation_top: int):
    per_layer = defaultdict(lambda: {
        "selected": [0] * n_experts,
        "router_abs_sum": [0.0] * n_experts,
        "output_l2_sum": [0.0] * n_experts,
        "saliency_sum": [0.0] * n_experts,
    })
    events = defaultdict(list)

    for event, layer, expert, weight, output_l2, saliency in rows:
        if expert >= n_experts:
            raise ValueError(f"trace expert {expert} exceeds --n-experts={n_experts}")
        s = per_layer[layer]
        s["selected"][expert] += 1
        s["router_abs_sum"][expert] += abs(weight)
        s["output_l2_sum"][expert] += output_l2
        s["saliency_sum"][expert] += saliency
        events[(event, layer)].append(expert)

    coact = defaultdict(Counter)
    layer_events = Counter()
    for (_, layer), ids in events.items():
        unique = sorted(set(ids))
        layer_events[layer] += 1
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                coact[layer][(a, b)] += 1

    report = {
        "trace_rows": len(rows),
        "n_experts": n_experts,
        "layers": {},
        "masks": {},
    }

    for layer in sorted(per_layer):
        s = per_layer[layer]
        experts = []
        for expert in range(n_experts):
            count = s["selected"][expert]
            experts.append({
                "expert": expert,
                "selected": count,
                "router_abs_sum": s["router_abs_sum"][expert],
                "output_l2_sum": s["output_l2_sum"][expert],
                "saliency_sum": s["saliency_sum"][expert],
                "saliency_mean_when_selected": s["saliency_sum"][expert] / count if count else 0.0,
            })
        ranking = sorted(
            range(n_experts),
            key=lambda e: (s["saliency_sum"][e], s["selected"][e], e),
        )
        top_pairs = [
            {"a": a, "b": b, "count": count}
            for (a, b), count in coact[layer].most_common(coactivation_top)
        ]
        report["layers"][str(layer)] = {
            "events": layer_events[layer],
            "experts": experts,
            "prune_order_low_to_high": ranking,
            "top_coactivation_pairs": top_pairs,
        }

    masks = {}
    for keep in keep_values:
        if keep < 1 or keep > n_experts:
            raise ValueError(f"invalid --keep={keep} for n_experts={n_experts}")
        prune_n = n_experts - keep
        disabled = []
        for layer in sorted(per_layer):
            s = per_layer[layer]
            ranking = sorted(
                range(n_experts),
                key=lambda e: (s["saliency_sum"][e], s["selected"][e], e),
            )
            for expert in ranking[:prune_n]:
                disabled.append((layer, expert, s["saliency_sum"][expert], s["selected"][expert]))
        masks[keep] = disabled
        report["masks"][str(keep)] = {
            "disabled_count": len(disabled),
            "disabled_per_layer": prune_n,
        }

    return report, masks


def write_mask(path: Path, keep: int, n_experts: int, disabled):
    with path.open("w", encoding="utf-8") as f:
        f.write("# KVL_MOE_MASK_V1\n")
        f.write(f"# n_experts={n_experts} keep={keep} disabled_per_layer={n_experts - keep}\n")
        f.write("# columns: layer expert ; comments show calibration saliency/count\n")
        for layer, expert, saliency, count in disabled:
            f.write(f"{layer} {expert} # saliency={saliency:.12g} selected={count}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path, nargs="+")
    ap.add_argument("--n-experts", type=int, default=64)
    ap.add_argument("--keep", type=int, nargs="+", default=[62, 60, 58])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--coactivation-top", type=int, default=20)
    args = ap.parse_args()

    if args.n_experts <= 0 or args.n_experts > 256:
        raise SystemExit("--n-experts must be in 1..256")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_id, path in enumerate(args.trace, 1):
        rows.extend(parse_trace(path, source_id))
    report, masks = analyze(rows, args.n_experts, args.keep, args.coactivation_top)
    report["trace_files"] = [str(p) for p in args.trace]

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for keep, disabled in masks.items():
        write_mask(args.out_dir / f"mask-keep{keep}.txt", keep, args.n_experts, disabled)

    layers = len(report["layers"])
    print(
        "KIMI_MOE_TRACE_ANALYSIS_PASS "
        f"rows={report['trace_rows']} files={len(args.trace)} layers={layers} "
        f"n_experts={args.n_experts} keep={','.join(map(str, args.keep))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
