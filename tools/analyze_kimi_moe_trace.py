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

Besides fixed keep-N masks, this tool emits diagnostic "unseen-cap" masks that
only disable experts never selected by the calibration traces, capped per layer.
These masks are intentionally conservative diagnostics, not evidence that an
unseen expert is globally unnecessary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
            rows.append((event, source_id, layer, expert, router_weight, output_l2, saliency))
    if not rows:
        raise ValueError(f"{path}: no trace rows")
    return rows


def entropy_metrics(values):
    total = float(sum(values))
    if total <= 0.0:
        return 0.0, 0.0
    probs = [float(v) / total for v in values if v > 0]
    h = -sum(p * math.log(p) for p in probs)
    n = len(values)
    normalized = h / math.log(n) if n > 1 else 0.0
    return normalized, math.exp(h)


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        v = values[order[pos]]
        while end < len(order) and values[order[end]] == v:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for j in range(pos, end):
            ranks[order[j]] = rank
        pos = end
    return ranks


def pearson(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    va = sum(x * x for x in da)
    vb = sum(x * x for x in db)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(da, db)) / math.sqrt(va * vb)


def spearman(a, b):
    return pearson(average_ranks(a), average_ranks(b))


def analyze(rows, n_experts: int, keep_values: list[int], coactivation_top: int,
            unseen_caps: list[int]):
    per_layer = defaultdict(lambda: {
        "selected": [0] * n_experts,
        "router_abs_sum": [0.0] * n_experts,
        "output_l2_sum": [0.0] * n_experts,
        "saliency_sum": [0.0] * n_experts,
        "sources": [set() for _ in range(n_experts)],
    })
    events = defaultdict(list)

    for event, source_id, layer, expert, weight, output_l2, saliency in rows:
        if expert >= n_experts:
            raise ValueError(f"trace expert {expert} exceeds --n-experts={n_experts}")
        s = per_layer[layer]
        s["selected"][expert] += 1
        s["router_abs_sum"][expert] += abs(weight)
        s["output_l2_sum"][expert] += output_l2
        s["saliency_sum"][expert] += saliency
        s["sources"][expert].add(source_id)
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
        "unseen_cap_masks": {},
    }

    support_hist = Counter()
    seen_per_layer = []
    total_saliency = 0.0

    for layer in sorted(per_layer):
        s = per_layer[layer]
        experts = []
        for expert in range(n_experts):
            count = s["selected"][expert]
            source_count = len(s["sources"][expert])
            support_hist[source_count] += 1
            experts.append({
                "expert": expert,
                "selected": count,
                "trace_file_count": source_count,
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
        seen = sum(1 for x in s["selected"] if x > 0)
        seen_per_layer.append(seen)
        sel_h, sel_eff = entropy_metrics(s["selected"])
        sal_h, sal_eff = entropy_metrics(s["saliency_sum"])
        sel_total = sum(s["selected"])
        sorted_sel = sorted(s["selected"], reverse=True)
        layer_saliency = sum(s["saliency_sum"])
        total_saliency += layer_saliency
        report["layers"][str(layer)] = {
            "events": layer_events[layer],
            "seen_experts": seen,
            "unseen_experts": n_experts - seen,
            "selection_entropy_normalized": sel_h,
            "selection_effective_experts": sel_eff,
            "saliency_entropy_normalized": sal_h,
            "saliency_effective_experts": sal_eff,
            "selection_top1_share": (sorted_sel[0] / sel_total) if sel_total else 0.0,
            "selection_top6_share": (sum(sorted_sel[:6]) / sel_total) if sel_total else 0.0,
            "frequency_saliency_spearman": spearman(s["selected"], s["saliency_sum"]),
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
        seen_disabled = 0
        disabled_saliency = 0.0
        per_layer_impact = {}
        for layer in sorted(per_layer):
            s = per_layer[layer]
            ranking = sorted(
                range(n_experts),
                key=lambda e: (s["saliency_sum"][e], s["selected"][e], e),
            )
            layer_disabled = ranking[:prune_n]
            layer_seen = sum(1 for expert in layer_disabled if s["selected"][expert] > 0)
            layer_saliency = sum(s["saliency_sum"])
            layer_disabled_saliency = sum(s["saliency_sum"][expert] for expert in layer_disabled)
            per_layer_impact[str(layer)] = {
                "disabled_count": len(layer_disabled),
                "disabled_seen_count": layer_seen,
                "disabled_unseen_count": len(layer_disabled) - layer_seen,
                "disabled_saliency_fraction": layer_disabled_saliency / layer_saliency if layer_saliency else 0.0,
            }
            seen_disabled += layer_seen
            disabled_saliency += layer_disabled_saliency
            for expert in layer_disabled:
                disabled.append((layer, expert, s["saliency_sum"][expert], s["selected"][expert]))
        masks[keep] = disabled
        report["masks"][str(keep)] = {
            "disabled_count": len(disabled),
            "disabled_per_layer": prune_n,
            "disabled_seen_count": seen_disabled,
            "disabled_unseen_count": len(disabled) - seen_disabled,
            "disabled_saliency_fraction": disabled_saliency / total_saliency if total_saliency else 0.0,
            "per_layer": per_layer_impact,
        }

    unseen_masks = {}
    for cap in unseen_caps:
        if cap < 0 or cap > n_experts:
            raise ValueError(f"invalid --unseen-cap={cap} for n_experts={n_experts}")
        disabled = []
        kept = []
        for layer in sorted(per_layer):
            s = per_layer[layer]
            unseen = [e for e in range(n_experts) if s["selected"][e] == 0]
            chosen = unseen[:cap]
            kept.append(n_experts - len(chosen))
            for expert in chosen:
                disabled.append((layer, expert, 0.0, 0))
        unseen_masks[cap] = disabled
        report["unseen_cap_masks"][str(cap)] = {
            "disabled_count": len(disabled),
            "disabled_seen_count": 0,
            "min_keep_per_layer": min(kept) if kept else n_experts,
            "mean_keep_per_layer": sum(kept) / len(kept) if kept else float(n_experts),
            "max_keep_per_layer": max(kept) if kept else n_experts,
        }

    layer_count = len(seen_per_layer)
    slots = layer_count * n_experts
    seen_slots = sum(seen_per_layer)
    report["coverage"] = {
        "layers": layer_count,
        "layer_expert_slots": slots,
        "seen_slots": seen_slots,
        "unseen_slots": slots - seen_slots,
        "seen_fraction": seen_slots / slots if slots else 0.0,
        "seen_min_per_layer": min(seen_per_layer) if seen_per_layer else 0,
        "seen_mean_per_layer": sum(seen_per_layer) / layer_count if layer_count else 0.0,
        "seen_max_per_layer": max(seen_per_layer) if seen_per_layer else 0,
        "trace_file_support_histogram": {str(k): support_hist[k] for k in sorted(support_hist)},
    }

    return report, masks, unseen_masks


def write_mask(path: Path, description: str, n_experts: int, disabled):
    with path.open("w", encoding="utf-8") as f:
        f.write("# KVL_MOE_MASK_V1\n")
        f.write(f"# n_experts={n_experts} {description}\n")
        f.write("# columns: layer expert ; comments show calibration saliency/count\n")
        for layer, expert, saliency, count in disabled:
            f.write(f"{layer} {expert} # saliency={saliency:.12g} selected={count}\n")


def n_experts_from_report(report):
    return int(report["n_experts"])


def write_layer_sensitivity(path: Path, report, keep_values, unseen_caps):
    fields = [
        "layer", "events", "seen_experts", "unseen_experts",
        "selection_effective_experts", "saliency_effective_experts",
        "frequency_saliency_spearman", "selection_top1_share", "selection_top6_share",
    ]
    for keep in keep_values:
        fields += [f"keep{keep}_disabled_seen", f"keep{keep}_disabled_saliency_fraction"]
    for cap in unseen_caps:
        fields += [f"unseen_cap{cap}_keep"]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for layer_s in sorted(report["layers"], key=int):
            d = report["layers"][layer_s]
            row = {
                "layer": int(layer_s),
                "events": d["events"],
                "seen_experts": d["seen_experts"],
                "unseen_experts": d["unseen_experts"],
                "selection_effective_experts": f'{d["selection_effective_experts"]:.6f}',
                "saliency_effective_experts": f'{d["saliency_effective_experts"]:.6f}',
                "frequency_saliency_spearman": f'{d["frequency_saliency_spearman"]:.6f}',
                "selection_top1_share": f'{d["selection_top1_share"]:.9f}',
                "selection_top6_share": f'{d["selection_top6_share"]:.9f}',
            }
            for keep in keep_values:
                impact = report["masks"][str(keep)]["per_layer"][layer_s]
                row[f"keep{keep}_disabled_seen"] = impact["disabled_seen_count"]
                row[f"keep{keep}_disabled_saliency_fraction"] = f'{impact["disabled_saliency_fraction"]:.12g}'
            for cap in unseen_caps:
                row[f"unseen_cap{cap}_keep"] = max(n_experts_from_report(report) - cap, d["seen_experts"])
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path, nargs="+")
    ap.add_argument("--n-experts", type=int, default=64)
    ap.add_argument("--keep", type=int, nargs="+", default=[62, 60, 58])
    ap.add_argument("--unseen-cap", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--coactivation-top", type=int, default=20)
    args = ap.parse_args()

    if args.n_experts <= 0 or args.n_experts > 256:
        raise SystemExit("--n-experts must be in 1..256")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_id, path in enumerate(args.trace, 1):
        rows.extend(parse_trace(path, source_id))
    report, masks, unseen_masks = analyze(
        rows, args.n_experts, args.keep, args.coactivation_top, args.unseen_cap
    )
    report["trace_files"] = [str(p) for p in args.trace]

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for keep, disabled in masks.items():
        write_mask(
            args.out_dir / f"mask-keep{keep}.txt",
            f"keep={keep} disabled_per_layer={args.n_experts - keep}",
            args.n_experts,
            disabled,
        )
    for cap, disabled in unseen_masks.items():
        write_mask(
            args.out_dir / f"mask-unseen-cap{cap}.txt",
            f"unseen_only=true cap_per_layer={cap}",
            args.n_experts,
            disabled,
        )
    write_layer_sensitivity(
        args.out_dir / "layer-sensitivity.tsv", report, args.keep, args.unseen_cap
    )

    layers = len(report["layers"])
    coverage = report["coverage"]
    print(
        "KIMI_MOE_TRACE_ANALYSIS_PASS "
        f"rows={report['trace_rows']} files={len(args.trace)} layers={layers} "
        f"n_experts={args.n_experts} keep={','.join(map(str, args.keep))} "
        f"seen_slots={coverage['seen_slots']}/{coverage['layer_expert_slots']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
