#!/usr/bin/env python3
"""Build conservative modality-aware REAP masks for Kimi-VL.

The Q8 runtime already emits the REAP primitive through KVL_MOE_TRACE:

    event layer expert router_weight output_l2 saliency

where saliency = abs(router_weight) * ||expert_output||_2.

This tool differs from the older functional-pruning analyzer in three important
ways:

1. REAP importance is the *mean saliency when selected*, rather than total
   saliency. This avoids turning raw route frequency into the importance score.
2. VL max_new=1 traces are split back into media-token and text-token events
   using the exact layer-major ordering and prompt-id mapping already used by
   run_kimi_pruning_vl_guard.py.
3. Each modality is normalized independently per layer, then the final expert
   score is the MAX over text / VL-text / media. An expert that is important in
   only one modality is therefore protected instead of being averaged away.

Masks are selected globally from the lowest final scores while respecting
per-layer and per-router-group caps. Candidate masks are nested by construction.
This is a research ranking/screening tool; passing a mask is not a global model
quality claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MEDIA_PAD_ID = 163605


@dataclass(frozen=True)
class TraceRow:
    event: int
    layer: int
    expert: int
    router_weight: float
    output_l2: float
    saliency: float


def parse_kind_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected KIND=PATH")
    kind, raw = value.split("=", 1)
    kind = kind.strip()
    if not kind or not raw.strip():
        raise argparse.ArgumentTypeError("expected non-empty KIND=PATH")
    return kind, Path(raw)


def parse_vl_pair(value: str) -> tuple[Path, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected TRACE=PROMPT_IDS")
    trace, prompt_ids = value.split("=", 1)
    if not trace.strip() or not prompt_ids.strip():
        raise argparse.ArgumentTypeError("expected non-empty TRACE=PROMPT_IDS")
    return Path(trace), Path(prompt_ids)


def read_trace(path: Path) -> list[TraceRow]:
    rows: list[TraceRow] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{path}:{lineno}: expected six columns")
        event, layer, expert = map(int, parts[:3])
        weight, output_l2, saliency = map(float, parts[3:])
        if event <= 0 or layer <= 0 or expert < 0:
            raise ValueError(f"{path}:{lineno}: invalid event/layer/expert")
        if not all(math.isfinite(x) for x in (weight, output_l2, saliency)):
            raise ValueError(f"{path}:{lineno}: non-finite metric")
        if output_l2 < 0.0 or saliency < 0.0:
            raise ValueError(f"{path}:{lineno}: negative norm/saliency")
        rows.append(TraceRow(event, layer, expert, weight, output_l2, saliency))
    if not rows:
        raise ValueError(f"{path}: empty trace")
    return rows


def read_prompt_ids(path: Path) -> list[int]:
    ids: list[int] = []
    for lineno, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            token = int(line)
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: invalid token id {line!r}") from e
        if token < 0:
            raise ValueError(f"{path}:{lineno}: negative token id")
        ids.append(token)
    if not ids:
        raise ValueError(f"{path}: empty prompt ids")
    return ids


def vl_event_kind(row: TraceRow, prompt_ids: list[int]) -> str:
    """Map a max_new=1 layer-major VL prefill event to media or VL text."""
    n = len(prompt_ids)
    zero = row.event - 1
    expected_layer = zero // n + 1
    position = zero % n
    if expected_layer != row.layer:
        raise ValueError(
            f"VL trace ordering mismatch event={row.event} layer={row.layer} "
            f"expected_layer={expected_layer} prompt_tokens={n}"
        )
    return "media" if prompt_ids[position] == MEDIA_PAD_ID else "vl_text"


def positive_median(values: list[float]) -> float:
    positive = [x for x in values if x > 0.0]
    return statistics.median(positive) if positive else 1.0


def build_report(
    text_traces: list[tuple[str, Path]],
    vl_pairs: list[tuple[Path, Path]],
    n_experts: int,
    n_groups: int,
    targets: list[int],
    max_disabled_per_layer: int,
    max_disabled_per_group: int,
) -> tuple[dict, dict[int, list[dict]]]:
    if n_experts <= 0 or n_groups <= 0 or n_experts % n_groups:
        raise ValueError("n_experts must be positive and divisible by n_groups")
    if not text_traces and not vl_pairs:
        raise ValueError("at least one trace is required")
    if not targets or any(t <= 0 for t in targets) or targets != sorted(set(targets)):
        raise ValueError("targets must be unique positive integers in increasing order")
    if max_disabled_per_layer <= 0 or max_disabled_per_group <= 0:
        raise ValueError("mask caps must be positive")

    # stats[(kind, layer, expert)] = [selected_count, saliency_sum]
    stats: dict[tuple[str, int, int], list[float]] = defaultdict(lambda: [0, 0.0])
    trace_rows: dict[str, int] = defaultdict(int)
    layers: set[int] = set()

    for kind, path in text_traces:
        for row in read_trace(path):
            if row.expert >= n_experts:
                raise ValueError(f"{path}: expert {row.expert} >= n_experts={n_experts}")
            key = (kind, row.layer, row.expert)
            stats[key][0] += 1
            stats[key][1] += row.saliency
            trace_rows[kind] += 1
            layers.add(row.layer)

    for trace_path, ids_path in vl_pairs:
        prompt_ids = read_prompt_ids(ids_path)
        for row in read_trace(trace_path):
            if row.expert >= n_experts:
                raise ValueError(f"{trace_path}: expert {row.expert} >= n_experts={n_experts}")
            kind = vl_event_kind(row, prompt_ids)
            key = (kind, row.layer, row.expert)
            stats[key][0] += 1
            stats[key][1] += row.saliency
            trace_rows[kind] += 1
            layers.add(row.layer)

    kinds = sorted(set(k for k, _, _ in stats))
    if "media" not in kinds and vl_pairs:
        raise ValueError("VL traces contained no media-token events")

    # REAP uses mean contribution when the expert is selected. Normalize each
    # modality independently inside each layer so media/text scale differences
    # cannot suppress a specialist merely because another modality has larger
    # absolute output norms.
    scales: dict[tuple[str, int], float] = {}
    for kind in kinds:
        for layer in layers:
            means = []
            for expert in range(n_experts):
                count, total = stats[(kind, layer, expert)]
                means.append(total / count if count else 0.0)
            scales[(kind, layer)] = positive_median(means)

    expert_rows: list[dict] = []
    for layer in sorted(layers):
        for expert in range(n_experts):
            by_kind = {}
            final_score = 0.0
            total_selected = 0
            support_kinds = 0
            for kind in kinds:
                count, total = stats[(kind, layer, expert)]
                count = int(count)
                mean = total / count if count else 0.0
                scale = scales[(kind, layer)]
                normalized = mean / scale if mean > 0.0 and scale > 0.0 else 0.0
                if count:
                    support_kinds += 1
                total_selected += count
                final_score = max(final_score, normalized)
                by_kind[kind] = {
                    "selected": count,
                    "saliency_sum": total,
                    "saliency_mean_when_selected": mean,
                    "layer_positive_median": scale,
                    "normalized_reap": normalized,
                }
            expert_rows.append({
                "layer": layer,
                "expert": expert,
                "final_score": final_score,
                "support_kinds": support_kinds,
                "selected_total": total_selected,
                "by_kind": by_kind,
            })

    # One greedy order, snapshotted at each target, guarantees nested masks.
    order = sorted(
        expert_rows,
        key=lambda r: (
            r["final_score"],
            r["support_kinds"],
            r["selected_total"],
            r["layer"],
            r["expert"],
        ),
    )
    per_group = n_experts // n_groups
    layer_used: dict[int, int] = defaultdict(int)
    group_used: dict[tuple[int, int], int] = defaultdict(int)
    chosen: list[dict] = []
    snapshots: dict[int, list[dict]] = {}
    wanted = set(targets)
    max_target = max(targets)
    for row in order:
        layer = int(row["layer"])
        expert = int(row["expert"])
        group = expert // per_group
        if layer_used[layer] >= max_disabled_per_layer:
            continue
        if group_used[(layer, group)] >= max_disabled_per_group:
            continue
        chosen.append(row)
        layer_used[layer] += 1
        group_used[(layer, group)] += 1
        if len(chosen) in wanted:
            snapshots[len(chosen)] = list(chosen)
        if len(chosen) >= max_target:
            break
    if len(chosen) < max_target:
        raise ValueError(
            f"mask caps permit only {len(chosen)} slots, below max target {max_target}"
        )

    total_saliency = {
        kind: sum(float(v[1]) for (k, _, _), v in stats.items() if k == kind)
        for kind in kinds
    }
    target_report = {}
    for target in targets:
        disabled = snapshots[target]
        by_kind = {}
        for kind in kinds:
            disabled_saliency = 0.0
            disabled_seen_slots = 0
            for row in disabled:
                d = row["by_kind"][kind]
                disabled_saliency += float(d["saliency_sum"])
                disabled_seen_slots += int(d["selected"] > 0)
            denom = total_saliency[kind]
            by_kind[kind] = {
                "disabled_seen_slots": disabled_seen_slots,
                "disabled_saliency_fraction": disabled_saliency / denom if denom else 0.0,
            }
        counts = defaultdict(int)
        groups = defaultdict(int)
        for row in disabled:
            counts[int(row["layer"])] += 1
            groups[(int(row["layer"]), int(row["expert"]) // per_group)] += 1
        target_report[str(target)] = {
            "disabled_count": target,
            "zero_score_slots": sum(int(float(r["final_score"]) == 0.0) for r in disabled),
            "max_disabled_in_one_layer": max(counts.values(), default=0),
            "max_disabled_in_one_router_group": max(groups.values(), default=0),
            "by_kind": by_kind,
        }

    report = {
        "schema": "kimi-reap-vl-v1",
        "scope": "modality-aware REAP ranking and logical-mask planning; not a quality claim",
        "n_experts": n_experts,
        "n_groups": n_groups,
        "layers": sorted(layers),
        "kinds": kinds,
        "trace_rows": dict(trace_rows),
        "score": "max over modality-normalized mean(abs(router_weight)*expert_output_l2 when selected)",
        "normalization": "per-kind per-layer median of positive expert means",
        "max_disabled_per_layer": max_disabled_per_layer,
        "max_disabled_per_group": max_disabled_per_group,
        "targets": target_report,
        "experts": expert_rows,
        "prune_order": [
            {"layer": r["layer"], "expert": r["expert"], "final_score": r["final_score"]}
            for r in chosen
        ],
    }
    return report, snapshots


def write_outputs(out_dir: Path, report: dict, masks: dict[int, list[dict]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    kinds = report["kinds"]
    with (out_dir / "ranking.tsv").open("w", encoding="utf-8", newline="") as f:
        fields = ["layer", "expert", "final_score", "support_kinds", "selected_total"]
        for kind in kinds:
            fields += [f"{kind}_selected", f"{kind}_mean", f"{kind}_normalized"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in sorted(report["experts"], key=lambda r: (r["final_score"], r["layer"], r["expert"])):
            out = {
                "layer": row["layer"],
                "expert": row["expert"],
                "final_score": f'{row["final_score"]:.12g}',
                "support_kinds": row["support_kinds"],
                "selected_total": row["selected_total"],
            }
            for kind in kinds:
                d = row["by_kind"][kind]
                out[f"{kind}_selected"] = d["selected"]
                out[f"{kind}_mean"] = f'{d["saliency_mean_when_selected"]:.12g}'
                out[f"{kind}_normalized"] = f'{d["normalized_reap"]:.12g}'
            w.writerow(out)

    for target, rows in masks.items():
        path = out_dir / f"mask-reap{target}.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write("# KVL_MOE_MASK_V1\n")
            f.write(f"# REAP-VL logical mask disabled_slots={target}\n")
            f.write("# final_score is max normalized mean contribution across modalities\n")
            for row in rows:
                kind_bits = " ".join(
                    f"{k}={row['by_kind'][k]['normalized_reap']:.6g}" for k in kinds
                )
                f.write(
                    f"{row['layer']} {row['expert']} # final={row['final_score']:.9g} {kind_bits}\n"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="append", type=parse_kind_path, default=[], help="KIND=PATH; repeatable")
    ap.add_argument("--vl-trace", action="append", type=parse_vl_pair, default=[], help="TRACE=PROMPT_IDS from a max_new=1 VL run")
    ap.add_argument("--n-experts", type=int, default=64)
    ap.add_argument("--n-groups", type=int, default=8)
    ap.add_argument("--target", type=int, nargs="+", default=[56, 80, 104, 128, 152])
    ap.add_argument("--max-disabled-per-layer", type=int, default=6)
    ap.add_argument("--max-disabled-per-group", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    report, masks = build_report(
        args.trace,
        args.vl_trace,
        args.n_experts,
        args.n_groups,
        args.target,
        args.max_disabled_per_layer,
        args.max_disabled_per_group,
    )
    write_outputs(args.out_dir, report, masks)
    print(
        "KIMI_REAP_VL_COMPLETE "
        f"kinds={','.join(report['kinds'])} layers={len(report['layers'])} "
        f"targets={','.join(map(str, args.target))} out={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
