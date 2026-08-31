#!/usr/bin/env python3
"""Build a domain-aware, multi-signal Kimi-VL MoE profile.

The preferred native trace is enabled with KVL_MOE_PROFILE_TRACE and has:

    event layer expert router_weight output_l2 saliency output_max_abs

Legacy six-column KVL_MOE_TRACE files remain readable, but cannot support
output-outlier profiling. This tool deliberately emits evidence and protection
orders, not a pruning mask: scalar ranking, layer-budget search, and quality
evaluation remain separate decisions.

Implemented signals:
  - routing frequency and router-weight statistics;
  - REAP = mean(abs(router_weight) * output_l2) on routed tokens;
  - MAN = mean(output_l2) on routed tokens;
  - MSAN = mean(output_l2 ** 2) on routed tokens;
  - down-projection output max-absolute tails from v2 traces;
  - within-layer routed-expert pairs and sentence/sample activation vectors;
  - independent domain rankings plus a round-robin coverage order.

The outlier rule implements the two observable magnitude conditions from the
Super Experts paper. It labels only "super-expert-like candidates" because the
paper's additional requirement that the expert occur in a massive-activation
formation layer is not observable from this trace alone.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
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
    output_max_abs: float | None = None


@dataclass
class MetricAccumulator:
    selected: int = 0
    router_weight_abs_sum: float = 0.0
    router_weight_abs_max: float = 0.0
    saliency_sum: float = 0.0
    output_l2_sum: float = 0.0
    output_l2_sq_sum: float = 0.0
    output_max_abs_values: list[float] = field(default_factory=list)

    def add(self, row: TraceRow) -> None:
        weight = abs(row.router_weight)
        self.selected += 1
        self.router_weight_abs_sum += weight
        self.router_weight_abs_max = max(self.router_weight_abs_max, weight)
        self.saliency_sum += row.saliency
        self.output_l2_sum += row.output_l2
        self.output_l2_sq_sum += row.output_l2 * row.output_l2
        if row.output_max_abs is not None:
            self.output_max_abs_values.append(row.output_max_abs)

    def finish(self, event_count: int) -> dict:
        n = self.selected
        values = self.output_max_abs_values
        return {
            "selected": n,
            "events": event_count,
            "route_frequency": n / event_count if event_count else 0.0,
            "router_weight_mean_abs": self.router_weight_abs_sum / n if n else 0.0,
            "router_weight_max_abs": self.router_weight_abs_max if n else 0.0,
            "reap": self.saliency_sum / n if n else 0.0,
            "man": self.output_l2_sum / n if n else 0.0,
            "msan": self.output_l2_sq_sum / n if n else 0.0,
            "saliency_sum": self.saliency_sum,
            "output_max_abs_observations": len(values),
            "output_max_abs_mean": sum(values) / len(values) if values else None,
            "output_max_abs_p95": percentile(values, 95.0) if values else None,
            "output_max_abs_p99": percentile(values, 99.0) if values else None,
            "output_max_abs_p99_5": percentile(values, 99.5) if values else None,
            "output_max_abs_max": max(values) if values else None,
        }


def percentile(values: list[float], q: float) -> float:
    """Deterministic linear percentile, matching NumPy's default interpolation."""
    if not values or not 0.0 <= q <= 100.0:
        raise ValueError("percentile requires values and q in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    fraction = rank - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def parse_domain_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected DOMAIN=PATH")
    domain, raw_path = value.split("=", 1)
    domain = domain.strip()
    if not domain or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty DOMAIN=PATH")
    return domain, Path(raw_path)


def parse_vl_input(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("expected DOMAIN=TRACE=PROMPT_IDS")
    return parts[0].strip(), Path(parts[1]), Path(parts[2])


def read_trace(path: Path) -> list[TraceRow]:
    rows: list[TraceRow] = []
    widths: set[int] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) not in (6, 7):
            raise ValueError(f"{path}:{lineno}: expected six or seven columns")
        widths.add(len(parts))
        event, layer, expert = map(int, parts[:3])
        weight, output_l2, saliency = map(float, parts[3:6])
        output_max_abs = float(parts[6]) if len(parts) == 7 else None
        metrics = [weight, output_l2, saliency]
        if output_max_abs is not None:
            metrics.append(output_max_abs)
        if event <= 0 or layer <= 0 or expert < 0:
            raise ValueError(f"{path}:{lineno}: invalid event/layer/expert")
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError(f"{path}:{lineno}: non-finite metric")
        if output_l2 < 0.0 or saliency < 0.0 or (
            output_max_abs is not None and output_max_abs < 0.0
        ):
            raise ValueError(f"{path}:{lineno}: negative norm/saliency/max_abs")
        rows.append(
            TraceRow(
                event,
                layer,
                expert,
                weight,
                output_l2,
                saliency,
                output_max_abs,
            )
        )
    if not rows:
        raise ValueError(f"{path}: empty trace")
    if len(widths) != 1:
        raise ValueError(f"{path}: mixed legacy/v2 row widths")
    return rows


def read_prompt_ids(path: Path) -> list[int]:
    ids: list[int] = []
    for lineno, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            token = int(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid token id {line!r}") from exc
        if token < 0:
            raise ValueError(f"{path}:{lineno}: negative token id")
        ids.append(token)
    if not ids:
        raise ValueError(f"{path}: empty prompt ids")
    return ids


def vl_event_modality(row: TraceRow, prompt_ids: list[int]) -> str:
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


def _empty_metric() -> MetricAccumulator:
    return MetricAccumulator()


def build_profile(
    text_traces: list[tuple[str, Path]],
    vl_inputs: list[tuple[str, Path, Path]],
    n_experts: int,
) -> dict:
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    if not text_traces and not vl_inputs:
        raise ValueError("at least one trace is required")

    # Each observation carries a source id because trace event counters restart
    # in every process. A VL source remains one sample even when its rows split
    # into media and VL-text modalities.
    observations: list[tuple[str, str, str, TraceRow]] = []
    trace_formats = {"legacy_v1": 0, "v2": 0}
    source_paths: dict[str, str] = {}
    source_index = 0

    for domain, path in text_traces:
        source_id = f"sample-{source_index:05d}"
        source_index += 1
        rows = read_trace(path)
        trace_formats["v2" if rows[0].output_max_abs is not None else "legacy_v1"] += 1
        source_paths[source_id] = str(path)
        for row in rows:
            observations.append((source_id, domain, "text", row))

    for domain, trace_path, ids_path in vl_inputs:
        source_id = f"sample-{source_index:05d}"
        source_index += 1
        rows = read_trace(trace_path)
        prompt_ids = read_prompt_ids(ids_path)
        trace_formats["v2" if rows[0].output_max_abs is not None else "legacy_v1"] += 1
        source_paths[source_id] = str(trace_path)
        for row in rows:
            observations.append(
                (source_id, domain, vl_event_modality(row, prompt_ids), row)
            )

    domains = sorted({domain for _, domain, _, _ in observations})
    modalities = sorted({modality for _, _, modality, _ in observations})
    layers = sorted({row.layer for _, _, _, row in observations})
    for source_id, domain, modality, row in observations:
        if row.expert >= n_experts:
            raise ValueError(
                f"{source_paths[source_id]}: expert {row.expert} >= n_experts={n_experts}"
            )

    overall_acc: dict[tuple[int, int], MetricAccumulator] = defaultdict(_empty_metric)
    domain_acc: dict[tuple[str, str, int, int], MetricAccumulator] = defaultdict(_empty_metric)
    overall_events: dict[int, set[tuple[str, int]]] = defaultdict(set)
    domain_events: dict[tuple[str, str, int], set[tuple[str, int]]] = defaultdict(set)
    event_rows: dict[tuple[str, str, str, int, int], list[TraceRow]] = defaultdict(list)
    sample_vectors: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for source_id, domain, modality, row in observations:
        event_key = (source_id, row.event)
        overall_acc[(row.layer, row.expert)].add(row)
        domain_acc[(domain, "all", row.layer, row.expert)].add(row)
        domain_acc[(domain, modality, row.layer, row.expert)].add(row)
        overall_events[row.layer].add(event_key)
        domain_events[(domain, "all", row.layer)].add(event_key)
        domain_events[(domain, modality, row.layer)].add(event_key)
        event_rows[(source_id, domain, modality, row.layer, row.event)].append(row)
        sample_vectors[(source_id, domain)][(row.layer, row.expert)] += abs(
            row.router_weight
        )

    experts: list[dict] = []
    for layer in layers:
        for expert in range(n_experts):
            aggregate = overall_acc[(layer, expert)].finish(len(overall_events[layer]))
            by_domain: dict[str, dict[str, dict]] = {}
            for domain in domains:
                domain_rows = {
                    "all": domain_acc[(domain, "all", layer, expert)].finish(
                        len(domain_events[(domain, "all", layer)])
                    )
                }
                for modality in modalities:
                    domain_rows[modality] = domain_acc[
                        (domain, modality, layer, expert)
                    ].finish(len(domain_events[(domain, modality, layer)]))
                by_domain[domain] = domain_rows
            experts.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "aggregate": aggregate,
                    "by_domain": by_domain,
                }
            )

    # Observable portion of the Super Experts criterion: compare every
    # expert's dataset maximum against P99.5 and one tenth of the global max.
    observed_maxima = [
        float(row["aggregate"]["output_max_abs_max"])
        for row in experts
        if row["aggregate"]["output_max_abs_max"] is not None
    ]
    p99_5 = percentile(observed_maxima, 99.5) if observed_maxima else None
    global_max = max(observed_maxima) if observed_maxima else None
    candidates = []
    for row in experts:
        value = row["aggregate"]["output_max_abs_max"]
        values = overall_acc[(row["layer"], row["expert"])].output_max_abs_values
        global_tail_count = (
            sum(observed > p99_5 for observed in values) if p99_5 is not None else 0
        )
        row["aggregate"]["output_max_abs_global_tail_count"] = global_tail_count
        row["aggregate"]["output_max_abs_global_tail_frequency"] = (
            global_tail_count / len(values) if values else None
        )
        is_candidate = bool(
            value is not None
            and p99_5 is not None
            and global_max is not None
            and value > p99_5
            and value > 0.1 * global_max
        )
        row["super_expert_like_candidate"] = is_candidate
        if is_candidate:
            candidates.append(
                {
                    "layer": row["layer"],
                    "expert": row["expert"],
                    "output_max_abs_max": value,
                }
            )

    pair_counts: dict[tuple[str, str, int, int, int], int] = defaultdict(int)
    for (_, domain, modality, layer, _), rows in event_rows.items():
        ids = sorted({row.expert for row in rows})
        for expert_a, expert_b in itertools.combinations(ids, 2):
            pair_counts[(domain, modality, layer, expert_a, expert_b)] += 1
    pair_rows = []
    for (domain, modality, layer, expert_a, expert_b), count in sorted(pair_counts.items()):
        denom = len(domain_events[(domain, modality, layer)])
        pair_rows.append(
            {
                "domain": domain,
                "modality": modality,
                "layer": layer,
                "expert_a": expert_a,
                "expert_b": expert_b,
                "count": count,
                "events": denom,
                "event_fraction": count / denom if denom else 0.0,
            }
        )

    vector_rows = []
    for (source_id, domain), values in sorted(sample_vectors.items()):
        vector_rows.append(
            {
                "sample_id": source_id,
                "domain": domain,
                "trace": source_paths[source_id],
                "values": [
                    {
                        "layer": layer,
                        "expert": expert,
                        "abs_router_weight_sum": value,
                    }
                    for (layer, expert), value in sorted(values.items())
                ],
            }
        )

    slot_lookup = {(row["layer"], row["expert"]): row for row in experts}
    domain_rankings: dict[str, dict[str, list[int]]] = {
        domain: {} for domain in domains
    }
    round_robin_order: dict[str, list[int]] = {}
    for layer in layers:
        for domain in domains:
            ranked = sorted(
                range(n_experts),
                key=lambda expert: (
                    -slot_lookup[(layer, expert)]["by_domain"][domain]["all"]["man"],
                    -slot_lookup[(layer, expert)]["by_domain"][domain]["all"]["msan"],
                    -slot_lookup[(layer, expert)]["by_domain"][domain]["all"]["selected"],
                    expert,
                ),
            )
            domain_rankings[domain][str(layer)] = ranked

        cursors = {domain: 0 for domain in domains}
        chosen: list[int] = []
        chosen_set: set[int] = set()
        while len(chosen) < n_experts:
            progress = False
            for domain in domains:
                ranked = domain_rankings[domain][str(layer)]
                while cursors[domain] < len(ranked):
                    expert = ranked[cursors[domain]]
                    cursors[domain] += 1
                    if expert in chosen_set:
                        continue
                    chosen.append(expert)
                    chosen_set.add(expert)
                    progress = True
                    break
            if not progress:
                break
        round_robin_order[str(layer)] = chosen

    return {
        "schema": "kimi-moe-multisignal-profile-v1",
        "scope": (
            "profiling and protection-order evidence only; not a pruning mask, "
            "quality result, or proof of Super Experts"
        ),
        "n_experts": n_experts,
        "layers": layers,
        "domains": domains,
        "modalities": modalities,
        "trace_formats": trace_formats,
        "signals": {
            "reap": "mean(abs(router_weight) * output_l2) over routed tokens",
            "man": "mean(output_l2) over routed tokens = S(1,0,1)",
            "msan": "mean(output_l2^2) over routed tokens = S(1,0,2)",
            "route_frequency": "selected routed rows / token-layer events",
        },
        "outlier_profile": {
            "source": "down_proj output max-absolute values from v2 traces",
            "observed_expert_maxima": len(observed_maxima),
            "p99_5_expert_max": p99_5,
            "global_expert_max": global_max,
            "paper_se_layer_condition_available": False,
            "classification": "candidate only; hidden-state massive-activation layer condition missing",
            "super_expert_like_candidates": candidates,
        },
        "coverage": {
            "score": "domain-specific MAN; descending",
            "domain_rankings": domain_rankings,
            "round_robin_order": round_robin_order,
            "claim_boundary": (
                "round-robin order preserves domain representation; a protection "
                "budget and final mask must be selected and evaluated separately"
            ),
        },
        "coactivation": {
            "within_layer_pairs": pair_rows,
            "sample_activation_vectors": vector_rows,
            "sample_vector_definition": (
                "sum(abs(router_weight)) per layer/expert over all tokens in one trace"
            ),
        },
        "experts": experts,
    }


def write_outputs(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metric_fields = [
        "selected",
        "events",
        "route_frequency",
        "router_weight_mean_abs",
        "router_weight_max_abs",
        "reap",
        "man",
        "msan",
        "output_max_abs_observations",
        "output_max_abs_p95",
        "output_max_abs_p99",
        "output_max_abs_p99_5",
        "output_max_abs_max",
        "output_max_abs_global_tail_count",
        "output_max_abs_global_tail_frequency",
    ]
    with (out_dir / "expert-profile.tsv").open("w", encoding="utf-8", newline="") as f:
        fields = ["layer", "expert", "super_expert_like_candidate"] + metric_fields
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in report["experts"]:
            out = {
                "layer": row["layer"],
                "expert": row["expert"],
                "super_expert_like_candidate": int(row["super_expert_like_candidate"]),
            }
            out.update({field: row["aggregate"][field] for field in metric_fields})
            writer.writerow(out)

    domain_metric_fields = [
        field for field in metric_fields if "global_tail" not in field
    ]
    with (out_dir / "domain-profile.tsv").open("w", encoding="utf-8", newline="") as f:
        fields = ["domain", "modality", "layer", "expert"] + domain_metric_fields
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in report["experts"]:
            for domain in report["domains"]:
                for modality, metrics in row["by_domain"][domain].items():
                    if not metrics["selected"]:
                        continue
                    out = {
                        "domain": domain,
                        "modality": modality,
                        "layer": row["layer"],
                        "expert": row["expert"],
                    }
                    out.update({field: metrics[field] for field in domain_metric_fields})
                    writer.writerow(out)

    pairs = report["coactivation"]["within_layer_pairs"]
    pair_fields = [
        "domain",
        "modality",
        "layer",
        "expert_a",
        "expert_b",
        "count",
        "events",
        "event_fraction",
    ]
    with (out_dir / "coactivation.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)

    (out_dir / "sample-activation-matrix.json").write_text(
        json.dumps(report["coactivation"]["sample_activation_vectors"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        type=parse_domain_path,
        default=[],
        help="DOMAIN=PATH for a text trace; repeatable",
    )
    parser.add_argument(
        "--vl-trace",
        action="append",
        type=parse_vl_input,
        default=[],
        help="DOMAIN=TRACE=PROMPT_IDS from a max_new=1 VL run; repeatable",
    )
    parser.add_argument("--n-experts", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    report = build_profile(args.trace, args.vl_trace, args.n_experts)
    write_outputs(args.out_dir, report)
    print(
        "KIMI_MOE_PROFILE_COMPLETE "
        f"domains={len(report['domains'])} modalities={len(report['modalities'])} "
        f"layers={len(report['layers'])} candidates="
        f"{len(report['outlier_profile']['super_expert_like_candidates'])} "
        f"out={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
