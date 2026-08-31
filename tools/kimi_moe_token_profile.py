#!/usr/bin/env python3
"""Token-aware companion analysis for Kimi-VL MoE traces.

This tool intentionally leaves analyze_kimi_moe_profile.py and its v1 schema
unchanged. It partitions exact prompt-token events into content, control, media,
or unclassified families, then recomputes the same core MoE signals per scope.
It emits evidence only; it never selects a pruning mask.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from analyze_kimi_moe_profile import MetricAccumulator, percentile, read_prompt_ids, read_trace

IM_END_ID = 163586
IM_USER_ID = 163587
IM_ASSISTANT_ID = 163588
IM_SYSTEM_ID = 163594
IM_MIDDLE_ID = 163601
MEDIA_START_ID = 163602
MEDIA_CONTENT_ID = 163603
MEDIA_END_ID = 163604
MEDIA_PAD_ID = 163605

TOKEN_FAMILY = {
    "system": "control",
    "user_control": "control",
    "user_content": "content",
    "media_control": "control",
    "media_pad": "media",
    "assistant_transition": "control",
    "unclassified": "unclassified",
}


def parse_input(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or any(not p.strip() for p in parts):
        raise argparse.ArgumentTypeError("expected DOMAIN=TRACE=PROMPT_IDS")
    return parts[0].strip(), Path(parts[1]), Path(parts[2])


def classify_prompt_tokens(prompt_ids: list[int]) -> list[str]:
    """Classify released Kimi-VL chat-template tokens without decoding base IDs."""
    if not prompt_ids:
        raise ValueError("prompt ids must not be empty")
    out: list[str] = []
    state = "outside"
    for token in prompt_ids:
        if token == IM_SYSTEM_ID:
            state = "system"
            out.append("system")
            continue
        if state == "system":
            out.append("system")
            if token == IM_END_ID:
                state = "outside"
            continue
        if token == IM_USER_ID:
            state = "user_header"
            out.append("user_control")
            continue
        if state == "user_header":
            out.append("user_control")
            if token == IM_MIDDLE_ID:
                state = "user_body"
            continue
        if state == "media":
            out.append("media_pad" if token == MEDIA_PAD_ID else "media_control")
            if token == MEDIA_END_ID:
                state = "user_body"
            continue
        if state == "user_body":
            if token == IM_END_ID:
                out.append("user_control")
                state = "outside"
            elif token == MEDIA_START_ID:
                out.append("media_control")
                state = "media"
            elif token == MEDIA_PAD_ID:
                out.append("media_pad")
            else:
                out.append("user_content")
            continue
        if token == IM_ASSISTANT_ID:
            state = "assistant"
            out.append("assistant_transition")
            continue
        if state == "assistant":
            out.append("assistant_transition")
            continue
        out.append("media_pad" if token == MEDIA_PAD_ID else "unclassified")
    return out


def event_position(event: int, layer: int, prompt_tokens: int) -> int:
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    zero = event - 1
    expected_layer = zero // prompt_tokens + 1
    if expected_layer != layer:
        raise ValueError(
            f"trace ordering mismatch event={event} layer={layer} "
            f"expected_layer={expected_layer} prompt_tokens={prompt_tokens}"
        )
    return zero % prompt_tokens


def _finish(acc: MetricAccumulator, events: int) -> dict:
    result = acc.finish(events)
    result["output_max_abs_global_tail_count"] = 0
    result["output_max_abs_global_tail_frequency"] = None
    return result


def _outlier_profile(rows: list[dict], accs: dict[tuple[int, int], MetricAccumulator]) -> dict:
    maxima = [r["metrics"]["output_max_abs_max"] for r in rows if r["metrics"]["output_max_abs_max"] is not None]
    p99_5 = percentile(maxima, 99.5) if maxima else None
    global_max = max(maxima) if maxima else None
    candidates = []
    for r in rows:
        m = r["metrics"]
        values = accs[(r["layer"], r["expert"])].output_max_abs_values
        tail = sum(v > p99_5 for v in values) if p99_5 is not None else 0
        m["output_max_abs_global_tail_count"] = tail
        m["output_max_abs_global_tail_frequency"] = tail / len(values) if values else None
        value = m["output_max_abs_max"]
        if value is not None and p99_5 is not None and global_max is not None and value > p99_5 and value > 0.1 * global_max:
            candidates.append({"layer": r["layer"], "expert": r["expert"], "output_max_abs_max": value})
    return {
        "source": "down_proj output max-absolute values from v2 traces",
        "p99_5_expert_max": p99_5,
        "global_expert_max": global_max,
        "paper_se_layer_condition_available": False,
        "classification": "candidate only within this token family; no pruning/quality claim",
        "super_expert_like_candidates": candidates,
    }


def build_token_profile(inputs: list[tuple[str, Path, Path]], n_experts: int = 64) -> dict:
    if not inputs:
        raise ValueError("at least one token-aware trace is required")
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")

    observations = []
    trace_formats = {"legacy_v1": 0, "v2": 0}
    for sample_idx, (domain, trace_path, ids_path) in enumerate(inputs):
        rows = read_trace(trace_path)
        ids = read_prompt_ids(ids_path)
        classes = classify_prompt_tokens(ids)
        trace_formats["v2" if rows[0].output_max_abs is not None else "legacy_v1"] += 1
        sample = f"sample-{sample_idx:05d}"
        for row in rows:
            if row.expert >= n_experts:
                raise ValueError(f"{trace_path}: expert {row.expert} >= n_experts={n_experts}")
            pos = event_position(row.event, row.layer, len(ids))
            token_class = classes[pos]
            observations.append((sample, domain, token_class, TOKEN_FAMILY[token_class], row))

    domains = sorted({x[1] for x in observations})
    classes = sorted({x[2] for x in observations})
    families = sorted({x[3] for x in observations})
    layers = sorted({x[4].layer for x in observations})

    class_acc = defaultdict(MetricAccumulator)
    family_acc = defaultdict(MetricAccumulator)
    domain_family_acc = defaultdict(MetricAccumulator)
    class_events = defaultdict(set)
    family_events = defaultdict(set)
    domain_family_events = defaultdict(set)
    event_rows = defaultdict(list)

    for sample, domain, token_class, family, row in observations:
        key = (sample, row.event)
        class_acc[(token_class, row.layer, row.expert)].add(row)
        family_acc[(family, row.layer, row.expert)].add(row)
        domain_family_acc[(domain, family, row.layer, row.expert)].add(row)
        class_events[(token_class, row.layer)].add(key)
        family_events[(family, row.layer)].add(key)
        domain_family_events[(domain, family, row.layer)].add(key)
        event_rows[(sample, domain, family, row.layer, row.event)].append(row)

    by_family: dict[str, list[dict]] = {}
    outliers = {}
    for family in families:
        rows = []
        acc_lookup = {}
        for layer in layers:
            for expert in range(n_experts):
                acc = family_acc[(family, layer, expert)]
                acc_lookup[(layer, expert)] = acc
                rows.append({
                    "layer": layer,
                    "expert": expert,
                    "metrics": _finish(acc, len(family_events[(family, layer)])),
                })
        outliers[family] = _outlier_profile(rows, acc_lookup)
        by_family[family] = rows

    by_class = []
    for token_class in classes:
        for layer in layers:
            for expert in range(n_experts):
                acc = class_acc[(token_class, layer, expert)]
                m = acc.finish(len(class_events[(token_class, layer)]))
                if m["selected"]:
                    by_class.append({"token_class": token_class, "layer": layer, "expert": expert, "metrics": m})

    by_domain_family = []
    for domain in domains:
        for family in families:
            for layer in layers:
                for expert in range(n_experts):
                    acc = domain_family_acc[(domain, family, layer, expert)]
                    m = acc.finish(len(domain_family_events[(domain, family, layer)]))
                    if m["selected"]:
                        by_domain_family.append({"domain": domain, "token_family": family, "layer": layer, "expert": expert, "metrics": m})

    pair_counts = defaultdict(int)
    for (_, domain, family, layer, _), rows in event_rows.items():
        ids = sorted({r.expert for r in rows})
        for a, b in itertools.combinations(ids, 2):
            pair_counts[(domain, family, layer, a, b)] += 1
    pairs = []
    for (domain, family, layer, a, b), count in sorted(pair_counts.items()):
        denom = len(domain_family_events[(domain, family, layer)])
        pairs.append({
            "domain": domain,
            "token_family": family,
            "layer": layer,
            "expert_a": a,
            "expert_b": b,
            "count": count,
            "events": denom,
            "event_fraction": count / denom if denom else 0.0,
        })

    return {
        "schema": "kimi-moe-token-aware-profile-v1",
        "scope": "token-family profiling evidence only; no mask, quality, speed, or Super-Expert proof",
        "n_experts": n_experts,
        "layers": layers,
        "domains": domains,
        "token_classes": classes,
        "token_families": families,
        "trace_formats": trace_formats,
        "token_partition": {
            "system": "entire system span including role/control markers",
            "user_control": "user role header and message boundary",
            "user_content": "user natural-language content outside media wrapper",
            "media_control": "media wrapper and image label",
            "media_pad": "projected visual-token placeholders",
            "assistant_transition": "generation hand-off span",
            "families": TOKEN_FAMILY,
        },
        "outlier_profiles": outliers,
        "by_family": by_family,
        "by_class": by_class,
        "by_domain_family": by_domain_family,
        "coactivation": pairs,
    }


METRIC_FIELDS = [
    "selected", "events", "route_frequency", "router_weight_mean_abs",
    "router_weight_max_abs", "reap", "man", "msan", "saliency_sum",
    "output_max_abs_observations", "output_max_abs_mean", "output_max_abs_p95",
    "output_max_abs_p99", "output_max_abs_p99_5", "output_max_abs_max",
]


def write_outputs(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "token-aware-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (out_dir / "token-family-profile.tsv").open("w", encoding="utf-8", newline="") as f:
        fields = ["token_family", "layer", "expert"] + METRIC_FIELDS + ["output_max_abs_global_tail_count", "output_max_abs_global_tail_frequency"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n"); w.writeheader()
        for family, rows in report["by_family"].items():
            for row in rows:
                out = {"token_family": family, "layer": row["layer"], "expert": row["expert"]}; out.update(row["metrics"]); w.writerow(out)

    with (out_dir / "token-class-profile.tsv").open("w", encoding="utf-8", newline="") as f:
        fields = ["token_class", "layer", "expert"] + METRIC_FIELDS
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n"); w.writeheader()
        for row in report["by_class"]:
            out = {k: row[k] for k in ("token_class", "layer", "expert")}; out.update({k: row["metrics"][k] for k in METRIC_FIELDS}); w.writerow(out)

    with (out_dir / "domain-token-family-profile.tsv").open("w", encoding="utf-8", newline="") as f:
        fields = ["domain", "token_family", "layer", "expert"] + METRIC_FIELDS
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n"); w.writeheader()
        for row in report["by_domain_family"]:
            out = {k: row[k] for k in ("domain", "token_family", "layer", "expert")}; out.update({k: row["metrics"][k] for k in METRIC_FIELDS}); w.writerow(out)

    with (out_dir / "token-family-coactivation.tsv").open("w", encoding="utf-8", newline="") as f:
        fields = ["domain", "token_family", "layer", "expert_a", "expert_b", "count", "events", "event_fraction"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(report["coactivation"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", action="append", type=parse_input, default=[], help="DOMAIN=TRACE=PROMPT_IDS; repeatable")
    ap.add_argument("--vl", action="append", type=parse_input, default=[], help="DOMAIN=TRACE=PROMPT_IDS; repeatable")
    ap.add_argument("--n-experts", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    report = build_token_profile(args.text + args.vl, args.n_experts)
    write_outputs(args.out_dir, report)
    counts = {k: len(v["super_expert_like_candidates"]) for k, v in report["outlier_profiles"].items()}
    print(f"KIMI_MOE_TOKEN_PROFILE_COMPLETE families={','.join(report['token_families'])} candidates={counts} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
