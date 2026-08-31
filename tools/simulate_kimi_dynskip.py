#!/usr/bin/env python3
"""Project Kimi-VL dynamic-expert skips from captured top-k routing traces.

The simulator mirrors the C pilot semantics:
- classify exact prompt tokens as control/content/media;
- use the original already-selected top-k route weights;
- threshold normalized top-k mass per token family + layer;
- enforce min_keep;
- do not reroute and do not renormalize survivors.

This is a routing-work projection only. It does not predict logit quality, cache bytes,
or wall-clock speed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CONTROL = 0
CONTENT = 1
MEDIA = 2
FAMILY_NAME = {CONTROL: "control", CONTENT: "content", MEDIA: "media"}

IM_END = 163586
IM_USER = 163587
IM_ASSISTANT = 163588
IM_SYSTEM = 163594
IM_MIDDLE = 163601
MEDIA_START = 163602
MEDIA_END = 163604
MEDIA_PAD = 163605


def classify_prompt(ids: list[int]) -> list[int]:
    if not ids:
        raise ValueError("empty prompt ids")
    OUTSIDE, SYSTEM, USER_HEADER, USER_BODY, MEDIA_STATE, ASSISTANT = range(6)
    state = OUTSIDE
    out: list[int] = []
    for token in ids:
        family = CONTROL
        if token == IM_SYSTEM:
            state = SYSTEM
        elif state == SYSTEM:
            if token == IM_END:
                state = OUTSIDE
        elif token == IM_USER:
            state = USER_HEADER
        elif state == USER_HEADER:
            if token == IM_MIDDLE:
                state = USER_BODY
        elif state == MEDIA_STATE:
            if token == MEDIA_PAD:
                family = MEDIA
            if token == MEDIA_END:
                state = USER_BODY
        elif state == USER_BODY:
            if token == IM_END:
                state = OUTSIDE
            elif token == MEDIA_START:
                state = MEDIA_STATE
            elif token == MEDIA_PAD:
                family = MEDIA
            else:
                family = CONTENT
        elif token == IM_ASSISTANT:
            state = ASSISTANT
        elif state == ASSISTANT:
            family = CONTROL
        elif token == MEDIA_PAD:
            family = MEDIA
        out.append(family)
    return out


def load_policy(path: Path) -> dict[tuple[int, int], tuple[float, int]]:
    mapping = {"content": CONTENT, "media": MEDIA}
    out: dict[tuple[int, int], tuple[float, int]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4 or parts[0] not in mapping:
            raise ValueError(f"{path}:{lineno}: expected 'content|media L threshold min_keep'")
        family = mapping[parts[0]]
        layer = int(parts[1])
        threshold = float(parts[2])
        min_keep = int(parts[3])
        if layer < 1 or not 0.0 <= threshold <= 1.0 or min_keep < 1:
            raise ValueError(f"{path}:{lineno}: invalid policy values")
        out[(family, layer)] = (threshold, min_keep)
    if not out or not any(thr > 0.0 for thr, _ in out.values()):
        raise ValueError("policy has no active threshold")
    return out


def resolve_artifact_path(root: Path, recorded: str) -> Path:
    marker = "/profile/"
    if marker in recorded:
        return root / recorded.split(marker, 1)[1]
    p = Path(recorded)
    return p if p.is_absolute() else root / p


def apply_policy(weights: list[float], threshold: float, min_keep: int) -> int:
    if threshold <= 0.0:
        return 0
    if not weights or any(x < 0.0 for x in weights):
        raise ValueError("invalid top-k weights")
    total = sum(weights)
    if total <= 0.0:
        return 0
    mass = [x / total for x in weights]
    keep = [x >= threshold for x in mass]
    target = min(max(min_keep, 1), len(weights))
    while sum(keep) < target:
        candidates = [i for i, flag in enumerate(keep) if not flag]
        if not candidates:
            break
        best = max(candidates, key=lambda i: mass[i])
        keep[best] = True
    return len(weights) - sum(keep)


def parse_trace(path: Path) -> dict[tuple[int, int], list[float]]:
    events: dict[tuple[int, int], list[float]] = defaultdict(list)
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}:{lineno}: malformed trace row")
        event, layer = int(parts[0]), int(parts[1])
        weight = float(parts[3])
        events[(event, layer)].append(weight)
    if not events:
        raise ValueError(f"{path}: no trace events")
    return events


def simulate(manifest_path: Path, artifact_root: Path, policy_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = load_policy(policy_path)
    stats: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: {"events": 0, "routed": 0, "skipped": 0})
    samples = manifest.get("text_samples", []) + manifest.get("vl_samples", [])
    for sample in samples:
        ids_path = resolve_artifact_path(artifact_root, sample["prompt_ids"])
        trace_path = resolve_artifact_path(artifact_root, sample["trace"])
        ids = [int(x) for x in ids_path.read_text(encoding="ascii").split()]
        families = classify_prompt(ids)
        for (event, layer), weights in parse_trace(trace_path).items():
            expected_layer = (event - 1) // len(ids) + 1
            if expected_layer != layer:
                raise ValueError(
                    f"{trace_path}: event {event} says layer {layer}, expected {expected_layer}"
                )
            pos = (event - 1) % len(ids)
            family = families[pos]
            threshold, min_keep = policy.get((family, layer), (0.0, len(weights)))
            skipped = apply_policy(weights, threshold, min_keep)
            slot = stats[(family, layer)]
            slot["events"] += 1
            slot["routed"] += len(weights)
            slot["skipped"] += skipped

    rows = []
    total_routed = 0
    total_skipped = 0
    for (family, layer), values in sorted(stats.items()):
        threshold, min_keep = policy.get((family, layer), (0.0, 0))
        routed = values["routed"]
        skipped = values["skipped"]
        total_routed += routed
        total_skipped += skipped
        rows.append({
            "family": FAMILY_NAME[family],
            "layer": layer,
            **values,
            "skip_fraction": skipped / routed if routed else 0.0,
            "threshold": threshold,
            "min_keep": min_keep,
        })
    return {
        "schema": "kimi-dynskip-trace-projection-v1",
        "manifest": str(manifest_path),
        "policy": str(policy_path),
        "samples": len(samples),
        "routed": total_routed,
        "skipped": total_skipped,
        "skip_fraction": total_skipped / total_routed if total_routed else 0.0,
        "rows": rows,
        "claim_boundary": (
            "Routing-work projection from captured top-k traces only; no logit quality, "
            "cache-byte, TTFT, or wall-clock speed claim."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = simulate(args.manifest, args.artifact_root, args.policy)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    print(
        "KIMI_DYNSKIP_TRACE_PROJECTION "
        f"samples={report['samples']} routed={report['routed']} "
        f"skipped={report['skipped']} skip_fraction={report['skip_fraction']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
