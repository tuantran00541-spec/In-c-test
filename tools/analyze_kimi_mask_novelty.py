#!/usr/bin/env python3
"""Measure how a calibration-derived Kimi MoE mask behaves on held-out routes.

Given a Phase-A evidence directory, this distinguishes:
  * direct masked selections: baseline routes that actually selected a masked expert;
  * cascade substitutions: additional full-vs-mask route substitutions after the
    hidden state changed because of earlier masking.

A calibration-unseen expert is therefore not called "dead" merely because its
calibration count was zero.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def read_mask(path: Path) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected layer expert")
        key = tuple(map(int, parts))
        if key[0] < 0 or key[1] < 0:
            raise ValueError(f"{path}:{lineno}: negative layer/expert")
        if key in out:
            raise ValueError(f"{path}:{lineno}: duplicate {key}")
        out.add(key)
    if not out:
        raise ValueError(f"{path}: empty mask")
    return out


def read_direct_hits(trace: Path, mask: set[tuple[int, int]]):
    hits = Counter()
    saliency = Counter()
    for lineno, raw in enumerate(trace.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{trace}:{lineno}: expected 6 columns")
        _, layer, expert = map(int, parts[:3])
        sal = float(parts[5])
        if not math.isfinite(sal) or sal < 0.0:
            raise ValueError(f"{trace}:{lineno}: invalid saliency")
        key = (layer, expert)
        if key in mask:
            hits[key] += 1
            saliency[key] += sal
    return hits, saliency


def analyze(evidence: Path, mask_path: Path, variant: str) -> dict:
    mask = read_mask(mask_path)
    heldout = evidence / "heldout"
    comparisons = evidence / "comparisons"
    if not heldout.is_dir() or not comparisons.is_dir():
        raise ValueError(f"{evidence}: missing heldout/comparisons directories")

    slot_hits = Counter()
    slot_saliency = Counter()
    per_prompt = []
    for pdir in sorted(p for p in heldout.iterdir() if p.is_dir()):
        trace = pdir / "full" / "route.tsv"
        route_json = comparisons / pdir.name / variant / "route.json"
        if not trace.is_file() or not route_json.is_file():
            raise ValueError(f"{pdir.name}: missing full route or {variant} comparison")
        hits, saliency = read_direct_hits(trace, mask)
        slot_hits.update(hits)
        slot_saliency.update(saliency)
        route = json.loads(route_json.read_text(encoding="utf-8"))
        substitutions = int(route["summary"]["substitutions"])
        direct = sum(hits.values())
        cascade = substitutions - direct
        if cascade < 0:
            raise ValueError(
                f"{pdir.name}: substitutions={substitutions} < direct masked selections={direct}"
            )
        per_prompt.append({
            "id": pdir.name,
            "direct_masked_selections": direct,
            "direct_masked_unique_slots": len(hits),
            "direct_masked_saliency": sum(saliency.values()),
            "route_substitutions": substitutions,
            "cascade_substitutions": cascade,
            "direct_slots": [
                {"layer": l, "expert": e, "count": hits[(l, e)],
                 "saliency": saliency[(l, e)]}
                for l, e in sorted(hits)
            ],
        })

    slot_rows = [
        {
            "layer": l,
            "expert": e,
            "heldout_selection_count": slot_hits[(l, e)],
            "heldout_saliency": slot_saliency[(l, e)],
        }
        for l, e in sorted(mask)
    ]
    direct_total = sum(slot_hits.values())
    substitutions_total = sum(p["route_substitutions"] for p in per_prompt)
    return {
        "schema_version": 1,
        "variant": variant,
        "mask_entries": len(mask),
        "prompts": len(per_prompt),
        "aggregate": {
            "direct_masked_selections": direct_total,
            "unique_masked_slots_hit": sum(int(r["heldout_selection_count"] > 0) for r in slot_rows),
            "masked_slots_never_hit": sum(int(r["heldout_selection_count"] == 0) for r in slot_rows),
            "prompts_with_direct_mask_hit": sum(int(p["direct_masked_selections"] > 0) for p in per_prompt),
            "route_substitutions": substitutions_total,
            "cascade_substitutions": substitutions_total - direct_total,
        },
        "per_prompt": per_prompt,
        "mask_slots": slot_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence", type=Path)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--variant", default="adaptive-unseen-cap6")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = analyze(args.evidence, args.mask, args.variant)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    a = report["aggregate"]
    print(
        "KIMI_MASK_NOVELTY_ANALYSIS_PASS "
        f"mask={report['mask_entries']} hit_slots={a['unique_masked_slots_hit']} "
        f"direct={a['direct_masked_selections']} substitutions={a['route_substitutions']} "
        f"cascade={a['cascade_substitutions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
