#!/usr/bin/env python3
"""Cheap longer-than-next-token screen for nested Kimi pruning candidates.

Runs one held-out text sentinel prompt once on the full model and once per
candidate mask, then selects the largest candidate whose generated token prefix
is byte-for-byte/token-for-token identical to the full baseline.

This is a targeted regression sentinel, not a global quality test. Candidates
that pass still require the full text/VL long-generation and physical sparse
store gates before promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_kimi_pruning_phase_a import DEFAULT_SUITE, validate_suite
from run_kimi_pruning_phase_b import first_divergence, run_one


def read_mask(path: Path) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected 'layer expert'")
        layer, expert = map(int, parts)
        key = (layer, expert)
        if not (1 <= layer <= 26 and 0 <= expert < 64) or key in out:
            raise ValueError(f"{path}:{lineno}: invalid/duplicate slot {key}")
        out.add(key)
    if not out:
        raise ValueError(f"empty mask: {path}")
    return out


def parse_candidate(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"candidate must be name=mask: {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name or not path.strip():
        raise ValueError(f"candidate must be name=mask: {spec!r}")
    return name, Path(path)


def validate_candidates(specs: list[str]) -> list[dict]:
    if not specs:
        raise ValueError("at least one candidate is required")
    rows: list[dict] = []
    names: set[str] = set()
    previous: set[tuple[int, int]] | None = None
    previous_count = -1
    for spec in specs:
        name, path = parse_candidate(spec)
        if name in names:
            raise ValueError(f"duplicate candidate name: {name}")
        names.add(name)
        if not path.is_file():
            raise ValueError(f"candidate mask not found: {path}")
        mask = read_mask(path)
        if len(mask) <= previous_count:
            raise ValueError("candidate disabled counts must be strictly increasing")
        if previous is not None and not previous < mask:
            raise ValueError("candidate masks must be strictly nested")
        rows.append({"name": name, "mask": path, "slots": mask, "disabled_count": len(mask)})
        previous = mask
        previous_count = len(mask)
    return rows


def select_strongest(rows: list[dict]) -> dict | None:
    exact = [r for r in rows if r.get("token_exact")]
    return max(exact, key=lambda r: int(r["disabled_count"])) if exact else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--sentinel-id", default="eval-vi-en-01")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--cache-mib", type=int, default=1536)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    data = validate_suite(args.suite)
    candidates = validate_candidates(args.candidate)
    try:
        source_item = next(x for x in data["heldout"] if x["id"] == args.sentinel_id)
    except StopIteration as e:
        raise SystemExit(f"sentinel id not found in heldout suite: {args.sentinel_id}") from e
    if args.max_new < 2 or args.max_new > int(source_item["max_new"]):
        raise SystemExit(
            f"--max-new must be in [2,{int(source_item['max_new'])}] for {args.sentinel_id}"
        )
    candidate_desc = ",".join(f"{c['name']}:{c['disabled_count']}" for c in candidates)
    print(
        "KIMI_SENTINEL_VALID "
        f"id={args.sentinel_id} max_new={args.max_new} candidates={candidate_desc}"
    )
    if args.validate_only:
        return 0
    required = (args.model_dir, args.binary, args.work_dir)
    if any(x is None for x in required):
        raise SystemExit("--model-dir, --binary and --work-dir required unless --validate-only")
    if args.cache_mib <= 0 or args.ram_mib <= 0:
        raise SystemExit("cache and RAM budgets must be positive")

    item = dict(source_item)
    item["max_new"] = args.max_new
    model_dir = args.model_dir.resolve()
    binary = args.binary.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    baseline = run_one(
        item, "full", None, model_dir, binary, work,
        args.cache_mib, args.ram_mib,
    )
    rows: list[dict] = []
    for candidate in candidates:
        cand = run_one(
            item, candidate["name"], candidate["mask"].resolve(),
            model_dir, binary, work, args.cache_mib, args.ram_mib,
        )
        div = first_divergence(baseline["generated_ids"], cand["generated_ids"])
        row = {
            "name": candidate["name"],
            "mask": str(candidate["mask"].resolve()),
            "disabled_count": candidate["disabled_count"],
            "token_exact": div is None,
            "first_divergence_position": div,
            "baseline_generated_ids": baseline["generated_ids"],
            "candidate_generated_ids": cand["generated_ids"],
            "candidate_runtime": {k: v for k, v in cand.items() if k in ("timing", "cache")},
        }
        rows.append(row)
        print(
            "KIMI_SENTINEL_RESULT "
            f"name={row['name']} disabled={row['disabled_count']} "
            f"exact={row['token_exact']} first_div={row['first_divergence_position']}"
        )

    selected = select_strongest(rows)
    summary = {
        "schema_version": 1,
        "scope": "single-prompt deterministic token-prefix sentinel; not a global quality claim",
        "sentinel_id": args.sentinel_id,
        "max_new": args.max_new,
        "baseline_generated_ids": baseline["generated_ids"],
        "candidates": rows,
        "selected": None if selected is None else {
            "name": selected["name"],
            "mask": selected["mask"],
            "disabled_count": selected["disabled_count"],
        },
    }
    out = work / "sentinel-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if selected is None:
        print(f"KIMI_SENTINEL_COMPLETE selected=none summary={out}")
        return 2
    print(
        f"KIMI_SENTINEL_SELECTED name={selected['name']} disabled={selected['disabled_count']}"
    )
    print(f"KIMI_SENTINEL_COMPLETE selected={selected['name']} summary={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
