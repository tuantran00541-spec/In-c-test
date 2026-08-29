#!/usr/bin/env python3
"""Run a longer-generation Kimi functional-pruning A/B for one chosen mask.

Phase B consumes a mask already selected by Phase-A sensitivity screening. It
runs deterministic greedy full-vs-mask generation over the held-out suite using
each prompt's configured max_new. It records token exactness, first divergence,
limited baseline/candidate sanity checks, timing, and expert-cache I/O.

This remains an A/B regression experiment; passing does not prove global quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from run_kimi_pruning_phase_a import (
    DEFAULT_SUITE,
    CHAT,
    parse_generated,
    parse_runtime,
    run_checked,
    validate_suite,
)


def chat_cmd(model_dir: Path, binary: Path, prompt: str, max_new: int,
             cache_mib: int, ram_mib: int) -> list[str]:
    return [
        sys.executable, str(CHAT), str(model_dir), prompt,
        "--binary", str(binary),
        "--cache-mib", str(cache_mib),
        "--ram-mib", str(ram_mib),
        "--max-new", str(max_new),
        "--temperature", "0",
        "--seed", "1",
        "--show-tokens",
    ]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def sanity_contains(text: str, needles) -> bool | None:
    if not needles:
        return None
    hay = text.casefold()
    return all(str(x).casefold() in hay for x in needles)


def run_one(item: dict, variant: str, mask: Path | None, model_dir: Path,
            binary: Path, work: Path, cache_mib: int, ram_mib: int) -> dict:
    root = work / "heldout" / item["id"] / variant
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if mask is None:
        env.pop("KVL_MOE_MASK", None)
    else:
        env["KVL_MOE_MASK"] = str(mask)
    proc = run_checked(
        chat_cmd(
            model_dir, binary, item["prompt"], int(item["max_new"]),
            cache_mib, ram_mib,
        ),
        env=env,
        stdout_path=root / "output.txt",
        stderr_path=root / "stderr.txt",
    )
    result = {
        "variant": variant,
        "generated_ids": parse_generated(proc.stderr),
        "text": proc.stdout.strip(),
    }
    result.update(parse_runtime(proc.stderr))
    return result


def summarize(per_prompt: list[dict], candidate: str) -> dict:
    exact = 0
    divergent = []
    checked = 0
    baseline_sanity_pass = 0
    candidate_sanity_pass = 0
    sanity_preserved = 0
    full_cache_ops = cand_cache_ops = 0
    full_cache_mib = cand_cache_mib = 0.0
    full_seconds = cand_seconds = 0.0

    for p in per_prompt:
        comp = p["comparison"]
        if comp["token_exact"]:
            exact += 1
        else:
            divergent.append({
                "id": p["id"],
                "first_divergence_position": comp["first_divergence_position"],
            })
        if comp["baseline_sanity"] is not None:
            checked += 1
            baseline_sanity_pass += int(comp["baseline_sanity"])
            candidate_sanity_pass += int(comp["candidate_sanity"])
            sanity_preserved += int(comp["baseline_sanity"] and comp["candidate_sanity"])

        full = p["variants"]["full"]
        cand = p["variants"][candidate]
        full_cache_ops += int(full.get("cache", {}).get("read_ops", 0))
        cand_cache_ops += int(cand.get("cache", {}).get("read_ops", 0))
        full_cache_mib += float(full.get("cache", {}).get("bytes_read_mib", 0.0))
        cand_cache_mib += float(cand.get("cache", {}).get("bytes_read_mib", 0.0))
        full_seconds += float(full.get("timing", {}).get("total_seconds", 0.0))
        cand_seconds += float(cand.get("timing", {}).get("total_seconds", 0.0))

    return {
        "prompts": len(per_prompt),
        "token_exact_prompts": exact,
        "divergent_prompts": divergent,
        "sanity_checked_prompts": checked,
        "baseline_sanity_pass": baseline_sanity_pass,
        "candidate_sanity_pass": candidate_sanity_pass,
        "sanity_preserved_when_baseline_passed": sanity_preserved,
        "full_cache_read_ops_total": full_cache_ops,
        "candidate_cache_read_ops_total": cand_cache_ops,
        "full_cache_bytes_read_mib_total": full_cache_mib,
        "candidate_cache_bytes_read_mib_total": cand_cache_mib,
        "full_total_seconds": full_seconds,
        "candidate_total_seconds": cand_seconds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--mask", type=Path)
    ap.add_argument("--candidate", default="candidate")
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--cache-mib", type=int, default=1536)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    data = validate_suite(args.suite)
    if any(int(item.get("max_new", 0)) <= 1 for item in data["heldout"]):
        raise SystemExit("Phase B requires heldout max_new > 1")
    print(
        f"KIMI_PHASE_B_SUITE_VALID heldout={len(data['heldout'])} "
        f"candidate={args.candidate}"
    )
    if args.validate_only:
        return 0
    if args.model_dir is None or args.binary is None or args.mask is None or args.work_dir is None:
        raise SystemExit("--model-dir, --binary, --mask and --work-dir are required unless --validate-only")
    if args.cache_mib <= 0 or args.ram_mib <= 0:
        raise SystemExit("cache and RAM budgets must be positive")
    if not args.mask.is_file():
        raise SystemExit(f"mask not found: {args.mask}")

    model_dir = args.model_dir.resolve()
    binary = args.binary.resolve()
    mask = args.mask.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    per_prompt = []
    heldout = data["heldout"]
    for index, item in enumerate(heldout, 1):
        base = run_one(
            item, "full", None, model_dir, binary, work,
            args.cache_mib, args.ram_mib,
        )
        cand = run_one(
            item, args.candidate, mask, model_dir, binary, work,
            args.cache_mib, args.ram_mib,
        )
        baseline_sanity = sanity_contains(base["text"], item.get("sanity_contains"))
        candidate_sanity = sanity_contains(cand["text"], item.get("sanity_contains"))
        divergence = first_divergence(base["generated_ids"], cand["generated_ids"])
        comparison = {
            "token_exact": divergence is None,
            "first_divergence_position": divergence,
            "baseline_sanity": baseline_sanity,
            "candidate_sanity": candidate_sanity,
        }
        per_prompt.append({
            "id": item["id"],
            "category": item["category"],
            "max_new": int(item["max_new"]),
            "variants": {"full": base, args.candidate: cand},
            "comparison": comparison,
        })
        print(
            f"PHASE_B_HELDOUT {index}/{len(heldout)} id={item['id']} "
            f"candidate={args.candidate} exact={comparison['token_exact']} "
            f"first_div={divergence}"
        )

    summary = {
        "schema_version": 1,
        "scope": "longer-generation deterministic A/B regression; not a global quality claim",
        "candidate": args.candidate,
        "mask": str(mask),
        "heldout_prompts": len(heldout),
        "per_prompt": per_prompt,
        "aggregate": summarize(per_prompt, args.candidate),
    }
    out = work / "phase-b-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    a = summary["aggregate"]
    print(
        "KIMI_FUNCTIONAL_PHASE_B_COMPLETE "
        f"candidate={args.candidate} exact={a['token_exact_prompts']}/{a['prompts']} "
        f"sanity={a['candidate_sanity_pass']}/{a['sanity_checked_prompts']} "
        f"summary={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
