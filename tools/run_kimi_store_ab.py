#!/usr/bin/env python3
"""Deterministic text A/B between two packed Kimi runtime stores.

Designed for quantization experiments where baseline and candidate have separate
expert stores but share the same pinned trunk/tokenizer. Records token drift,
semantic smoke checks already present in the pruning suite, timing, and cache I/O.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from run_kimi_pruning_phase_a import DEFAULT_SUITE, CHAT, parse_generated, parse_runtime, run_checked, validate_suite


def chat_cmd(model_dir: Path, binary: Path, prompt: str, max_new: int,
             cache_mib: int, ram_mib: int) -> list[str]:
    return [
        sys.executable, str(CHAT), str(model_dir), prompt,
        "--binary", str(binary),
        "--cache-mib", str(cache_mib), "--ram-mib", str(ram_mib),
        "--max-new", str(max_new), "--temperature", "0", "--seed", "1",
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


def run_one(item: dict, variant: str, model_dir: Path, binary: Path,
            work: Path, cache_mib: int, ram_mib: int) -> dict:
    root = work / "heldout" / item["id"] / variant
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("KVL_MOE_MASK", None)
    env.pop("KVL_MOE_TRACE", None)
    env.pop("KVL_LOGITS_DUMP", None)
    env.pop("KVL_LOGITS_DUMP_LIMIT", None)
    proc = run_checked(
        chat_cmd(model_dir, binary, item["prompt"], int(item["max_new"]), cache_mib, ram_mib),
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


def summarize(rows: list[dict]) -> dict:
    exact = 0
    divergent = []
    checked = baseline_sanity = candidate_sanity = preserved = 0
    base_cache_ops = cand_cache_ops = 0
    base_cache_mib = cand_cache_mib = 0.0
    base_seconds = cand_seconds = 0.0
    for row in rows:
        comp = row["comparison"]
        if comp["token_exact"]:
            exact += 1
        else:
            divergent.append({"id": row["id"], "first_divergence_position": comp["first_divergence_position"]})
        if comp["baseline_sanity"] is not None:
            checked += 1
            baseline_sanity += int(comp["baseline_sanity"])
            candidate_sanity += int(comp["candidate_sanity"])
            preserved += int(comp["baseline_sanity"] and comp["candidate_sanity"])
        base = row["baseline"]
        cand = row["candidate"]
        base_cache_ops += int(base.get("cache", {}).get("read_ops", 0))
        cand_cache_ops += int(cand.get("cache", {}).get("read_ops", 0))
        base_cache_mib += float(base.get("cache", {}).get("bytes_read_mib", 0.0))
        cand_cache_mib += float(cand.get("cache", {}).get("bytes_read_mib", 0.0))
        base_seconds += float(base.get("timing", {}).get("total_seconds", 0.0))
        cand_seconds += float(cand.get("timing", {}).get("total_seconds", 0.0))
    return {
        "prompts": len(rows),
        "token_exact_prompts": exact,
        "divergent_prompts": divergent,
        "sanity_checked_prompts": checked,
        "baseline_sanity_pass": baseline_sanity,
        "candidate_sanity_pass": candidate_sanity,
        "sanity_preserved_when_baseline_passed": preserved,
        "baseline_cache_read_ops_total": base_cache_ops,
        "candidate_cache_read_ops_total": cand_cache_ops,
        "baseline_cache_bytes_read_mib_total": base_cache_mib,
        "candidate_cache_bytes_read_mib_total": cand_cache_mib,
        "baseline_total_seconds": base_seconds,
        "candidate_total_seconds": cand_seconds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--baseline-model-dir", type=Path)
    ap.add_argument("--candidate-model-dir", type=Path)
    ap.add_argument("--baseline-binary", type=Path)
    ap.add_argument("--candidate-binary", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--cache-mib", type=int, default=1536)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    data = validate_suite(args.suite)
    if any(int(item.get("max_new", 0)) <= 1 for item in data["heldout"]):
        raise SystemExit("store A/B requires heldout max_new > 1")
    print(f"KIMI_STORE_AB_SUITE_VALID heldout={len(data['heldout'])}")
    if args.validate_only:
        return 0
    required = [args.baseline_model_dir, args.candidate_model_dir, args.baseline_binary, args.candidate_binary, args.work_dir]
    if any(x is None for x in required):
        raise SystemExit("baseline/candidate model dirs, binaries, and work-dir are required")

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(data["heldout"], 1):
        base = run_one(item, "baseline-q8", args.baseline_model_dir.resolve(), args.baseline_binary.resolve(), work, args.cache_mib, args.ram_mib)
        cand = run_one(item, "candidate-q5", args.candidate_model_dir.resolve(), args.candidate_binary.resolve(), work, args.cache_mib, args.ram_mib)
        bs = sanity_contains(base["text"], item.get("sanity_contains"))
        cs = sanity_contains(cand["text"], item.get("sanity_contains"))
        div = first_divergence(base["generated_ids"], cand["generated_ids"])
        comparison = {
            "token_exact": div is None,
            "first_divergence_position": div,
            "baseline_sanity": bs,
            "candidate_sanity": cs,
        }
        rows.append({
            "id": item["id"], "category": item["category"], "max_new": int(item["max_new"]),
            "baseline": base, "candidate": cand, "comparison": comparison,
        })
        print(f"STORE_AB {index}/{len(data['heldout'])} id={item['id']} exact={div is None} first_div={div} sanity={cs}", flush=True)

    summary = {
        "schema_version": 1,
        "scope": "deterministic packed-store A/B regression; token parity plus limited known-answer smoke checks, not a global quality claim",
        "baseline": "Q8 routed experts",
        "candidate": "Q5 group128 FP32-scale routed experts",
        "per_prompt": rows,
        "aggregate": summarize(rows),
    }
    out = work / "store-ab-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    a = summary["aggregate"]
    print(f"KIMI_STORE_AB_COMPLETE exact={a['token_exact_prompts']}/{a['prompts']} sanity={a['candidate_sanity_pass']}/{a['sanity_checked_prompts']} summary={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
