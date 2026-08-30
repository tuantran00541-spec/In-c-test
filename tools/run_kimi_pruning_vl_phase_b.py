#!/usr/bin/env python3
"""Run deterministic long-generation multimodal full-vs-mask A/B for Kimi-VL.

This stage is intended only after a candidate passes the next-token multimodal
route/logit guard. It reruns the same image+text cases with each case's configured
max_new and compares the complete generated token sequence. It records outputs,
first divergence positions, timing and expert-cache I/O.

Exact agreement here is a regression result against the full-store baseline, not
a global multimodal quality guarantee and not by itself a storage-pruning proof.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from run_kimi_pruning_phase_a import ROOT, run_checked
from run_kimi_pruning_vl_guard import (
    DEFAULT_SUITE,
    VL_CHAT,
    parse_generated,
    parse_vl_runtime,
    validate_suite as validate_guard_suite,
)


def validate_suite(path: Path) -> dict:
    data = validate_guard_suite(path)
    for item in data["cases"]:
        max_new = item.get("max_new")
        if not isinstance(max_new, int) or max_new <= 1:
            raise ValueError(f"{item['id']}: VL long-generation max_new must be integer > 1")
    return data


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def vl_cmd(model: Path, image: Path, prompt: str, vision_binary: Path,
           generate_binary: Path, cache_mib: int, ram_mib: int,
           max_new: int) -> list[str]:
    return [
        sys.executable, str(VL_CHAT), str(model), str(image), prompt,
        "--vision-binary", str(vision_binary),
        "--generate-binary", str(generate_binary),
        "--cache-mib", str(cache_mib), "--ram-mib", str(ram_mib),
        "--max-new", str(max_new), "--temperature", "0", "--seed", "1",
        "--show-tokens",
    ]


def run_one(item: dict, variant: str, mask: Path | None, model: Path,
            image_root: Path, vision_binary: Path, generate_binary: Path,
            work: Path, cache_mib: int, ram_mib: int) -> dict:
    root = work / "cases" / item["id"] / variant
    root.mkdir(parents=True, exist_ok=True)
    image = image_root / item["image"]
    if not image.is_file():
        raise RuntimeError(f"missing VL image: {image}")
    env = os.environ.copy()
    env.pop("KVL_MOE_TRACE", None)
    env.pop("KVL_LOGITS_DUMP", None)
    env.pop("KVL_LOGITS_DUMP_LIMIT", None)
    if mask is None:
        env.pop("KVL_MOE_MASK", None)
    else:
        env["KVL_MOE_MASK"] = str(mask)
    proc = run_checked(
        vl_cmd(
            model, image, item["prompt"], vision_binary, generate_binary,
            cache_mib, ram_mib, int(item["max_new"]),
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
    result.update(parse_vl_runtime(proc.stderr))
    return result


def summarize(rows: list[dict]) -> dict:
    exact = 0
    divergent = []
    full_cache_ops = cand_cache_ops = 0
    full_cache_mib = cand_cache_mib = 0.0
    full_text_seconds = cand_text_seconds = 0.0
    full_vision_seconds = cand_vision_seconds = 0.0
    full_generated = cand_generated = 0

    for row in rows:
        comp = row["comparison"]
        if comp["token_exact"]:
            exact += 1
        else:
            divergent.append({
                "id": row["id"],
                "first_divergence_position": comp["first_divergence_position"],
            })
        full = row["full"]
        cand = row["candidate"]
        full_cache_ops += int(full.get("cache", {}).get("read_ops", 0))
        cand_cache_ops += int(cand.get("cache", {}).get("read_ops", 0))
        full_cache_mib += float(full.get("cache", {}).get("bytes_read_mib", 0.0))
        cand_cache_mib += float(cand.get("cache", {}).get("bytes_read_mib", 0.0))
        full_text_seconds += float(full.get("vl_timing", {}).get("text_total_seconds", 0.0))
        cand_text_seconds += float(cand.get("vl_timing", {}).get("text_total_seconds", 0.0))
        full_vision_seconds += float(full.get("vl_timing", {}).get("vision_seconds", 0.0))
        cand_vision_seconds += float(cand.get("vl_timing", {}).get("vision_seconds", 0.0))
        full_generated += len(full["generated_ids"])
        cand_generated += len(cand["generated_ids"])

    return {
        "cases": len(rows),
        "token_exact_cases": exact,
        "divergent_cases": divergent,
        "full_generated_tokens_total": full_generated,
        "candidate_generated_tokens_total": cand_generated,
        "full_cache_read_ops_total": full_cache_ops,
        "candidate_cache_read_ops_total": cand_cache_ops,
        "full_cache_bytes_read_mib_total": full_cache_mib,
        "candidate_cache_bytes_read_mib_total": cand_cache_mib,
        "full_text_total_seconds": full_text_seconds,
        "candidate_text_total_seconds": cand_text_seconds,
        "full_vision_total_seconds": full_vision_seconds,
        "candidate_vision_total_seconds": cand_vision_seconds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--image-root", type=Path)
    ap.add_argument("--vision-binary", type=Path)
    ap.add_argument("--generate-binary", type=Path)
    ap.add_argument("--mask", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    data = validate_suite(args.suite)
    print(
        f"KIMI_VL_PHASE_B_SUITE_VALID cases={len(data['cases'])} "
        f"max_new_total={sum(int(x['max_new']) for x in data['cases'])}"
    )
    if args.validate_only:
        return 0
    required = (
        args.model_dir, args.image_root, args.vision_binary,
        args.generate_binary, args.mask, args.work_dir,
    )
    if any(x is None for x in required):
        raise SystemExit("model/image/binaries/mask/work-dir required unless --validate-only")
    if args.cache_mib <= 0 or args.ram_mib <= 0:
        raise SystemExit("cache and RAM budgets must be positive")
    if not args.mask.is_file():
        raise SystemExit(f"mask not found: {args.mask}")

    model = args.model_dir.resolve()
    image_root = args.image_root.resolve()
    vision_binary = args.vision_binary.resolve()
    generate_binary = args.generate_binary.resolve()
    mask = args.mask.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, item in enumerate(data["cases"], 1):
        full = run_one(
            item, "full", None, model, image_root, vision_binary, generate_binary,
            work, args.cache_mib, args.ram_mib,
        )
        candidate = run_one(
            item, "candidate", mask, model, image_root, vision_binary, generate_binary,
            work, args.cache_mib, args.ram_mib,
        )
        div = first_divergence(full["generated_ids"], candidate["generated_ids"])
        comparison = {
            "token_exact": div is None,
            "first_divergence_position": div,
        }
        rows.append({
            "id": item["id"],
            "image": item["image"],
            "prompt": item["prompt"],
            "max_new": int(item["max_new"]),
            "full": full,
            "candidate": candidate,
            "comparison": comparison,
        })
        print(
            f"VL_PHASE_B {index}/{len(data['cases'])} id={item['id']} "
            f"exact={comparison['token_exact']} first_div={div} "
            f"full_tokens={len(full['generated_ids'])} candidate_tokens={len(candidate['generated_ids'])}",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "scope": "long-generation deterministic multimodal A/B regression; not a global multimodal quality or physical-storage claim",
        "mask": str(mask),
        "cases": rows,
        "aggregate": summarize(rows),
    }
    out = work / "vl-phase-b-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    a = summary["aggregate"]
    print(
        "KIMI_VL_PHASE_B_COMPLETE "
        f"exact={a['token_exact_cases']}/{a['cases']} "
        f"full_tokens={a['full_generated_tokens_total']} "
        f"candidate_tokens={a['candidate_generated_tokens_total']} summary={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
