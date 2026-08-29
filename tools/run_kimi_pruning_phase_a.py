#!/usr/bin/env python3
"""Run a next-token-only Kimi functional-pruning A/B sweep.

The expensive model pack is intentionally external to this tool. Given one packed
Q8 Kimi directory, it:
  1. collects route-only calibration traces from the suite,
  2. builds fixed 62/60/58 and unseen-cap masks,
  3. evaluates held-out prompts with max_new=1 for full/60/58/adaptive,
  4. compares route changes and first-next-token logits,
  5. writes one machine-readable phase-a-summary.json.

This phase does not claim semantic quality: it is a sensitivity screen used to
choose which variants deserve longer generation A/B tests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "tools" / "kvl_chat.py"
ANALYZE = ROOT / "tools" / "analyze_kimi_moe_trace.py"
ROUTE_COMPARE = ROOT / "tools" / "compare_kimi_moe_routes.py"
LOGIT_COMPARE = ROOT / "tools" / "compare_kimi_logits.py"
DEFAULT_SUITE = ROOT / "tests" / "data" / "kimi-functional-pruning-suite.json"

GENERATED_RE = re.compile(r"\[kvl\] generated ids:\s*([^\n]*)")
TIMING_RE = re.compile(
    r"\[kvl\] timing first_token=([0-9.]+)s avg_next=([0-9.]+)s "
    r"total=([0-9.]+)s generated=(\d+)"
)
CACHE_RE = re.compile(
    r"kvl_cache:.*?req=(\d+) hit=(\d+) miss=(\d+).*?evict=(\d+) "
    r"prefetch=(\d+)/(\d+) reads=(\d+) bytes=([0-9.]+) MiB .*?failures=(\d+)"
)


def validate_suite(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError(f"unsupported suite version: {data.get('version')}")
    calibration = data.get("calibration")
    heldout = data.get("heldout")
    if not isinstance(calibration, list) or not isinstance(heldout, list):
        raise ValueError("suite must contain calibration and heldout lists")
    if len(calibration) < 12 or len(heldout) < 12:
        raise ValueError(f"suite too small: calibration={len(calibration)} heldout={len(heldout)}")
    all_ids = set()
    all_prompts = set()
    for split_name, split in (("calibration", calibration), ("heldout", heldout)):
        for item in split:
            ident = item.get("id")
            prompt = item.get("prompt")
            if not isinstance(ident, str) or not ident:
                raise ValueError(f"{split_name}: invalid id {ident!r}")
            if ident in all_ids:
                raise ValueError(f"duplicate suite id: {ident}")
            all_ids.add(ident)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{ident}: empty prompt")
            if prompt in all_prompts:
                raise ValueError(f"duplicate calibration/heldout prompt: {ident}")
            all_prompts.add(prompt)
    return data


def parse_generated(stderr: str) -> list[int]:
    m = GENERATED_RE.search(stderr)
    if not m:
        raise ValueError("missing generated ids in kvl_chat stderr")
    text = m.group(1).strip()
    return [int(x) for x in text.split()] if text else []


def parse_runtime(stderr: str) -> dict:
    out = {}
    m = TIMING_RE.search(stderr)
    if m:
        out["timing"] = {
            "first_token_seconds": float(m.group(1)),
            "avg_next_seconds": float(m.group(2)),
            "total_seconds": float(m.group(3)),
            "generated": int(m.group(4)),
        }
    m = CACHE_RE.search(stderr)
    if m:
        out["cache"] = {
            "requests": int(m.group(1)),
            "hits": int(m.group(2)),
            "misses": int(m.group(3)),
            "evictions": int(m.group(4)),
            "prefetch_reads": int(m.group(5)),
            "prefetch_batches": int(m.group(6)),
            "read_ops": int(m.group(7)),
            "bytes_read_mib": float(m.group(8)),
            "failures": int(m.group(9)),
        }
    return out


def run_checked(cmd: list[str], *, env=None, stdout_path: Path | None = None,
                stderr_path: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(proc.stdout, encoding="utf-8")
    if stderr_path:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    return proc


def chat_cmd(model_dir: Path, binary: Path, prompt: str, cache_mib: int,
             ram_mib: int) -> list[str]:
    return [
        sys.executable, str(CHAT), str(model_dir), prompt,
        "--binary", str(binary),
        "--cache-mib", str(cache_mib),
        "--ram-mib", str(ram_mib),
        "--max-new", "1",
        "--temperature", "0",
        "--seed", "1",
        "--show-tokens",
    ]


def run_calibration(data: dict, model_dir: Path, binary: Path, work: Path,
                    cache_mib: int, ram_mib: int) -> list[Path]:
    root = work / "calibration"
    root.mkdir(parents=True, exist_ok=True)
    traces = []
    for index, item in enumerate(data["calibration"], 1):
        ident = item["id"]
        prefix = root / f"{index:02d}-{ident}"
        trace = prefix.with_suffix(".route.tsv")
        env = os.environ.copy()
        env["KVL_MOE_TRACE"] = str(trace)
        proc = run_checked(
            chat_cmd(model_dir, binary, item["prompt"], cache_mib, ram_mib),
            env=env,
            stdout_path=prefix.with_suffix(".out"),
            stderr_path=prefix.with_suffix(".err"),
        )
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError(f"empty calibration trace: {trace}")
        ids = parse_generated(proc.stderr)
        print(f"PHASE_A_CALIBRATION {index}/{len(data['calibration'])} id={ident} generated={ids}")
        traces.append(trace)
    return traces


def build_masks(traces: list[Path], work: Path) -> tuple[Path, dict]:
    analysis = work / "analysis"
    cmd = [
        sys.executable, str(ANALYZE), *map(str, traces),
        "--n-experts", "64", "--keep", "62", "60", "58",
        "--unseen-cap", "2", "4", "6",
        "--out-dir", str(analysis), "--coactivation-top", "20",
    ]
    proc = run_checked(cmd, stdout_path=analysis / "analyze.stdout",
                       stderr_path=analysis / "analyze.stderr")
    print(proc.stdout.strip())
    report = json.loads((analysis / "report.json").read_text(encoding="utf-8"))
    return analysis, report


def run_heldout_one(item: dict, variant: str, mask: Path | None,
                    model_dir: Path, binary: Path, work: Path,
                    cache_mib: int, ram_mib: int) -> dict:
    root = work / "heldout" / item["id"] / variant
    root.mkdir(parents=True, exist_ok=True)
    trace = root / "route.tsv"
    logits = root / "logits.bin"
    env = os.environ.copy()
    env["KVL_MOE_TRACE"] = str(trace)
    env["KVL_LOGITS_DUMP"] = str(logits)
    env["KVL_LOGITS_DUMP_LIMIT"] = "1"
    if mask is not None:
        env["KVL_MOE_MASK"] = str(mask)
    else:
        env.pop("KVL_MOE_MASK", None)
    proc = run_checked(
        chat_cmd(model_dir, binary, item["prompt"], cache_mib, ram_mib),
        env=env,
        stdout_path=root / "output.txt",
        stderr_path=root / "stderr.txt",
    )
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(f"empty heldout route trace: {trace}")
    if not logits.is_file() or logits.stat().st_size == 0:
        raise RuntimeError(f"empty heldout logits dump: {logits}")
    result = {
        "variant": variant,
        "generated_ids": parse_generated(proc.stderr),
        "trace": str(trace),
        "logits": str(logits),
    }
    result.update(parse_runtime(proc.stderr))
    return result


def compare_variant(item_id: str, base: dict, variant: dict, work: Path) -> dict:
    compare_root = work / "comparisons" / item_id / variant["variant"]
    compare_root.mkdir(parents=True, exist_ok=True)
    route_json = compare_root / "route.json"
    logit_json = compare_root / "logits.json"
    run_checked([
        sys.executable, str(ROUTE_COMPARE), base["trace"], variant["trace"],
        "--out", str(route_json),
    ], stdout_path=compare_root / "route.stdout", stderr_path=compare_root / "route.stderr")
    run_checked([
        sys.executable, str(LOGIT_COMPARE), base["logits"], variant["logits"],
        "--topk", "10", "--out", str(logit_json),
    ], stdout_path=compare_root / "logits.stdout", stderr_path=compare_root / "logits.stderr")
    route = json.loads(route_json.read_text(encoding="utf-8"))
    logits = json.loads(logit_json.read_text(encoding="utf-8"))
    return {
        "first_token_exact": variant["generated_ids"] == base["generated_ids"],
        "route": route["summary"],
        "route_first_divergence": route["first_divergence"],
        "logits": logits["summary"],
    }


def aggregate(per_prompt: list[dict], variant: str) -> dict:
    rows = [p["variants"][variant] for p in per_prompt]
    comparisons = [r["comparison_to_full"] for r in rows]
    caches = [r.get("cache", {}) for r in rows]
    timings = [r.get("timing", {}) for r in rows]
    return {
        "prompts": len(rows),
        "first_token_exact": sum(int(c["first_token_exact"]) for c in comparisons),
        "route_substitutions": sum(c["route"]["substitutions"] for c in comparisons),
        "route_min_selected_retention": min(c["route"]["selected_retention_fraction"] for c in comparisons),
        "route_min_set_exact_fraction": min(c["route"]["set_exact_fraction"] for c in comparisons),
        "logit_argmax_agree": sum(c["logits"]["argmax_agree_records"] for c in comparisons),
        "logit_max_abs_delta": max(c["logits"]["max_abs_logit_delta"] for c in comparisons),
        "logit_max_js_divergence": max(c["logits"]["max_js_divergence"] for c in comparisons),
        "logit_min_topk_overlap": min(c["logits"]["min_topk_overlap_fraction"] for c in comparisons),
        "cache_read_ops_total": sum(c.get("read_ops", 0) for c in caches),
        "cache_bytes_read_mib_total": sum(c.get("bytes_read_mib", 0.0) for c in caches),
        "first_token_seconds_total": sum(t.get("first_token_seconds", 0.0) for t in timings),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    data = validate_suite(args.suite)
    print(f"KIMI_PHASE_A_SUITE_VALID calibration={len(data['calibration'])} heldout={len(data['heldout'])}")
    if args.validate_only:
        return 0
    if args.model_dir is None or args.binary is None or args.work_dir is None:
        raise SystemExit("--model-dir, --binary and --work-dir are required unless --validate-only")
    if args.cache_mib <= 0 or args.ram_mib <= 0:
        raise SystemExit("cache and RAM budgets must be positive")

    model_dir = args.model_dir.resolve()
    binary = args.binary.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    traces = run_calibration(data, model_dir, binary, work, args.cache_mib, args.ram_mib)
    analysis, calibration_report = build_masks(traces, work)
    variants = {
        "full": None,
        "keep60": analysis / "mask-keep60.txt",
        "keep58": analysis / "mask-keep58.txt",
        "adaptive-unseen-cap6": analysis / "mask-unseen-cap6.txt",
    }
    for name, mask in variants.items():
        if mask is not None and not mask.is_file():
            raise RuntimeError(f"missing mask for {name}: {mask}")

    per_prompt = []
    for index, item in enumerate(data["heldout"], 1):
        prompt_record = {"id": item["id"], "category": item["category"], "variants": {}}
        base = run_heldout_one(
            item, "full", None, model_dir, binary, work, args.cache_mib, args.ram_mib
        )
        prompt_record["variants"]["full"] = base
        print(f"PHASE_A_HELDOUT {index}/{len(data['heldout'])} id={item['id']} variant=full ids={base['generated_ids']}")
        for variant in ("keep60", "keep58", "adaptive-unseen-cap6"):
            cur = run_heldout_one(
                item, variant, variants[variant], model_dir, binary, work,
                args.cache_mib, args.ram_mib,
            )
            cur["comparison_to_full"] = compare_variant(item["id"], base, cur, work)
            prompt_record["variants"][variant] = cur
            c = cur["comparison_to_full"]
            print(
                f"PHASE_A_HELDOUT {index}/{len(data['heldout'])} id={item['id']} "
                f"variant={variant} token_exact={c['first_token_exact']} "
                f"route_sub={c['route']['substitutions']} "
                f"argmax={c['logits']['argmax_agree_records']}/{c['logits']['records']} "
                f"js={c['logits']['max_js_divergence']:.9g}"
            )
        per_prompt.append(prompt_record)

    summary = {
        "schema_version": 1,
        "scope": "next-token-only sensitivity screen; not a semantic quality claim",
        "suite": str(args.suite),
        "calibration_prompts": len(data["calibration"]),
        "heldout_prompts": len(data["heldout"]),
        "calibration": {
            "coverage": calibration_report["coverage"],
            "masks": calibration_report["masks"],
            "unseen_cap_masks": calibration_report["unseen_cap_masks"],
        },
        "per_prompt": per_prompt,
        "aggregate": {
            variant: aggregate(per_prompt, variant)
            for variant in ("keep60", "keep58", "adaptive-unseen-cap6")
        },
    }
    out = work / "phase-a-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"KIMI_FUNCTIONAL_PHASE_A_COMPLETE summary={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
