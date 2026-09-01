#!/usr/bin/env python3
"""Real-weight diverse text A/B gate for decode-only layer-pin hysteresis."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CASES = [
    {
        "id": "en-primes",
        "prompt": "List the first twelve prime numbers in ascending order, separated by spaces. Output only the numbers.",
    },
    {
        "id": "vi-even",
        "prompt": "Hãy liệt kê mười hai số chẵn dương đầu tiên theo thứ tự tăng dần, cách nhau bằng dấu cách. Chỉ xuất các số.",
    },
    {
        "id": "en-vi-translation",
        "prompt": "Translate the following sentence into natural Vietnamese. Output only the translation: The patient engineer checked every cable twice before starting the quiet machine at sunrise.",
    },
    {
        "id": "en-reorder",
        "prompt": "Reorder these words into one grammatical English sentence. Output only the sentence: after / the / storm / the / young / gardener / carefully / tied / the / bent / tomato / plants / to / wooden / stakes",
    },
]

TIMING_RX = re.compile(
    r"\[kvl\] timing first_token=([0-9.]+)s avg_next=([0-9.]+)s "
    r"total=([0-9.]+)s generated=([0-9]+)"
)
IDS_RX = re.compile(r"\[kvl\] generated ids:\s*([^\r\n]+)")
CACHE_RX = re.compile(
    r"kvl_cache: .*? req=(\d+) hit=(\d+) miss=(\d+) hit_rate=[0-9.]+% "
    r"evict=(\d+) prefetch=(\d+)/(\d+) reads=(\d+) bytes=([0-9.]+) MiB "
    r"rate=([0-9.]+) MiB/s failures=(\d+)"
)
HYST_RX = re.compile(r"kvl_layerpin_hyst: decode_pass=(\d+) retained_total=(\d+)")


def parse_run(stdout: str, stderr: str) -> dict:
    tm = TIMING_RX.search(stderr)
    im = IDS_RX.search(stderr)
    cm = CACHE_RX.search(stderr)
    if not tm or not im or not cm:
        missing = []
        if not tm:
            missing.append("timing")
        if not im:
            missing.append("generated_ids")
        if not cm:
            missing.append("cache")
        raise RuntimeError("unable to parse run: missing " + ",".join(missing))

    hm = HYST_RX.findall(stderr)
    return {
        "output": stdout.strip(),
        "ids": [int(x) for x in im.group(1).split()],
        "direct_io": "expert_direct_io=yes" in stderr and "trunk_direct_io=yes" in stderr,
        "policy_hysteresis": "policy=hysteresis" in stderr,
        "policy_topweight": "policy=topweight" in stderr,
        "hyst_decode_passes": int(hm[-1][0]) if hm else 0,
        "hyst_retained_total": int(hm[-1][1]) if hm else 0,
        "first_token_s": float(tm.group(1)),
        "avg_next_s": float(tm.group(2)),
        "total_s": float(tm.group(3)),
        "generated": int(tm.group(4)),
        "requests": int(cm.group(1)),
        "hits": int(cm.group(2)),
        "misses": int(cm.group(3)),
        "evictions": int(cm.group(4)),
        "prefetch_reads": int(cm.group(5)),
        "prefetch_batches": int(cm.group(6)),
        "read_ops": int(cm.group(7)),
        "bytes_mib": float(cm.group(8)),
        "rate_mib_s": float(cm.group(9)),
        "failures": int(cm.group(10)),
    }


def run_one(args: argparse.Namespace, case: dict, mode: str) -> dict:
    out_path = args.evidence_dir / f"{case['id']}-{mode}.out"
    err_path = args.evidence_dir / f"{case['id']}-{mode}.err"

    env = os.environ.copy()
    if mode == "hysteresis":
        env["KVL_LAYERPIN_HYSTERESIS"] = "1"
    else:
        env.pop("KVL_LAYERPIN_HYSTERESIS", None)

    cmd = [
        sys.executable,
        str(args.chat_script),
        str(args.runtime_dir),
        case["prompt"],
        "--binary",
        str(args.binary),
        "--cache-mib",
        str(args.cache_mib),
        "--ram-mib",
        str(args.ram_mib),
        "--max-new",
        str(args.max_new),
        "--temperature",
        "0",
        "--seed",
        str(args.seed),
        "--show-tokens",
    ]
    print(f"KIMI_HYST_DIVERSE_BEGIN case={case['id']} mode={mode}", flush=True)
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    out_path.write_text(proc.stdout, encoding="utf-8")
    err_path.write_text(proc.stderr, encoding="utf-8")

    if proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"{case['id']} {mode} exited {proc.returncode}")

    row = parse_run(proc.stdout, proc.stderr)
    print(
        "KIMI_HYST_DIVERSE_END "
        f"case={case['id']} mode={mode} generated={row['generated']} "
        f"prefetch_reads={row['prefetch_reads']} bytes_mib={row['bytes_mib']:.2f} "
        f"avg_next_s={row['avg_next_s']:.3f}",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runtime_dir", type=Path)
    ap.add_argument("binary", type=Path)
    ap.add_argument("evidence_dir", type=Path)
    ap.add_argument("--chat-script", type=Path, default=Path("tools/kvl_chat.py"))
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-generated", type=int, default=8)
    args = ap.parse_args()

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    if not args.runtime_dir.is_dir():
        raise SystemExit(f"runtime directory missing: {args.runtime_dir}")
    if not args.binary.is_file():
        raise SystemExit(f"binary missing: {args.binary}")
    if not args.chat_script.is_file():
        raise SystemExit(f"chat script missing: {args.chat_script}")

    results = {}
    failures = []
    for i, case in enumerate(CASES):
        order = ("uniform", "hysteresis") if i % 2 == 0 else ("hysteresis", "uniform")
        arms = {}
        for mode in order:
            arms[mode] = run_one(args, case, mode)

        u = arms["uniform"]
        h = arms["hysteresis"]
        sequence_exact = u["ids"] == h["ids"]
        request_exact = u["requests"] == h["requests"]
        batch_exact = u["prefetch_batches"] == h["prefetch_batches"]
        io_delta_reads = u["prefetch_reads"] - h["prefetch_reads"]
        io_delta_bytes = u["bytes_mib"] - h["bytes_mib"]
        io_relation = (
            "better" if io_delta_reads > 0 and io_delta_bytes > 0
            else "equal" if io_delta_reads == 0 and abs(io_delta_bytes) < 1e-9
            else "worse"
        )

        case_summary = {
            "prompt": case["prompt"],
            "order": list(order),
            "uniform": u,
            "hysteresis": h,
            "sequence_exact": sequence_exact,
            "request_exact": request_exact,
            "prefetch_batch_exact": batch_exact,
            "saved_expert_loads": io_delta_reads,
            "saved_expert_mib": io_delta_bytes,
            "io_relation": io_relation,
        }
        results[case["id"]] = case_summary
        (args.evidence_dir / f"{case['id']}-summary.json").write_text(
            json.dumps(case_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if not sequence_exact:
            failures.append(f"{case['id']}: generated token sequence changed")
        if not request_exact or not batch_exact:
            failures.append(f"{case['id']}: logical request/batch counts changed")
        if min(u["generated"], h["generated"]) < args.min_generated:
            failures.append(f"{case['id']}: generation too short for decode evidence")
        if not u["direct_io"] or not h["direct_io"] or u["failures"] or h["failures"]:
            failures.append(f"{case['id']}: direct-I/O/infrastructure failure")
        if not u["policy_topweight"] or u["policy_hysteresis"]:
            failures.append(f"{case['id']}: uniform policy marker invalid")
        if not h["policy_hysteresis"] or h["policy_topweight"]:
            failures.append(f"{case['id']}: hysteresis policy marker invalid")
        if h["hyst_decode_passes"] <= 1 or h["hyst_retained_total"] <= 0:
            failures.append(f"{case['id']}: hysteresis did not retain prior pins")

    uniform_reads = sum(x["uniform"]["prefetch_reads"] for x in results.values())
    hyst_reads = sum(x["hysteresis"]["prefetch_reads"] for x in results.values())
    uniform_bytes = sum(x["uniform"]["bytes_mib"] for x in results.values())
    hyst_bytes = sum(x["hysteresis"]["bytes_mib"] for x in results.values())
    uniform_ops = sum(x["uniform"]["read_ops"] for x in results.values())
    hyst_ops = sum(x["hysteresis"]["read_ops"] for x in results.values())

    uniform_next_num = sum(
        x["uniform"]["avg_next_s"] * max(x["uniform"]["generated"] - 1, 0)
        for x in results.values()
    )
    uniform_next_den = sum(max(x["uniform"]["generated"] - 1, 0) for x in results.values())
    hyst_next_num = sum(
        x["hysteresis"]["avg_next_s"] * max(x["hysteresis"]["generated"] - 1, 0)
        for x in results.values()
    )
    hyst_next_den = sum(max(x["hysteresis"]["generated"] - 1, 0) for x in results.values())
    uniform_next = uniform_next_num / uniform_next_den if uniform_next_den else None
    hyst_next = hyst_next_num / hyst_next_den if hyst_next_den else None

    better = sum(x["io_relation"] == "better" for x in results.values())
    equal = sum(x["io_relation"] == "equal" for x in results.values())
    worse = sum(x["io_relation"] == "worse" for x in results.values())
    aggregate_io_better = hyst_reads < uniform_reads and hyst_bytes < uniform_bytes and hyst_ops < uniform_ops

    summary = {
        "claim_boundary": (
            "Windows-2025 hosted-runner real-weight four-prompt text gate comparing decode-only "
            "uniform top-weight layer pins with conservative hysteresis. Same binary, router/math, "
            "512 MiB expert cache, AVX2 and native direct I/O. Hysteresis retains an old pin only "
            "while that expert remains in the current top-6. Aggregate I/O is a gate; hosted-runner "
            "timing is observational only and not a target-laptop or global speed claim."
        ),
        "policy": {
            "pins_per_routed_layer": 2,
            "routed_layers": 26,
            "reserved_pin_slots": 52,
            "transient_slots": 6,
            "cache_mib": args.cache_mib,
            "hysteresis_rule": "retain previous pin iff still in current top-k; fill remaining pins by current route weight",
        },
        "case_count": len(results),
        "cases": results,
        "sequence_exact_cases": sum(x["sequence_exact"] for x in results.values()),
        "request_exact_cases": sum(x["request_exact"] for x in results.values()),
        "prefetch_batch_exact_cases": sum(x["prefetch_batch_exact"] for x in results.values()),
        "io_better_cases": better,
        "io_equal_cases": equal,
        "io_worse_cases": worse,
        "uniform_total_prefetch_reads": uniform_reads,
        "hysteresis_total_prefetch_reads": hyst_reads,
        "saved_expert_loads_total": uniform_reads - hyst_reads,
        "uniform_total_read_ops": uniform_ops,
        "hysteresis_total_read_ops": hyst_ops,
        "saved_read_ops_total": uniform_ops - hyst_ops,
        "uniform_total_bytes_mib": uniform_bytes,
        "hysteresis_total_bytes_mib": hyst_bytes,
        "saved_expert_mib_total": uniform_bytes - hyst_bytes,
        "expert_bytes_reduction_fraction_total": (
            (uniform_bytes - hyst_bytes) / uniform_bytes if uniform_bytes else None
        ),
        "uniform_weighted_avg_next_s": uniform_next,
        "hysteresis_weighted_avg_next_s": hyst_next,
        "hosted_runner_observed_next_token_ratio_x": (
            uniform_next / hyst_next if uniform_next and hyst_next else None
        ),
        "aggregate_io_better": aggregate_io_better,
        "failures": failures,
    }
    (args.evidence_dir / "gate-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if not aggregate_io_better:
        failures.append("aggregate physical expert I/O did not improve versus uniform layer pins")
    if failures:
        for failure in failures:
            print("GATE_FAIL:", failure, file=sys.stderr)
        return 1
    print("KIMI_LAYERPIN_HYST_DIVERSE_GATE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
