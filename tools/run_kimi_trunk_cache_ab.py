#!/usr/bin/env python3
"""Real-weight A/B for bounded trunk caching on the direct-GGUF Kimi text runtime."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

PROMPT = (
    "Continue this sequence with the next eight positive integers, separated by single spaces only: "
    "1 2 3 4"
)

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
TRUNK_RX = re.compile(
    r"kvl_trunk_cache: resident=([0-9.]+)/([0-9.]+) MiB loads=(\d+) hits=(\d+) "
    r"inserts=(\d+) reads=([0-9.]+) MiB"
)


def parse_run(stdout: str, stderr: str) -> dict:
    tm = TIMING_RX.search(stderr)
    im = IDS_RX.search(stderr)
    cm = CACHE_RX.search(stderr)
    tr = TRUNK_RX.search(stderr)
    if not tm or not im or not cm or not tr:
        missing = []
        if not tm:
            missing.append("timing")
        if not im:
            missing.append("generated_ids")
        if not cm:
            missing.append("expert_cache")
        if not tr:
            missing.append("trunk_cache")
        raise RuntimeError("unable to parse run: missing " + ",".join(missing))
    return {
        "output": stdout.strip(),
        "ids": [int(x) for x in im.group(1).split()],
        "direct_io": "expert_direct_io=yes" in stderr and "trunk_direct_io=yes" in stderr,
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
        "expert_read_ops": int(cm.group(7)),
        "expert_bytes_mib": float(cm.group(8)),
        "expert_rate_mib_s": float(cm.group(9)),
        "expert_failures": int(cm.group(10)),
        "trunk_resident_mib": float(tr.group(1)),
        "trunk_budget_mib": float(tr.group(2)),
        "trunk_loads": int(tr.group(3)),
        "trunk_hits": int(tr.group(4)),
        "trunk_inserts": int(tr.group(5)),
        "trunk_reads_mib": float(tr.group(6)),
    }


def run_one(args: argparse.Namespace, mode: str, rep: int) -> dict:
    trunk_cache_mib = 0 if mode == "stream" else args.trunk_cache_mib
    env = os.environ.copy()
    env["KVL_LAYERPIN_HYSTERESIS"] = "1"
    cmd = [
        sys.executable, str(args.chat_script), str(args.runtime_dir), PROMPT,
        "--binary", str(args.binary),
        "--cache-mib", str(args.cache_mib),
        "--trunk-cache-mib", str(trunk_cache_mib),
        "--ram-mib", str(args.ram_mib),
        "--max-new", str(args.max_new),
        "--temperature", "0", "--seed", "1", "--show-tokens",
    ]
    print(f"KIMI_TRUNK_CACHE_BEGIN mode={mode} rep={rep}", flush=True)
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    (args.evidence_dir / f"{mode}-{rep}.out").write_text(proc.stdout, encoding="utf-8")
    (args.evidence_dir / f"{mode}-{rep}.err").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"{mode} rep={rep} exited {proc.returncode}")
    row = parse_run(proc.stdout, proc.stderr)
    print(proc.stdout.strip(), flush=True)
    print(
        "KIMI_TRUNK_CACHE_END "
        f"mode={mode} rep={rep} generated={row['generated']} "
        f"avg_next_s={row['avg_next_s']:.3f} trunk_reads_mib={row['trunk_reads_mib']:.2f} "
        f"trunk_hits={row['trunk_hits']} expert_reads={row['prefetch_reads']}",
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
    ap.add_argument("--trunk-cache-mib", type=int, default=1706)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=12)
    args = ap.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)

    order = [("stream", 1), ("cache", 1), ("cache", 2), ("stream", 2)]
    rows = [run_one(args, mode, rep) for mode, rep in order]
    stream = [r for (mode, _), r in zip(order, rows) if mode == "stream"]
    cache = [r for (mode, _), r in zip(order, rows) if mode == "cache"]
    failures = []

    reference_ids = rows[0]["ids"]
    if any(r["ids"] != reference_ids for r in rows[1:]):
        failures.append("generated token sequence changed")
    if any(not r["direct_io"] or r["expert_failures"] for r in rows):
        failures.append("direct-I/O/infrastructure failure")

    logical = {(r["requests"], r["prefetch_batches"]) for r in rows}
    expert_physical = {(r["prefetch_reads"], r["expert_read_ops"], r["expert_bytes_mib"]) for r in rows}
    if len(logical) != 1:
        failures.append("logical routed request counts changed")
    if len(expert_physical) != 1:
        failures.append("expert cache physical behavior changed")
    if any(r["trunk_resident_mib"] != 0.0 or r["trunk_hits"] != 0 for r in stream):
        failures.append("stream control unexpectedly retained trunk tensors")
    if any(r["trunk_resident_mib"] < 1600.0 or r["trunk_hits"] <= 0 for r in cache):
        failures.append("bounded trunk cache did not become resident/reused")
    if any(c["trunk_reads_mib"] >= s["trunk_reads_mib"] for s, c in zip(stream, cache)):
        failures.append("trunk cache did not reduce physical trunk reads")

    stream_trunk = statistics.median(r["trunk_reads_mib"] for r in stream)
    cache_trunk = statistics.median(r["trunk_reads_mib"] for r in cache)
    stream_next = statistics.median(r["avg_next_s"] for r in stream)
    cache_next = statistics.median(r["avg_next_s"] for r in cache)
    decode_forwards = min(r["generated"] - 1 for r in rows)
    saved_per_decode = (stream_trunk - cache_trunk) / decode_forwards if decode_forwards > 0 else None

    summary = {
        "claim_boundary": (
            "One deterministic short text prompt on Windows hosted runner, Q8_0 GGUF direct-I/O, "
            "512 MiB routed-expert cache, hysteresis enabled. Trunk cache changes storage residency only; "
            "token equality and unchanged expert counters guard math/routing regressions. Timing is hosted-runner observational evidence."
        ),
        "prompt": PROMPT,
        "order": [mode for mode, _ in order],
        "runs": [{"mode": mode, "rep": rep, **row} for (mode, rep), row in zip(order, rows)],
        "token_exact_all": not any(r["ids"] != reference_ids for r in rows[1:]),
        "logical_request_exact": len(logical) == 1,
        "expert_physical_exact": len(expert_physical) == 1,
        "stream_trunk_reads_mib_median": stream_trunk,
        "cache_trunk_reads_mib_median": cache_trunk,
        "saved_trunk_reads_mib_median": stream_trunk - cache_trunk,
        "saved_trunk_mib_per_decode_forward": saved_per_decode,
        "stream_avg_next_s_median": stream_next,
        "cache_avg_next_s_median": cache_next,
        "hosted_decode_ratio": (stream_next / cache_next) if cache_next > 0 else None,
        "failures": failures,
    }
    (args.evidence_dir / "trunk-cache-ab-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        for failure in failures:
            print("GATE_FAIL:", failure, file=sys.stderr)
        return 1
    print("KIMI_TRUNK_CACHE_AB_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
