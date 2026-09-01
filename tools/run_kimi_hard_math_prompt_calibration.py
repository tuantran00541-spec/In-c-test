#!/usr/bin/env python3
"""Baseline-only prompt calibration for the real Kimi direct-GGUF hard-math gate.

This intentionally does not compare cache policies.  Its only purpose is to
check whether the unchanged uniform/top-weight runtime can solve the same four
problems when it is allowed a tiny visible scratchpad before FINAL=... .
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

CASES = [
    {
        "id": "crt-en",
        "prompt": (
            "Find the smallest positive integer n satisfying n ≡ 3 (mod 7), "
            "n ≡ 5 (mod 11), and n ≡ 8 (mod 13). "
            "Show only a compact calculation, with at most 2 short calculation lines and no explanatory prose. "
            "Then output a final line exactly FINAL=<integer>."
        ),
        "kind": "int",
        "answer": "346",
    },
    {
        "id": "comb-en",
        "prompt": (
            "How many 5-digit decimal numbers can be formed from the digits 0,1,2,3,4,5,6,7 "
            "without repetition, if the number contains exactly two even digits? "
            "A 5-digit number cannot start with 0. "
            "Show only a compact calculation, with at most 2 short calculation lines and no explanatory prose. "
            "Then output a final line exactly FINAL=<integer>."
        ),
        "kind": "int",
        "answer": "2592",
    },
    {
        "id": "rectangle-vi",
        "prompt": (
            "Một hình chữ nhật có chu vi 70 và đường chéo dài 25. Hãy tính diện tích hình chữ nhật. "
            "Chỉ viết phép tính ngắn gọn, tối đa 2 dòng phép tính và không giải thích bằng văn xuôi. "
            "Sau đó xuất một dòng cuối đúng chính xác dạng FINAL=<số nguyên>."
        ),
        "kind": "int",
        "answer": "300",
    },
    {
        "id": "cards-en",
        "prompt": (
            "Three cards are drawn uniformly without replacement from a standard 52-card deck. "
            "What is the probability that exactly two of the three cards are aces? Give the exact reduced fraction. "
            "Show only a compact calculation, with at most 2 short calculation lines and no explanatory prose. "
            "Then output a final line exactly FINAL=<p/q>."
        ),
        "kind": "fraction",
        "answer": "72/5525",
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
FINAL_RX = re.compile(r"FINAL\s*[:=]\s*([+-]?\d+(?:/\d+)?)", re.IGNORECASE)
NUMBER_RX = re.compile(r"(?<![\d/])[+-]?\d+(?:/\d+)?(?![\d/])")


def verify_ground_truth() -> None:
    crt = [n for n in range(1, 1002) if n % 7 == 3 and n % 11 == 5 and n % 13 == 8]
    assert crt and crt[0] == 346

    count = 0
    for p in itertools.permutations("01234567", 5):
        if p[0] == "0":
            continue
        if sum((int(d) % 2) == 0 for d in p) == 2:
            count += 1
    assert count == 2592

    assert (35 * 35 - 25 * 25) // 2 == 300
    assert Fraction(math.comb(4, 2) * math.comb(48, 1), math.comb(52, 3)) == Fraction(72, 5525)


def canonical_value(raw: str, kind: str) -> str | None:
    raw = raw.strip()
    try:
        if kind == "int":
            return str(int(raw))
        if kind == "fraction":
            return str(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        return None
    return None


def score_answer(text: str, expected: str, kind: str) -> dict:
    markers = FINAL_RX.findall(text)
    marker_value = canonical_value(markers[-1], kind) if markers else None
    expected_value = canonical_value(expected, kind)
    all_numbers = NUMBER_RX.findall(text)
    fallback_value = canonical_value(all_numbers[-1], kind) if all_numbers else None
    semantic_value = marker_value if marker_value is not None else fallback_value
    return {
        "expected": expected_value,
        "semantic_value": semantic_value,
        "semantic_correct": semantic_value == expected_value,
        "format_compliant": marker_value is not None,
        "marker_value": marker_value,
        "fallback_value": fallback_value,
    }


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
    return {
        "output": stdout.strip(),
        "ids": [int(x) for x in im.group(1).split()],
        "direct_io": "expert_direct_io=yes" in stderr and "trunk_direct_io=yes" in stderr,
        "policy_hysteresis": "policy=hysteresis" in stderr,
        "policy_topweight": "policy=topweight" in stderr,
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


def run_one(args: argparse.Namespace, case: dict) -> dict:
    env = os.environ.copy()
    env.pop("KVL_LAYERPIN_HYSTERESIS", None)
    out_path = args.evidence_dir / f"{case['id']}-uniform.out"
    err_path = args.evidence_dir / f"{case['id']}-uniform.err"
    cmd = [
        sys.executable,
        str(args.chat_script),
        str(args.runtime_dir),
        case["prompt"],
        "--binary", str(args.binary),
        "--cache-mib", str(args.cache_mib),
        "--ram-mib", str(args.ram_mib),
        "--max-new", str(args.max_new),
        "--temperature", "0",
        "--seed", str(args.seed),
        "--show-tokens",
    ]
    print(f"KIMI_HARD_MATH_CAL_BEGIN case={case['id']}", flush=True)
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    out_path.write_text(proc.stdout, encoding="utf-8")
    err_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"{case['id']} exited {proc.returncode}")

    row = parse_run(proc.stdout, proc.stderr)
    row["score"] = score_answer(row["output"], case["answer"], case["kind"])
    if not row["direct_io"] or row["failures"]:
        raise RuntimeError(f"{case['id']}: direct-I/O/infrastructure failure")
    if not row["policy_topweight"] or row["policy_hysteresis"]:
        raise RuntimeError(f"{case['id']}: uniform policy marker invalid")
    print(proc.stdout.strip(), flush=True)
    print(
        "KIMI_HARD_MATH_CAL_END "
        f"case={case['id']} correct={int(row['score']['semantic_correct'])} "
        f"format={int(row['score']['format_compliant'])} generated={row['generated']} "
        f"prefetch_reads={row['prefetch_reads']} bytes_mib={row['bytes_mib']:.2f}",
        flush=True,
    )
    return row


def main() -> int:
    verify_ground_truth()
    ap = argparse.ArgumentParser()
    ap.add_argument("runtime_dir", type=Path)
    ap.add_argument("binary", type=Path)
    ap.add_argument("evidence_dir", type=Path)
    ap.add_argument("--chat-script", type=Path, default=Path("tools/kvl_chat.py"))
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)

    cases = {}
    for case in CASES:
        row = run_one(args, case)
        cases[case["id"]] = {
            "prompt": case["prompt"],
            "expected": case["answer"],
            "uniform": row,
        }
        (args.evidence_dir / f"{case['id']}-summary.json").write_text(
            json.dumps(cases[case["id"]], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    correct = sum(x["uniform"]["score"]["semantic_correct"] for x in cases.values())
    compliant = sum(x["uniform"]["score"]["format_compliant"] for x in cases.values())
    summary = {
        "claim_boundary": (
            "Baseline-only prompt calibration on the unchanged Windows-2025 hosted-runner "
            "direct-GGUF Q8_0 uniform/top-weight runtime with 512 MiB cache. This is not a "
            "cache-policy quality comparison and hosted timing is not a target-laptop benchmark."
        ),
        "case_count": len(cases),
        "correct_cases": correct,
        "format_compliant_cases": compliant,
        "quality_ready_for_ab": correct == len(cases),
        "cases": cases,
    }
    (args.evidence_dir / "hard-math-calibration-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("KIMI_HARD_MATH_PROMPT_CALIBRATION_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
