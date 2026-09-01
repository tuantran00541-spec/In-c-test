#!/usr/bin/env python3
"""Real-weight hard-math A/B gate for uniform versus hysteresis layer pins."""
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
            "Reason briefly in at most three sentences, then end with FINAL=<integer>."
        ),
        "kind": "int",
        "answer": "346",
    },
    {
        "id": "comb-en",
        "prompt": (
            "How many 5-digit decimal numbers can be formed from the digits 0,1,2,3,4,5,6,7 "
            "without repetition, if the number contains exactly two even digits? "
            "A 5-digit number cannot start with 0. Reason briefly in at most three sentences, "
            "then end with FINAL=<integer>."
        ),
        "kind": "int",
        "answer": "2592",
    },
    {
        "id": "rectangle-vi",
        "prompt": (
            "Một hình chữ nhật có chu vi 70 và đường chéo dài 25. "
            "Hãy tính diện tích hình chữ nhật. Giải thích ngắn gọn trong tối đa ba câu, "
            "sau đó kết thúc bằng FINAL=<số nguyên>."
        ),
        "kind": "int",
        "answer": "300",
    },
    {
        "id": "cards-en",
        "prompt": (
            "Three cards are drawn uniformly without replacement from a standard 52-card deck. "
            "What is the probability that exactly two of the three cards are aces? "
            "Give the exact reduced fraction. Reason briefly in at most three sentences, "
            "then end with FINAL=<p/q>."
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
HYST_RX = re.compile(r"kvl_layerpin_hyst: decode_pass=(\d+) retained_total=(\d+)")
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

    # If sides are a,b then a+b=35 and a^2+b^2=625, hence 2ab=35^2-625=600.
    assert (35 * 35 - 25 * 25) // 2 == 300

    prob = Fraction(math.comb(4, 2) * math.comb(48, 1), math.comb(52, 3))
    assert prob == Fraction(72, 5525)


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
    env = os.environ.copy()
    if mode == "hysteresis":
        env["KVL_LAYERPIN_HYSTERESIS"] = "1"
    else:
        env.pop("KVL_LAYERPIN_HYSTERESIS", None)

    out_path = args.evidence_dir / f"{case['id']}-{mode}.out"
    err_path = args.evidence_dir / f"{case['id']}-{mode}.err"
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
    print(f"KIMI_HARD_MATH_BEGIN case={case['id']} mode={mode}", flush=True)
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    out_path.write_text(proc.stdout, encoding="utf-8")
    err_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"{case['id']} {mode} exited {proc.returncode}")

    row = parse_run(proc.stdout, proc.stderr)
    row["score"] = score_answer(row["output"], case["answer"], case["kind"])
    print(proc.stdout.strip(), flush=True)
    print(
        "KIMI_HARD_MATH_END "
        f"case={case['id']} mode={mode} correct={int(row['score']['semantic_correct'])} "
        f"format={int(row['score']['format_compliant'])} generated={row['generated']} "
        f"prefetch_reads={row['prefetch_reads']} bytes_mib={row['bytes_mib']:.2f} "
        f"avg_next_s={row['avg_next_s']:.3f}",
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
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []
    regressions = []

    for i, case in enumerate(CASES):
        order = ("uniform", "hysteresis") if i % 2 == 0 else ("hysteresis", "uniform")
        arms = {mode: run_one(args, case, mode) for mode in order}
        u, h = arms["uniform"], arms["hysteresis"]

        exact = u["ids"] == h["ids"]
        logical_exact = (
            u["requests"] == h["requests"]
            and u["prefetch_batches"] == h["prefetch_batches"]
        )
        if not exact:
            failures.append(f"{case['id']}: generated token sequence changed")
        if not logical_exact:
            failures.append(f"{case['id']}: logical routed request counts changed")
        if not u["direct_io"] or not h["direct_io"] or u["failures"] or h["failures"]:
            failures.append(f"{case['id']}: direct-I/O/infrastructure failure")
        if not u["policy_topweight"] or u["policy_hysteresis"]:
            failures.append(f"{case['id']}: uniform policy marker invalid")
        if not h["policy_hysteresis"] or h["policy_topweight"]:
            failures.append(f"{case['id']}: hysteresis policy marker invalid")
        if h["hyst_decode_passes"] <= 1 or h["hyst_retained_total"] <= 0:
            failures.append(f"{case['id']}: hysteresis did not retain prior pins")

        baseline_correct = u["score"]["semantic_correct"]
        candidate_correct = h["score"]["semantic_correct"]
        if baseline_correct and not candidate_correct:
            regressions.append(case["id"])

        results[case["id"]] = {
            "prompt": case["prompt"],
            "expected": case["answer"],
            "order": list(order),
            "uniform": u,
            "hysteresis": h,
            "sequence_exact": exact,
            "logical_request_exact": logical_exact,
            "saved_expert_loads": u["prefetch_reads"] - h["prefetch_reads"],
            "saved_expert_mib": u["bytes_mib"] - h["bytes_mib"],
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "intelligence_regression": baseline_correct and not candidate_correct,
        }
        (args.evidence_dir / f"{case['id']}-summary.json").write_text(
            json.dumps(results[case["id"]], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    ur = sum(x["uniform"]["prefetch_reads"] for x in results.values())
    hr = sum(x["hysteresis"]["prefetch_reads"] for x in results.values())
    ub = sum(x["uniform"]["bytes_mib"] for x in results.values())
    hb = sum(x["hysteresis"]["bytes_mib"] for x in results.values())
    baseline_score = sum(x["baseline_correct"] for x in results.values())
    candidate_score = sum(x["candidate_correct"] for x in results.values())

    summary = {
        "claim_boundary": (
            "Four deterministic hard-math text prompts on Windows-2025 hosted runner, "
            "Q8_0 GGUF direct-I/O, 512 MiB cache. Quality is scored against independently "
            "verified ground truth; a baseline-correct to hysteresis-wrong transition is an "
            "intelligence-retention regression. Token exactness is a separate stronger regression check. "
            "Hosted-runner timings are observational only."
        ),
        "case_count": len(results),
        "cases": results,
        "sequence_exact_cases": sum(x["sequence_exact"] for x in results.values()),
        "baseline_correct_cases": baseline_score,
        "candidate_correct_cases": candidate_score,
        "intelligence_regressions": regressions,
        "uniform_total_prefetch_reads": ur,
        "hysteresis_total_prefetch_reads": hr,
        "saved_expert_loads_total": ur - hr,
        "uniform_total_bytes_mib": ub,
        "hysteresis_total_bytes_mib": hb,
        "saved_expert_mib_total": ub - hb,
        "failures": failures,
    }
    (args.evidence_dir / "hard-math-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if regressions:
        failures.append("baseline-correct -> hysteresis-wrong intelligence regression")
    if failures:
        for failure in failures:
            print("GATE_FAIL:", failure, file=sys.stderr)
        return 1
    print("KIMI_LAYERPIN_HYST_HARD_MATH_GATE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
