#!/usr/bin/env python3
"""Run a six-case real-Q8 Kimi-VL dynamic-skip first-token A/B gate.

This runner is intentionally correctness/evidence oriented. It reuses one MoonViT
output per case across baseline and candidate, compares same-prefix first-token
logits, checks that only media routes were skipped, and records runtime-reported
expert-store counters. It does not make a wall-clock speed claim.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from kimi_image import write_patches
from kimi_tokenizer import build_encoding, encode_image_chat

CASES = [
    {
        "id": "vl-user-vi-description",
        "image": "user-vi.jpg",
        "prompt": "Hãy mô tả nhân vật và biểu cảm trong hình bằng tiếng Việt, một câu ngắn.",
        "expected_l22_skips": 31,
    },
    {
        "id": "vl-user-en-description",
        "image": "user-en.jpg",
        "prompt": "Look at this image and answer only in English. Describe the character and facial expression in one short sentence.",
        "expected_l22_skips": 33,
    },
    {
        "id": "vl-synthetic-shapes",
        "image": "synthetic-shapes.png",
        "prompt": "Describe the colored geometric shapes in this image in one short sentence.",
        "expected_l22_skips": 68,
    },
    {
        "id": "vl-synthetic-checker",
        "image": "synthetic-checker.png",
        "prompt": "Describe the main visual pattern in this image in one short sentence.",
        "expected_l22_skips": 65,
    },
    {
        "id": "vl-phase-c-diagonal",
        "image": "synthetic-diagonal.png",
        "prompt": "Describe the dominant line pattern in this image in one short sentence.",
        "expected_l22_skips": 93,
    },
    {
        "id": "vl-phase-c-count",
        "image": "synthetic-count.png",
        "prompt": "How many red circles are visible? Reply with only the integer.",
        "expected_l22_skips": 86,
    },
]

CACHE_RE = re.compile(
    r"kvl_cache: .*?req=(?P<requests>\d+) hit=(?P<hits>\d+) miss=(?P<misses>\d+) "
    r"hit_rate=(?P<hit_rate>[0-9.]+)% evict=(?P<evictions>\d+) "
    r"prefetch=(?P<prefetch_reads>\d+)/(?P<prefetch_batches>\d+) "
    r"reads=(?P<read_ops>\d+) bytes=(?P<bytes_mib>[0-9.]+) MiB"
)


def run(cmd: list[str], *, env: dict[str, str] | None = None,
        stdout_path: Path | None = None, stderr_path: Path | None = None) -> None:
    stdout = stdout_path.open("w", encoding="utf-8") if stdout_path else None
    stderr = stderr_path.open("w", encoding="utf-8") if stderr_path else None
    try:
        subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, check=True, text=True)
    finally:
        if stdout:
            stdout.close()
        if stderr:
            stderr.close()


def parse_cache(stderr_path: Path) -> dict[str, int | float]:
    text = stderr_path.read_text(encoding="utf-8")
    matches = list(CACHE_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"{stderr_path}: expected one kvl_cache line, got {len(matches)}")
    g = matches[0].groupdict()
    return {
        "requests": int(g["requests"]),
        "hits": int(g["hits"]),
        "misses": int(g["misses"]),
        "hit_rate_percent": float(g["hit_rate"]),
        "evictions": int(g["evictions"]),
        "prefetch_reads": int(g["prefetch_reads"]),
        "prefetch_batches": int(g["prefetch_batches"]),
        "read_ops": int(g["read_ops"]),
        "bytes_read_mib": float(g["bytes_mib"]),
    }


def restore_images(repo: Path, image_root: Path) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        ("kimi-user-vi.jpg.b64", "user-vi.jpg"),
        ("kimi-user-en.jpg.b64", "user-en.jpg"),
    ):
        raw = (repo / ".github" / "testdata" / src_name).read_bytes()
        (image_root / dst_name).write_bytes(base64.b64decode(raw))

    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((40, 55, 190, 205), fill="red")
    d.ellipse((245, 55, 395, 205), fill="blue")
    d.polygon([(220, 245), (105, 405), (335, 405)], fill="green")
    im.save(image_root / "synthetic-shapes.png")

    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    cell = 56
    for y in range(8):
        for x in range(8):
            if (x + y) % 2:
                d.rectangle((x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1), fill="black")
    im.save(image_root / "synthetic-checker.png")

    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    for k in range(-448, 896, 80):
        d.line((k, 0, k - 448, 448), fill="purple", width=24)
    im.save(image_root / "synthetic-diagonal.png")

    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    for x, y in [(90, 90), (220, 90), (350, 90), (155, 260), (285, 260)]:
        d.ellipse((x - 38, y - 38, x + 38, y + 38), fill="red")
    im.save(image_root / "synthetic-count.png")


def load_stats(path: Path) -> tuple[list[dict[str, str]], int, int, int]:
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    routed = sum(int(r["routed"]) for r in rows)
    skipped = sum(int(r["skipped"]) for r in rows)
    non_media = sum(int(r["skipped"]) for r in rows if r["family"] != "media")
    return rows, routed, skipped, non_media


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--vision-binary", default="./build/kvl_vision")
    ap.add_argument("--baseline-binary", default="./build/kvl_generate_vl")
    ap.add_argument("--candidate-binary", default="./build/kvl_generate_vl_dynskip")
    ap.add_argument("--cache-bytes", type=int, default=536870912)
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()

    if args.cache_bytes <= 0:
        raise SystemExit("--cache-bytes must be positive")
    repo = Path.cwd()
    model = args.model_dir.resolve()
    work = args.work_dir.resolve()
    evidence = args.evidence_dir.resolve()
    policy = args.policy.resolve()
    work.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    image_root = work / "images"
    restore_images(repo, image_root)

    enc, _, special = build_encoding(model)
    media_pad_id = special["<|media_pad|>"]
    results = []

    for index, case in enumerate(CASES, 1):
        case_root = work / case["id"]
        case_evidence = evidence / case["id"]
        case_root.mkdir(parents=True, exist_ok=True)
        case_evidence.mkdir(parents=True, exist_ok=True)

        patches = case_root / "patches.f32"
        media = case_root / "media.f32"
        ids_path = case_root / "prompt.ids"
        gh, gw = write_patches(model, image_root / case["image"], patches)
        media_tokens = (gh // 2) * (gw // 2)
        ids = encode_image_chat(enc, case["prompt"], media_tokens, "You are a helpful assistant")
        if sum(token == media_pad_id for token in ids) != media_tokens:
            raise RuntimeError(f"{case['id']}: media placeholder count mismatch")
        ids_path.write_text("\n".join(map(str, ids)) + "\n", encoding="ascii")

        run([
            args.vision_binary,
            str(model / "vision.bin"), str(model / "vision.idx"),
            str(patches), str(gh), str(gw), str(media),
        ], stdout_path=case_evidence / "vision.out", stderr_path=case_evidence / "vision.err")

        baseline_out = case_evidence / "baseline.out"
        baseline_err = case_evidence / "baseline.err"
        baseline_logits = case_evidence / "baseline.logits"
        env = os.environ.copy()
        env["KVL_LOGITS_DUMP"] = str(baseline_logits)
        env["KVL_LOGITS_DUMP_LIMIT"] = "1"
        run([
            args.baseline_binary,
            str(model / "trunk.bin"), str(model / "trunk.idx"),
            str(model / "experts.bin"), str(model / "experts.idx"),
            str(ids_path), str(media), str(args.cache_bytes), "1", "0", "1",
        ], env=env, stdout_path=baseline_out, stderr_path=baseline_err)

        candidate_out = case_evidence / "candidate.out"
        candidate_err = case_evidence / "candidate.err"
        candidate_logits = case_evidence / "candidate.logits"
        stats_path = case_evidence / "dynskip-stats.tsv"
        env = os.environ.copy()
        env["KVL_MOE_DYNSKIP_POLICY"] = str(policy)
        env["KVL_MOE_DYNSKIP_PROMPT_IDS"] = str(ids_path)
        env["KVL_MOE_DYNSKIP_STATS"] = str(stats_path)
        env["KVL_LOGITS_DUMP"] = str(candidate_logits)
        env["KVL_LOGITS_DUMP_LIMIT"] = "1"
        env.pop("KVL_MOE_MASK", None)
        run([
            args.candidate_binary,
            str(model / "trunk.bin"), str(model / "trunk.idx"),
            str(model / "experts.bin"), str(model / "experts.idx"),
            str(ids_path), str(media), str(args.cache_bytes), "1", "0", "1",
        ], env=env, stdout_path=candidate_out, stderr_path=candidate_err)

        compare_json = case_evidence / "logit-compare.json"
        compare_stdout = case_evidence / "logit-compare.stdout"
        run([
            sys.executable, "tools/compare_kimi_logits.py",
            str(baseline_logits), str(candidate_logits),
            "--topk", "10", "--out", str(compare_json),
        ], stdout_path=compare_stdout)
        comp = json.loads(compare_json.read_text(encoding="utf-8"))
        summary = comp["summary"]

        rows, routed, skipped, non_media = load_stats(stats_path)
        l22 = next(
            (int(r["skipped"]) for r in rows if r["family"] == "media" and int(r["layer"]) == 22),
            None,
        )
        baseline_cache = parse_cache(baseline_err)
        candidate_cache = parse_cache(candidate_err)
        token_exact = baseline_out.read_text(encoding="utf-8") == candidate_out.read_text(encoding="utf-8")
        result = {
            "id": case["id"],
            "grid": [gh, gw],
            "media_tokens": media_tokens,
            "prompt_tokens": len(ids),
            "token_exact": token_exact,
            "baseline_token_line": baseline_out.read_text(encoding="utf-8").strip(),
            "candidate_token_line": candidate_out.read_text(encoding="utf-8").strip(),
            "routed": routed,
            "skipped": skipped,
            "skip_fraction": skipped / routed if routed else 0.0,
            "non_media_skipped": non_media,
            "l22_skipped": l22,
            "expected_l22_skips": case["expected_l22_skips"],
            "l22_checksum_match": l22 == case["expected_l22_skips"],
            "argmax_agree_records": summary["argmax_agree_records"],
            "records": summary["records"],
            "min_probability_overlap": summary["min_probability_overlap"],
            "max_total_variation": summary["max_total_variation"],
            "max_js_divergence": summary["max_js_divergence"],
            "min_topk_overlap_fraction": summary["min_topk_overlap_fraction"],
            "baseline_cache": baseline_cache,
            "candidate_cache": candidate_cache,
            "expert_store_bytes_delta_mib": (
                candidate_cache["bytes_read_mib"] - baseline_cache["bytes_read_mib"]
            ),
        }
        results.append(result)
        (case_evidence / "case-summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"KIMI_DYNSKIP_VL_CASE {index}/{len(CASES)} id={case['id']} "
            f"token_exact={int(token_exact)} skipped={skipped} l22={l22} "
            f"overlap={summary['min_probability_overlap']:.9g}",
            flush=True,
        )

    aggregate = {
        "schema": "kimi-dynskip-vl-gate-v1",
        "model": "moonshotai/Kimi-VL-A3B-Instruct",
        "revision": args.revision,
        "policy": str(policy),
        "cases": results,
        "summary": {
            "cases": len(results),
            "token_exact_cases": sum(int(r["token_exact"]) for r in results),
            "argmax_agree_cases": sum(int(r["argmax_agree_records"] == r["records"]) for r in results),
            "l22_checksum_match_cases": sum(int(r["l22_checksum_match"]) for r in results),
            "non_media_skipped": sum(r["non_media_skipped"] for r in results),
            "routed": sum(r["routed"] for r in results),
            "skipped": sum(r["skipped"] for r in results),
            "min_probability_overlap": min(r["min_probability_overlap"] for r in results),
            "max_total_variation": max(r["max_total_variation"] for r in results),
            "max_js_divergence": max(r["max_js_divergence"] for r in results),
            "min_topk_overlap_fraction": min(r["min_topk_overlap_fraction"] for r in results),
            "expert_store_bytes_delta_mib": sum(r["expert_store_bytes_delta_mib"] for r in results),
        },
        "claim_boundary": (
            "Six deterministic VL first-token full-Q8 A/B cases with one fixed conservative "
            "media-only policy; no autoregressive quality or wall-clock speed claim."
        ),
    }
    (evidence / "gate-summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    s = aggregate["summary"]
    print("KIMI_DYNSKIP_VL_GATE_COMPLETE " + json.dumps(s, sort_keys=True), flush=True)

    failed = (
        s["token_exact_cases"] != len(CASES)
        or s["argmax_agree_cases"] != len(CASES)
        or s["l22_checksum_match_cases"] != len(CASES)
        or s["non_media_skipped"] != 0
        or s["skipped"] <= 0
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
