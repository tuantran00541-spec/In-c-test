#!/usr/bin/env python3
"""Run a next-token multimodal pruning sensitivity guard for Kimi-VL.

This guard complements the text-only Phase A/B experiments. For each image+text
case it runs the full Q8 expert store and one logical expert mask with max_new=1,
collecting routed-expert traces and the first next-token logit distribution.
It reports direct baseline selections of masked experts, route cascades, token
agreement, and logit drift.

Passing this screen is not a global multimodal quality claim and does not by
itself authorize physical expert deletion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from run_kimi_pruning_phase_a import (
    LOGIT_COMPARE,
    ROUTE_COMPARE,
    ROOT,
    parse_runtime,
    run_checked,
)

VL_CHAT = ROOT / "tools" / "kvl_vl_chat.py"
DEFAULT_SUITE = ROOT / "tests" / "data" / "kimi-functional-pruning-vl-suite.json"
VL_GENERATED_RE = re.compile(r"\[kvl-vl\] generated ids:\s*([^\n]*)")
VL_TIMING_RE = re.compile(
    r"\[kvl-vl\] timing vision=([0-9.]+)s first_text_token=([0-9.]+)s "
    r"avg_next=([0-9.]+)s text_total=([0-9.]+)s generated=(\d+)"
)


def validate_suite(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError(f"unsupported VL suite version: {data.get('version')}")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) < 4:
        raise ValueError("VL suite requires at least four cases")
    ids = set()
    for item in cases:
        ident = item.get("id")
        image = item.get("image")
        prompt = item.get("prompt")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError(f"invalid/duplicate VL id: {ident!r}")
        ids.add(ident)
        if not isinstance(image, str) or not image or Path(image).name != image:
            raise ValueError(f"{ident}: image must be a simple filename")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{ident}: empty prompt")
    return data


def read_mask(path: Path) -> set[tuple[int, int]]:
    out = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected layer expert")
        key = tuple(map(int, parts))
        if key[0] < 0 or key[1] < 0 or key in out:
            raise ValueError(f"{path}:{lineno}: invalid/duplicate mask entry {key}")
        out.add(key)
    if not out:
        raise ValueError(f"{path}: empty mask")
    return out


def parse_generated(stderr: str) -> list[int]:
    m = VL_GENERATED_RE.search(stderr)
    if not m:
        raise ValueError("missing [kvl-vl] generated ids")
    text = m.group(1).strip()
    return [int(x) for x in text.split()] if text else []


def parse_vl_runtime(stderr: str) -> dict:
    out = parse_runtime(stderr)  # Reuse cache-report parser; VL timing has a different line.
    m = VL_TIMING_RE.search(stderr)
    if m:
        out["vl_timing"] = {
            "vision_seconds": float(m.group(1)),
            "first_text_token_seconds": float(m.group(2)),
            "avg_next_seconds": float(m.group(3)),
            "text_total_seconds": float(m.group(4)),
            "generated": int(m.group(5)),
        }
    return out


def direct_mask_hits(trace: Path, mask: set[tuple[int, int]]) -> dict:
    counts: dict[tuple[int, int], int] = {}
    saliency: dict[tuple[int, int], float] = {}
    total = 0
    for lineno, raw in enumerate(trace.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{trace}:{lineno}: expected six columns")
        _, layer, expert = map(int, parts[:3])
        key = (layer, expert)
        if key not in mask:
            continue
        s = float(parts[5])
        counts[key] = counts.get(key, 0) + 1
        saliency[key] = saliency.get(key, 0.0) + s
        total += 1
    return {
        "selections": total,
        "unique_slots": len(counts),
        "slots": [
            {"layer": l, "expert": e, "count": counts[(l, e)], "saliency": saliency[(l, e)]}
            for l, e in sorted(counts)
        ],
    }


def vl_cmd(model: Path, image: Path, prompt: str, vision_binary: Path,
           generate_binary: Path, cache_mib: int, ram_mib: int) -> list[str]:
    return [
        sys.executable, str(VL_CHAT), str(model), str(image), prompt,
        "--vision-binary", str(vision_binary),
        "--generate-binary", str(generate_binary),
        "--cache-mib", str(cache_mib), "--ram-mib", str(ram_mib),
        "--max-new", "1", "--temperature", "0", "--seed", "1", "--show-tokens",
    ]


def run_one(item: dict, variant: str, mask: Path | None, model: Path,
            image_root: Path, vision_binary: Path, generate_binary: Path,
            work: Path, cache_mib: int, ram_mib: int) -> dict:
    root = work / "cases" / item["id"] / variant
    root.mkdir(parents=True, exist_ok=True)
    image = image_root / item["image"]
    if not image.is_file():
        raise RuntimeError(f"missing VL image: {image}")
    trace = root / "route.tsv"
    logits = root / "logits.bin"
    env = os.environ.copy()
    env["KVL_MOE_TRACE"] = str(trace)
    env["KVL_LOGITS_DUMP"] = str(logits)
    env["KVL_LOGITS_DUMP_LIMIT"] = "1"
    if mask is None:
        env.pop("KVL_MOE_MASK", None)
    else:
        env["KVL_MOE_MASK"] = str(mask)
    proc = run_checked(
        vl_cmd(model, image, item["prompt"], vision_binary, generate_binary, cache_mib, ram_mib),
        env=env, stdout_path=root / "output.txt", stderr_path=root / "stderr.txt",
    )
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(f"empty VL route trace: {trace}")
    if not logits.is_file() or logits.stat().st_size == 0:
        raise RuntimeError(f"empty VL logits dump: {logits}")
    result = {
        "variant": variant,
        "generated_ids": parse_generated(proc.stderr),
        "trace": str(trace),
        "logits": str(logits),
        "text": proc.stdout.strip(),
    }
    result.update(parse_vl_runtime(proc.stderr))
    return result


def compare_case(item_id: str, base: dict, cand: dict, work: Path) -> dict:
    root = work / "comparisons" / item_id
    root.mkdir(parents=True, exist_ok=True)
    route_json = root / "route.json"
    logits_json = root / "logits.json"
    run_checked([
        sys.executable, str(ROUTE_COMPARE), base["trace"], cand["trace"], "--out", str(route_json)
    ], stdout_path=root / "route.stdout", stderr_path=root / "route.stderr")
    run_checked([
        sys.executable, str(LOGIT_COMPARE), base["logits"], cand["logits"],
        "--topk", "10", "--out", str(logits_json)
    ], stdout_path=root / "logits.stdout", stderr_path=root / "logits.stderr")
    route = json.loads(route_json.read_text(encoding="utf-8"))
    logits = json.loads(logits_json.read_text(encoding="utf-8"))
    return {
        "first_token_exact": cand["generated_ids"] == base["generated_ids"],
        "route": route["summary"],
        "route_first_divergence": route["first_divergence"],
        "logits": logits["summary"],
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
    print(f"KIMI_VL_GUARD_SUITE_VALID cases={len(data['cases'])}")
    if args.validate_only:
        return 0
    required = (args.model_dir, args.image_root, args.vision_binary,
                args.generate_binary, args.mask, args.work_dir)
    if any(x is None for x in required):
        raise SystemExit("model/image/binaries/mask/work-dir required unless --validate-only")
    if args.cache_mib <= 0 or args.ram_mib <= 0:
        raise SystemExit("cache and RAM budgets must be positive")

    model = args.model_dir.resolve()
    image_root = args.image_root.resolve()
    vision_binary = args.vision_binary.resolve()
    generate_binary = args.generate_binary.resolve()
    mask_path = args.mask.resolve()
    work = args.work_dir.resolve()
    mask = read_mask(mask_path)
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    all_hit_slots: set[tuple[int, int]] = set()
    for index, item in enumerate(data["cases"], 1):
        base = run_one(item, "full", None, model, image_root, vision_binary,
                       generate_binary, work, args.cache_mib, args.ram_mib)
        cand = run_one(item, "candidate", mask_path, model, image_root, vision_binary,
                       generate_binary, work, args.cache_mib, args.ram_mib)
        direct = direct_mask_hits(Path(base["trace"]), mask)
        for s in direct["slots"]:
            all_hit_slots.add((s["layer"], s["expert"]))
        comp = compare_case(item["id"], base, cand, work)
        substitutions = int(comp["route"]["substitutions"])
        cascade = substitutions - int(direct["selections"])
        if cascade < 0:
            raise RuntimeError(
                f"{item['id']}: substitutions={substitutions} < direct mask hits={direct['selections']}"
            )
        rows.append({
            "id": item["id"], "image": item["image"], "prompt": item["prompt"],
            "full": base, "candidate": cand, "direct_mask_hits": direct,
            "comparison": comp, "cascade_substitutions": cascade,
        })
        print(
            f"VL_GUARD {index}/{len(data['cases'])} id={item['id']} "
            f"token_exact={comp['first_token_exact']} direct={direct['selections']} "
            f"route_sub={substitutions} cascade={cascade} "
            f"argmax={comp['logits']['argmax_agree_records']}/{comp['logits']['records']} "
            f"js={comp['logits']['max_js_divergence']:.9g}",
            flush=True,
        )

    comparisons = [r["comparison"] for r in rows]
    direct_total = sum(r["direct_mask_hits"]["selections"] for r in rows)
    substitutions_total = sum(r["comparison"]["route"]["substitutions"] for r in rows)
    summary = {
        "schema_version": 1,
        "scope": "next-token multimodal sensitivity guard; not a global multimodal quality claim",
        "mask": str(mask_path),
        "mask_entries": len(mask),
        "cases": rows,
        "aggregate": {
            "cases": len(rows),
            "first_token_exact": sum(int(c["first_token_exact"]) for c in comparisons),
            "logit_argmax_agree": sum(c["logits"]["argmax_agree_records"] for c in comparisons),
            "logit_max_js_divergence": max(c["logits"]["max_js_divergence"] for c in comparisons),
            "logit_min_topk_overlap": min(c["logits"]["min_topk_overlap_fraction"] for c in comparisons),
            "direct_masked_selections": direct_total,
            "unique_masked_slots_hit": len(all_hit_slots),
            "route_substitutions": substitutions_total,
            "cascade_substitutions": substitutions_total - direct_total,
        },
    }
    out = work / "vl-guard-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    a = summary["aggregate"]
    print(
        "KIMI_VL_PRUNING_GUARD_COMPLETE "
        f"exact={a['first_token_exact']}/{a['cases']} "
        f"argmax={a['logit_argmax_agree']}/{a['cases']} "
        f"hit_slots={a['unique_masked_slots_hit']} direct={a['direct_masked_selections']} "
        f"substitutions={a['route_substitutions']} max_js={a['logit_max_js_divergence']:.9g} "
        f"summary={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
