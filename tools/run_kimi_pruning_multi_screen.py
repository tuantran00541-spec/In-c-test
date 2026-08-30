#!/usr/bin/env python3
"""Screen several nested Kimi MoE masks while sharing full baselines.

This is a deterministic next-token sensitivity screen, not a global quality
benchmark. It shares one full-text and one full-VL baseline per case across all
candidate masks, then selects the largest candidate that clears conservative
text and multimodal regression gates.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from run_kimi_pruning_phase_a import (
    DEFAULT_SUITE as DEFAULT_TEXT_SUITE,
    aggregate as aggregate_text,
    compare_variant as compare_text,
    run_heldout_one as run_text_one,
    validate_suite as validate_text_suite,
)
from run_kimi_pruning_vl_guard import (
    DEFAULT_SUITE as DEFAULT_VL_SUITE,
    compare_case as compare_vl,
    direct_mask_hits,
    read_mask,
    read_prompt_ids,
    run_one as run_vl_one,
    validate_suite as validate_vl_suite,
)


def parse_candidate(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"candidate must be NAME=MASK, got {spec!r}")
    name, raw = spec.split("=", 1)
    name = name.strip()
    raw = raw.strip()
    if not name or not raw or any(c.isspace() for c in name):
        raise ValueError(f"invalid candidate spec: {spec!r}")
    return name, Path(raw)


def candidate_specs(specs: list[str]) -> list[dict]:
    if not specs:
        raise ValueError("at least one --candidate NAME=MASK is required")
    out = []
    seen_names = set()
    for raw in specs:
        name, path = parse_candidate(raw)
        if name in seen_names:
            raise ValueError(f"duplicate candidate name: {name}")
        seen_names.add(name)
        if not path.is_file():
            raise ValueError(f"candidate mask not found: {path}")
        mask = read_mask(path)
        out.append({"name": name, "path": path.resolve(), "mask": mask, "disabled_count": len(mask)})
    counts = [x["disabled_count"] for x in out]
    if counts != sorted(counts):
        raise ValueError(f"candidates must be ordered weakest->strongest by disabled count: {counts}")
    for prev, cur in zip(out, out[1:]):
        if not prev["mask"].issubset(cur["mask"]):
            raise ValueError(f"candidate masks must be nested: {prev['name']} is not subset of {cur['name']}")
    return out


def screen_pass(text: dict, vl: dict) -> tuple[bool, list[str]]:
    failures = []
    tp = int(text["prompts"])
    if int(text["first_token_exact"]) != tp:
        failures.append(f"text_exact={text['first_token_exact']}/{tp}")
    if int(text["logit_argmax_agree"]) != tp:
        failures.append(f"text_argmax={text['logit_argmax_agree']}/{tp}")
    if float(text["logit_min_topk_overlap"]) < 0.9:
        failures.append(f"text_top10={text['logit_min_topk_overlap']:.6g}<0.9")
    if float(text["logit_max_js_divergence"]) > 0.005:
        failures.append(f"text_js={text['logit_max_js_divergence']:.6g}>0.005")

    vc = int(vl["cases"])
    if int(vl["first_token_exact"]) != vc:
        failures.append(f"vl_exact={vl['first_token_exact']}/{vc}")
    if int(vl["logit_argmax_agree"]) != vc:
        failures.append(f"vl_argmax={vl['logit_argmax_agree']}/{vc}")
    if float(vl["logit_min_topk_overlap"]) < 0.8:
        failures.append(f"vl_top10={vl['logit_min_topk_overlap']:.6g}<0.8")
    if float(vl["logit_max_js_divergence"]) > 0.01:
        failures.append(f"vl_js={vl['logit_max_js_divergence']:.6g}>0.01")
    return not failures, failures


def aggregate_vl(rows: list[dict]) -> dict:
    comps = [r["comparison"] for r in rows]
    direct_total = sum(int(r["direct_mask_hits"]["selections"]) for r in rows)
    media_direct_total = sum(int(r["direct_mask_hits"]["media_selections"]) for r in rows)
    text_direct_total = sum(int(r["direct_mask_hits"]["text_selections"]) for r in rows)
    all_hit = set()
    media_hit = set()
    text_hit = set()
    for row in rows:
        for s in row["direct_mask_hits"]["slots"]:
            key = (int(s["layer"]), int(s["expert"]))
            all_hit.add(key)
            if int(s.get("media_count") or 0):
                media_hit.add(key)
            if int(s.get("text_count") or 0):
                text_hit.add(key)
    substitutions = sum(int(c["route"]["substitutions"]) for c in comps)
    return {
        "cases": len(rows),
        "first_token_exact": sum(int(c["first_token_exact"]) for c in comps),
        "logit_argmax_agree": sum(int(c["logits"]["argmax_agree_records"]) for c in comps),
        "logit_max_abs_delta": max(float(c["logits"]["max_abs_logit_delta"]) for c in comps),
        "logit_max_js_divergence": max(float(c["logits"]["max_js_divergence"]) for c in comps),
        "logit_min_topk_overlap": min(float(c["logits"]["min_topk_overlap_fraction"]) for c in comps),
        "route_substitutions": substitutions,
        "route_min_selected_retention": min(float(c["route"]["selected_retention_fraction"]) for c in comps),
        "route_min_set_exact_fraction": min(float(c["route"]["set_exact_fraction"]) for c in comps),
        "direct_masked_selections": direct_total,
        "media_direct_masked_selections": media_direct_total,
        "text_direct_masked_selections": text_direct_total,
        "unique_masked_slots_hit": len(all_hit),
        "media_unique_masked_slots_hit": len(media_hit),
        "text_unique_masked_slots_hit": len(text_hit),
        "cascade_substitutions": substitutions - direct_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=DEFAULT_TEXT_SUITE)
    ap.add_argument("--vl-suite", type=Path, default=DEFAULT_VL_SUITE)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--image-root", type=Path)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--vision-binary", type=Path)
    ap.add_argument("--generate-binary", type=Path)
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    text_data = validate_text_suite(args.suite)
    vl_data = validate_vl_suite(args.vl_suite)
    candidates = candidate_specs(args.candidate)
    print(
        "KIMI_MULTI_SCREEN_VALID "
        f"text_heldout={len(text_data['heldout'])} vl_cases={len(vl_data['cases'])} "
        "candidates=" + ",".join(f"{c['name']}:{c['disabled_count']}" for c in candidates)
    )
    if args.validate_only:
        return 0

    required = (args.model_dir, args.image_root, args.binary, args.vision_binary,
                args.generate_binary, args.work_dir)
    if any(x is None for x in required):
        raise SystemExit("model/image/binaries/work-dir are required unless --validate-only")
    if args.cache_mib <= 0 or args.ram_mib <= 0:
        raise SystemExit("cache and RAM budgets must be positive")

    model = args.model_dir.resolve()
    image_root = args.image_root.resolve()
    binary = args.binary.resolve()
    vision_binary = args.vision_binary.resolve()
    generate_binary = args.generate_binary.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    text_rows: dict[str, list[dict]] = {c["name"]: [] for c in candidates}
    for index, item in enumerate(text_data["heldout"], 1):
        base = run_text_one(item, "full", None, model, binary, work / "text",
                            args.cache_mib, args.ram_mib)
        for c in candidates:
            cand = run_text_one(item, c["name"], c["path"], model, binary,
                                work / "text", args.cache_mib, args.ram_mib)
            comp = compare_text(item["id"], base, cand, work / "text" / c["name"])
            text_rows[c["name"]].append({
                "id": item["id"],
                "variants": {"full": base, c["name"]: {**cand, "comparison_to_full": comp}},
            })
            print(
                f"MULTI_TEXT {index}/{len(text_data['heldout'])} candidate={c['name']} "
                f"id={item['id']} exact={comp['first_token_exact']} "
                f"argmax={comp['logits']['argmax_agree_records']}/{comp['logits']['records']} "
                f"js={comp['logits']['max_js_divergence']:.9g}", flush=True,
            )

    text_aggregates = {
        c["name"]: aggregate_text(text_rows[c["name"]], c["name"])
        for c in candidates
    }

    vl_rows: dict[str, list[dict]] = {c["name"]: [] for c in candidates}
    for index, item in enumerate(vl_data["cases"], 1):
        base_root = work / "vl" / "full"
        base = run_vl_one(item, "full", None, model, image_root, vision_binary,
                          generate_binary, base_root, args.cache_mib, args.ram_mib)
        base_prompt_ids = read_prompt_ids(Path(base["prompt_ids"]))
        for c in candidates:
            cand_root = work / "vl" / c["name"]
            cand = run_vl_one(item, c["name"], c["path"], model, image_root,
                              vision_binary, generate_binary, cand_root,
                              args.cache_mib, args.ram_mib)
            cand_prompt_ids = read_prompt_ids(Path(cand["prompt_ids"]))
            if cand_prompt_ids != base_prompt_ids:
                raise RuntimeError(f"{item['id']} {c['name']}: full/candidate prompt ids differ")
            direct = direct_mask_hits(Path(base["trace"]), c["mask"], base_prompt_ids)
            comp = compare_vl(item["id"], base, cand, work / "vl" / "compare" / c["name"])
            substitutions = int(comp["route"]["substitutions"])
            if substitutions < int(direct["selections"]):
                raise RuntimeError(f"{item['id']} {c['name']}: substitutions < direct masked hits")
            vl_rows[c["name"]].append({
                "id": item["id"], "direct_mask_hits": direct, "comparison": comp,
                "cascade_substitutions": substitutions - int(direct["selections"]),
            })
            print(
                f"MULTI_VL {index}/{len(vl_data['cases'])} candidate={c['name']} "
                f"id={item['id']} exact={comp['first_token_exact']} "
                f"media_direct={direct['media_selections']} "
                f"argmax={comp['logits']['argmax_agree_records']}/{comp['logits']['records']} "
                f"js={comp['logits']['max_js_divergence']:.9g}", flush=True,
            )

    vl_aggregates = {c["name"]: aggregate_vl(vl_rows[c["name"]]) for c in candidates}
    result_candidates = []
    selected = None
    for c in candidates:
        name = c["name"]
        passed, failures = screen_pass(text_aggregates[name], vl_aggregates[name])
        row = {
            "name": name,
            "mask": str(c["path"]),
            "disabled_count": c["disabled_count"],
            "text": text_aggregates[name],
            "vl": vl_aggregates[name],
            "screen_pass": passed,
            "screen_failures": failures,
        }
        result_candidates.append(row)
        if passed:
            selected = row

    summary = {
        "schema_version": 1,
        "scope": "shared-baseline deterministic next-token text+multimodal pruning screen; not a global quality or speed claim",
        "text_heldout_prompts": len(text_data["heldout"]),
        "vl_cases": len(vl_data["cases"]),
        "candidates": result_candidates,
        "selected": None if selected is None else {
            "name": selected["name"],
            "mask": selected["mask"],
            "disabled_count": selected["disabled_count"],
        },
    }
    out = work / "multi-screen-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if selected is not None:
        shutil.copyfile(selected["mask"], work / "selected.mask")
        (work / "selected-name.txt").write_text(selected["name"] + "\n", encoding="utf-8")
        print(f"KIMI_MULTI_SCREEN_SELECTED name={selected['name']} disabled={selected['disabled_count']}")
    else:
        print("KIMI_MULTI_SCREEN_NO_SELECTION")
    print(f"KIMI_MULTI_SCREEN_COMPLETE summary={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
