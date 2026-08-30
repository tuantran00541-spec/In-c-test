#!/usr/bin/env python3
"""Run the Q0 boundary-expert BF16 -> simulated low-bit measurement pilot.

The runner deliberately measures only the experts listed in a pilot manifest.
It resolves their three source tensors through the pinned checkpoint index,
downloads only required source shards, groups experts to reuse a shard, and can
delete shards again once no future pilot expert needs them.

No quantized model weights are written. Output contains sensitivity JSON only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kimi_measure_bf16_expert import PINNED_REVISION, PREFIX, measure  # noqa: E402

REPO = "moonshotai/Kimi-VL-A3B-Instruct"


def validate_manifest(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != "kimi-compression-q0-pilot-v1":
        raise ValueError(f"unsupported pilot schema {doc.get('schema')!r}")
    if doc.get("source_revision") != PINNED_REVISION:
        raise ValueError("pilot manifest is not pinned to the expected revision")
    experts = doc.get("experts")
    if not isinstance(experts, list) or not experts:
        raise ValueError("pilot manifest contains no experts")
    seen = set()
    for row in experts:
        key = (int(row["layer"]), int(row["expert"]))
        if key in seen:
            raise ValueError(f"duplicate pilot expert {key}")
        seen.add(key)
    bits = [int(x) for x in doc.get("bits", [])]
    if not bits or any(x < 2 or x > 8 for x in bits):
        raise ValueError("invalid pilot bit widths")
    if int(doc.get("group_size", 0)) <= 0:
        raise ValueError("invalid group_size")
    return doc


def expert_shards(weight_map: dict[str, str], layer: int, expert: int) -> tuple[str, ...]:
    shards = []
    for part in ("gate_proj", "up_proj", "down_proj"):
        name = PREFIX.format(layer=layer, expert=expert, part=part)
        try:
            shard = weight_map[name]
        except KeyError as exc:
            raise KeyError(f"checkpoint index missing {name}") from exc
        if shard not in shards:
            shards.append(shard)
    return tuple(sorted(shards))


def build_plan(index: dict, manifest: dict, reservoirs: Path) -> list[dict]:
    wm = index["weight_map"]
    plan = []
    for row in manifest["experts"]:
        layer, expert = int(row["layer"]), int(row["expert"])
        reservoir = reservoirs / f"layer-{layer:02d}-expert-{expert:02d}.npz"
        plan.append(
            {
                **row,
                "layer": layer,
                "expert": expert,
                "reservoir": reservoir,
                "shards": expert_shards(wm, layer, expert),
            }
        )
    plan.sort(key=lambda x: (x["shards"], x["rank"], x["layer"], x["expert"]))
    return plan


def _download(repo: str, revision: str, model_dir: Path, shard: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            revision=revision,
            filename=shard,
            local_dir=str(model_dir),
        )
    )


def run(
    model_dir: Path,
    reservoirs: Path,
    pilot_manifest: Path,
    out_dir: Path,
    repo: str,
    revision: str,
    cleanup_downloaded: bool,
) -> dict:
    if revision != PINNED_REVISION:
        raise ValueError(f"refusing revision {revision}; expected {PINNED_REVISION}")
    manifest = validate_manifest(pilot_manifest)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    plan = build_plan(index, manifest, reservoirs)
    missing = [x for x in plan if not x["reservoir"].is_file()]
    if missing:
        keys = [f"L{x['layer']:02d}E{x['expert']:02d}" for x in missing]
        raise ValueError(f"pilot calibration did not observe required experts: {keys}")

    remaining = Counter(shard for row in plan for shard in row["shards"])
    downloaded_here: set[str] = set()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    bits = [int(x) for x in manifest["bits"]]
    group_size = int(manifest["group_size"])

    try:
        for i, row in enumerate(plan, 1):
            for shard in row["shards"]:
                p = model_dir / shard
                if not p.is_file():
                    print(f"Q0_DOWNLOAD shard={shard}", flush=True)
                    _download(repo, revision, model_dir, shard)
                    downloaded_here.add(shard)
            key = f"L{row['layer']:02d}E{row['expert']:02d}"
            print(
                f"Q0_MEASURE {i}/{len(plan)} expert={key} rank={row['rank']} "
                f"bits={bits} group={group_size} shards={','.join(row['shards'])}",
                flush=True,
            )
            result = measure(
                model_dir,
                row["reservoir"],
                bits,
                group_size,
                revision,
            )
            result["pilot"] = {
                "rank": int(row["rank"]),
                "c2_score": float(row["c2_score"]),
                "boundary_role": row["boundary_role"],
            }
            out = out_dir / f"{key}.json"
            out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            by_bits = {int(c["bits"]): c for c in result["candidates"]}
            print(
                "Q0_RESULT " + key + " " + " ".join(
                    f"q{b}_rel_l2={by_bits[b]['relative_l2']:.9g} q{b}_cos={by_bits[b]['output_cosine']:.9g}"
                    for b in bits
                ),
                flush=True,
            )
            results.append(result)
            for shard in row["shards"]:
                remaining[shard] -= 1
                if cleanup_downloaded and remaining[shard] == 0 and shard in downloaded_here:
                    p = model_dir / shard
                    if p.is_file():
                        print(f"Q0_DELETE_SOURCE shard={shard}", flush=True)
                        p.unlink()
                    downloaded_here.remove(shard)
    finally:
        if cleanup_downloaded:
            for shard in sorted(downloaded_here):
                p = model_dir / shard
                if p.is_file():
                    p.unlink()

    summary_rows = []
    for r in sorted(results, key=lambda x: int(x["pilot"]["rank"])):
        candidates = {int(c["bits"]): c for c in r["candidates"]}
        summary_rows.append(
            {
                "layer": int(r["metadata"]["layer"]),
                "expert": int(r["metadata"]["expert"]),
                "rank": int(r["pilot"]["rank"]),
                "c2_score": float(r["pilot"]["c2_score"]),
                "boundary_role": r["pilot"]["boundary_role"],
                "tokens": int(r["tokens"]),
                "candidates": {
                    str(b): {
                        "relative_l2": float(candidates[b]["relative_l2"]),
                        "output_cosine": float(candidates[b]["output_cosine"]),
                        "output_mse": float(candidates[b]["output_mse"]),
                        "output_max_abs": float(candidates[b]["output_max_abs"]),
                        "projected_total_bytes_f16_scales": int(candidates[b]["projected_total_bytes_f16_scales"]),
                    }
                    for b in bits
                },
            }
        )

    summary = {
        "schema": "kimi-compression-q0-pilot-result-v1",
        "source_revision": revision,
        "pilot_manifest": str(pilot_manifest),
        "expert_count": len(summary_rows),
        "bits": bits,
        "group_size": group_size,
        "projection_only": True,
        "native_low_bit_format_implemented": False,
        "end_to_end_low_bit_quality_tested": False,
        "experts": summary_rows,
    }
    (out_dir / "q0-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("reservoirs", type=Path)
    ap.add_argument("pilot_manifest", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=PINNED_REVISION)
    ap.add_argument("--cleanup-downloaded", action="store_true")
    args = ap.parse_args()
    summary = run(
        args.model_dir,
        args.reservoirs,
        args.pilot_manifest,
        args.out_dir,
        args.repo,
        args.revision,
        args.cleanup_downloaded,
    )
    print(
        f"Q0_DONE experts={summary['expert_count']} bits={','.join(map(str, summary['bits']))} "
        "projection_only=yes native_low_bit=no",
        flush=True,
    )


if __name__ == "__main__":
    main()
