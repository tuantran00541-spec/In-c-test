#!/usr/bin/env python3
"""Download only the Kimi-VL checkpoint shards needed for one real V2 MoE-layer test.

This intentionally does NOT snapshot the full ~32.8 GB repository. It first downloads
config.json and model.safetensors.index.json, resolves which safetensors shards contain
`language_model.model.layers.<L>.mlp.*`, then downloads only those shards.

The resulting directory is directly consumable by:
  tools/pack_experts.py ... --layer <L>
  tools/dump_real_moe_reference.py ... --layer <L>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install with: pip install huggingface_hub"
    ) from exc

DEFAULT_REPO = "moonshotai/Kimi-VL-A3B-Instruct"


def fetch(repo_id: str, filename: str, out_dir: Path, revision: str) -> Path:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(out_dir),
    )
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument(
        "--metadata-only",
        action="store_true",
        help="resolve and print required shards without downloading the large safetensors files",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"repo      : {args.repo}@{args.revision}")
    print(f"layer     : {args.layer}")
    print(f"output    : {args.out_dir.resolve()}")

    fetch(args.repo, "config.json", args.out_dir, args.revision)
    index_path = fetch(
        args.repo, "model.safetensors.index.json", args.out_dir, args.revision
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]

    prefix = f"language_model.model.layers.{args.layer}.mlp."
    names = sorted(name for name in weight_map if name.startswith(prefix))
    if not names:
        raise SystemExit(f"No tensors found for prefix {prefix!r}; tensor naming may have changed")

    shards = sorted({weight_map[name] for name in names})
    routed = [n for n in names if ".experts." in n]
    shared = [n for n in names if ".shared_experts." in n]
    router = [n for n in names if ".gate." in n]

    print(f"matched   : {len(names)} MLP tensors")
    print(f"  routed  : {len(routed)}")
    print(f"  shared  : {len(shared)}")
    print(f"  router  : {len(router)}")
    print(f"shards    : {len(shards)}")
    for shard in shards:
        print(f"  - {shard}")

    plan = {
        "repo_id": args.repo,
        "revision": args.revision,
        "layer": args.layer,
        "tensor_prefix": prefix,
        "tensor_count": len(names),
        "shards": shards,
    }
    (args.out_dir / "real_v2_download_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )

    if args.metadata_only:
        print("metadata-only: large shard download skipped")
        return

    for i, shard in enumerate(shards, 1):
        print(f"[{i}/{len(shards)}] downloading {shard}")
        fetch(args.repo, shard, args.out_dir, args.revision)

    print("real V2 checkpoint subset ready")
    print("next:")
    print(
        f"  python tools/pack_experts.py {args.out_dir} {args.out_dir / 'packed-layer'} --layer {args.layer}"
    )
    print(
        f"  python tools/dump_real_moe_reference.py {args.out_dir} "
        f"{args.out_dir / 'packed-layer' / 'layer.fixture'} --layer {args.layer}"
    )


if __name__ == "__main__":
    main()
