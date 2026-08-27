#!/usr/bin/env python3
"""Download only checkpoint shards needed for one complete Kimi-VL decoder layer.

V3 validates a full decoder layer, so unlike fetch_real_v2.py this resolves every tensor
under `language_model.model.layers.<L>.*`: input/post-attention norms, MLA/self-attention,
router/shared experts, and routed experts. It still avoids snapshotting the full model.
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
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(out_dir),
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"repo      : {args.repo}@{args.revision}")
    print(f"layer     : {args.layer}")
    print(f"output    : {args.out_dir.resolve()}")

    fetch(args.repo, "config.json", args.out_dir, args.revision)
    index_path = fetch(args.repo, "model.safetensors.index.json", args.out_dir, args.revision)
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    prefix = f"language_model.model.layers.{args.layer}."
    names = sorted(n for n in weight_map if n.startswith(prefix))
    if not names:
        raise SystemExit(f"No tensors found for prefix {prefix!r}")

    shards = sorted({weight_map[n] for n in names})
    attn = [n for n in names if ".self_attn." in n]
    norms = [n for n in names if "layernorm.weight" in n]
    routed = [n for n in names if ".mlp.experts." in n]
    shared = [n for n in names if ".mlp.shared_experts." in n]
    router = [n for n in names if ".mlp.gate." in n]

    print(f"matched   : {len(names)} layer tensors")
    print(f"  attn    : {len(attn)}")
    print(f"  norms   : {len(norms)}")
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
    (args.out_dir / "real_v3_download_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )

    if args.metadata_only:
        print("metadata-only: large shard download skipped")
        return

    for i, shard in enumerate(shards, 1):
        print(f"[{i}/{len(shards)}] downloading {shard}")
        fetch(args.repo, shard, args.out_dir, args.revision)
    print("real V3 checkpoint subset ready")


if __name__ == "__main__":
    main()
