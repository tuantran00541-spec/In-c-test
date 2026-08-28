#!/usr/bin/env python3
"""Resolve official Kimi-VL vision/projector tensors to checkpoint shards without loading weights."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "moonshotai/Kimi-VL-A3B-Instruct"
PREFIXES = ("vision_tower.", "multi_modal_projector.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    idx = Path(hf_hub_download(repo_id=args.repo, revision=args.revision,
                               filename="model.safetensors.index.json",
                               local_dir=str(args.out_dir)))
    wm = json.loads(idx.read_text())["weight_map"]
    names = sorted(k for k in wm if k.startswith(PREFIXES))
    if not names:
        raise SystemExit("no vision/projector tensors found")
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_shard[wm[name]].append(name)
    print(f"vision/projector tensors={len(names)} shards={len(by_shard)}")
    for shard in sorted(by_shard):
        print(f"SHARD {shard} tensors={len(by_shard[shard])}")
        counts = Counter()
        for n in by_shard[shard]:
            if ".encoder.blocks." in n:
                p = n.split(".encoder.blocks.", 1)[1]
                counts["block_" + p.split(".", 1)[0]] += 1
            elif n.startswith("vision_tower.patch_embed"):
                counts["patch_embed"] += 1
            elif n.startswith("vision_tower.encoder.final_layernorm"):
                counts["final_norm"] += 1
            elif n.startswith("multi_modal_projector"):
                counts["projector"] += 1
            else:
                counts["other"] += 1
        print("  groups:", " ".join(f"{k}={v}" for k,v in sorted(counts.items())))
    print("\nTENSORS")
    for n in names:
        print(f"{wm[n]}\t{n}")


if __name__ == "__main__":
    main()
