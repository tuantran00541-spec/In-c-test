#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "moonshotai/Kimi-VL-A3B-Instruct"
REVISION = "398eede0903cd983a2bfa0cc634e9ac1d843f375"
FRONTEND_FILES = (
    "tiktoken.model",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)

def run(*args: object) -> None:
    cmd = [str(x) for x in args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def download(repo: str, revision: str, root: Path, name: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo,
            revision=revision,
            filename=name,
            local_dir=str(root),
        )
    )

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download and pack the validated low-RAM Kimi-VL Q8 runtime"
    )
    ap.add_argument("work_dir", type=Path,
                    help="temporary/source-shard directory (safe to reuse for resume)")
    ap.add_argument("out_dir", type=Path,
                    help="final packed runtime directory")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION,
                    help="checkpoint revision; default is the validated pinned commit")
    ap.add_argument("--keep-source-shards", action="store_true",
                    help="do not delete consumed safetensor shards after packing")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    tools = root / "tools"
    work = args.work_dir.resolve()
    out = args.out_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    print(f"repository={args.repo}")
    print(f"revision={args.revision}")
    print(f"work_dir={work}")
    print(f"out_dir={out}")

    config_path = download(args.repo, args.revision, work, "config.json")
    index_path = download(args.repo, args.revision, work, "model.safetensors.index.json")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    vision_shards = sorted(
        {
            shard
            for name, shard in weight_map.items()
            if name.startswith("vision_tower.") or name.startswith("multi_modal_projector.")
        }
    )
    if not vision_shards:
        raise SystemExit("checkpoint index contains no vision/projector tensors")
    print("vision/projector source shards:", ", ".join(vision_shards))
    for shard in vision_shards:
        download(args.repo, args.revision, work, shard)

    run(sys.executable, tools / "pack_vision.py", work, out)

    text_cmd: list[object] = [
        sys.executable,
        tools / "pack_full_text.py",
        work,
        out,
        "--repo",
        args.repo,
        "--revision",
        args.revision,
        "--expert-format",
        "q8",
    ]
    if args.keep_source_shards:
        text_cmd.append("--keep-source-shards")
    run(*text_cmd)

    run(
        sys.executable,
        tools / "fetch_tokenizer.py",
        out,
        "--repo",
        args.repo,
        "--revision",
        args.revision,
    )

    shutil.copy2(config_path, out / "config.json")
    shutil.copy2(index_path, out / "source_model.safetensors.index.json")
    (out / "SOURCE_REVISION.txt").write_text(args.revision + "\n", encoding="ascii")

    expected = (
        "trunk.bin", "trunk.idx", "experts.bin", "experts.idx",
        "vision.bin", "vision.idx", *FRONTEND_FILES,
        "config.json", "source_model.safetensors.index.json", "SOURCE_REVISION.txt",
    )
    missing = [name for name in expected if not (out / name).is_file()]
    if missing:
        raise SystemExit("runtime preparation incomplete; missing: " + ", ".join(missing))

    sizes = {
        name: (out / name).stat().st_size
        for name in ("trunk.bin", "experts.bin", "vision.bin")
    }
    total = sum(sizes.values())
    print("\nPASS: pinned Kimi-VL Q8 runtime is ready")
    print(f"  revision: {args.revision}")
    for name, size in sizes.items():
        print(f"  {name}: {size / 1024**3:.3f} GiB")
    print(f"  packed weight total: {total / 1024**3:.3f} GiB")
    print(f"  runtime directory: {out}")
    if not args.keep_source_shards:
        print("  consumed source shards were deleted as packing progressed")
    print("\nNext: python tools/kvl_vl_chat.py <runtime_dir> <image> <prompt> --temperature 0")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
