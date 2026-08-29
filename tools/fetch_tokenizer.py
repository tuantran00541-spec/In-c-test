#!/usr/bin/env python3
"""Fetch small official tokenizer/frontend assets into an already-packed runtime directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "moonshotai/Kimi-VL-A3B-Instruct"
FILES = (
    "tiktoken.model",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default="main",
                    help="Hugging Face revision/commit used for all frontend assets")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = Path(hf_hub_download(repo_id=args.repo, revision=args.revision, filename=name))
        dst = out / name
        shutil.copy2(src, dst)
        print(f"runtime asset: {dst} ({dst.stat().st_size} bytes) revision={args.revision}")


if __name__ == "__main__":
    main()
