#!/usr/bin/env python3
"""Fetch the small official tokenizer assets into an already-packed runtime directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "moonshotai/Kimi-VL-A3B-Instruct"
FILES = ("tiktoken.model", "tokenizer_config.json", "generation_config.json", "chat_template.jinja")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = Path(hf_hub_download(args.repo, name))
        dst = out / name
        shutil.copy2(src, dst)
        print(f"tokenizer asset: {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
