#!/usr/bin/env python3
"""Text-only V7 frontend for the C generation core.

Example:
  python tools/kvl_chat.py D:/models/Kimi-VL-packed "Xin chao" --binary build/kvl_generate.exe
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from kimi_tokenizer import build_encoding, decode_generated, encode_chat

EOS_ID = 163585
IM_END_ID = 163586


def default_binary() -> str:
    return str(Path("build") / ("kvl_generate.exe" if os.name == "nt" else "kvl_generate"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run text-only Kimi-VL generation through the C low-RAM runtime")
    ap.add_argument("model_dir", help="packed runtime directory containing trunk/expert stores + tokenizer assets")
    ap.add_argument("prompt")
    ap.add_argument("--system", default="You are a helpful assistant")
    ap.add_argument("--binary", default=default_binary())
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="0 = greedy; official generation_config uses 0.2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show-tokens", action="store_true")
    args = ap.parse_args()

    model = Path(args.model_dir)
    required = [
        "trunk.bin", "trunk.idx", "experts.bin", "experts.idx",
        "tiktoken.model", "tokenizer_config.json",
    ]
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise SystemExit("missing runtime files: " + ", ".join(missing))

    enc, _, _ = build_encoding(model)
    prompt_ids = encode_chat(enc, args.prompt, args.system)
    print(f"[kvl] prompt tokens={len(prompt_ids)} cache={args.cache_mib} MiB "
          f"max_new={args.max_new} temperature={args.temperature}", file=sys.stderr)
    if args.show_tokens:
        print("[kvl] prompt ids:", " ".join(map(str, prompt_ids)), file=sys.stderr)

    with tempfile.NamedTemporaryFile("w", encoding="ascii", delete=False, suffix=".ids") as f:
        ids_path = Path(f.name)
        f.write("\n".join(map(str, prompt_ids)))
        f.write("\n")

    cmd = [
        args.binary,
        str(model / "trunk.bin"), str(model / "trunk.idx"),
        str(model / "experts.bin"), str(model / "experts.idx"),
        str(ids_path), str(args.cache_mib * 1024 * 1024), str(args.max_new),
        str(args.temperature), str(args.seed),
    ]
    generated: list[int] = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("TOKEN "):
                token = int(line.split()[1])
                generated.append(token)
                if args.show_tokens:
                    print(f"[kvl] token={token}", file=sys.stderr)
        rc = proc.wait()
        if rc != 0:
            print(f"[kvl] C runtime exited with code {rc}", file=sys.stderr)
            return rc
    finally:
        try:
            ids_path.unlink()
        except OSError:
            pass

    text = decode_generated(enc, generated, {EOS_ID, IM_END_ID})
    print(text)
    if args.show_tokens:
        print("[kvl] generated ids:", " ".join(map(str, generated)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
