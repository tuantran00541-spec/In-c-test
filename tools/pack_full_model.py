#!/usr/bin/env python3
"""Build a complete V9 runtime directory from a local official Kimi-VL checkpoint.

Expected input is the normal Hugging Face snapshot with config.json,
model.safetensors.index.json and all seven shard files. The output intentionally contains
runtime-packed weights and small frontend assets only; neither source nor packed weights are
meant to be committed to git.

Routed experts default to the release BF16 format. Pass ``--expert-format q8`` to use the
validated symmetric per-row int8 routed-expert format while keeping trunk, routers, shared
experts and vision weights BF16.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack the complete Kimi-VL V9 low-RAM runtime")
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--expert-format", choices=("bf16", "q8"), default="bf16",
                    help="routed expert storage format; default: bf16")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    tools = root / "tools"
    src = Path(args.model_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    required = ["config.json", "model.safetensors.index.json"] + [
        f"model-{i:05d}-of-00007.safetensors" for i in range(1, 8)
    ]
    missing = [n for n in required if not (src / n).is_file()]
    if missing:
        raise SystemExit("official checkpoint is incomplete; missing: " + ", ".join(missing))

    py = sys.executable
    layers = ",".join(str(i) for i in range(27))
    run(py, str(tools / "pack_trunk.py"), str(src), str(out),
        "--layers", layers, "--include-globals")
    expert_packer = tools / ("pack_experts_q8.py" if args.expert_format == "q8" else "pack_experts.py")
    run(py, str(expert_packer), str(src), str(out))
    run(py, str(tools / "pack_vision.py"), str(src), str(out))
    run(py, str(tools / "fetch_tokenizer.py"), str(out))

    # Keep the exact model config alongside the packed stores for inspection/version checks.
    shutil.copy2(src / "config.json", out / "config.json")
    shutil.copy2(src / "model.safetensors.index.json", out / "source_model.safetensors.index.json")

    expected = [
        "trunk.bin", "trunk.idx", "experts.bin", "experts.idx", "vision.bin", "vision.idx",
        "tiktoken.model", "tokenizer_config.json", "generation_config.json",
        "chat_template.jinja", "preprocessor_config.json", "config.json",
    ]
    missing_out = [n for n in expected if not (out / n).is_file()]
    if missing_out:
        raise SystemExit("packer finished but runtime is incomplete: " + ", ".join(missing_out))

    total = sum((out / n).stat().st_size for n in expected if n.endswith((".bin", ".idx")))
    print(f"PASS: complete V9 runtime at {out}")
    print(f"routed expert format={args.expert_format}")
    print(f"packed binary/index bytes={total} ({total/1024**3:.3f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
