#!/usr/bin/env python3
"""Text frontend for the Kimi-VL low-RAM C generation core.

V8 adds a conservative total-working-set planner around the runtime. The C engine still owns
its expert-cache hard cap; this frontend additionally refuses prompt/cache configurations whose
known peak allocations would exceed --ram-mib.

Example:
  python tools/kvl_chat.py D:/models/Kimi-VL-packed "Xin chao" \
      --binary build/Release/kvl_generate.exe --ram-mib 4096
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from kimi_tokenizer import build_encoding, decode_generated, encode_chat

EOS_ID = 163585
IM_END_ID = 163586

# Released Kimi-VL text dimensions. These constants are deliberately conservative and are
# part of the V8 memory contract rather than an estimate based on free system RAM.
HIDDEN = 2048
LAYERS = 27
KV_LATENT_PLUS_ROPE = 512 + 64
GLOBAL_BF16_BYTES = 2 * (163840 * HIDDEN * 2) + HIDDEN * 2  # embed + lm_head + final norm
FIXED_RUNTIME_SAFETY = 256 * 1024 * 1024
# Batch prefill currently holds six [S,H] FP32 outer buffers. kvl_mla_prefill_bf16 also holds
# expanded Q/K/V at 32 KiB/token. Add 8 KiB/token margin for temporary vectors/allocator slack.
BATCH_PREFILL_BYTES_PER_TOKEN = (6 * HIDDEN * 4) + (32 * 1024) + (8 * 1024)


def default_binary() -> str:
    if os.name == "nt":
        release = Path("build") / "Release" / "kvl_generate.exe"
        return str(release if release.exists() else Path("build") / "kvl_generate.exe")
    return str(Path("build") / "kvl_generate")


def planned_bytes(prompt_tokens: int, max_new: int, cache_mib: int) -> dict[str, int]:
    capacity = prompt_tokens + max_new
    compressed_state = LAYERS * capacity * KV_LATENT_PLUS_ROPE * 4
    batch_prefill = prompt_tokens * BATCH_PREFILL_BYTES_PER_TOKEN
    cache = cache_mib * 1024 * 1024
    parts = {
        "globals": GLOBAL_BF16_BYTES,
        "expert_cache": cache,
        "compressed_state": compressed_state,
        "batch_prefill_peak": batch_prefill,
        "runtime_safety": FIXED_RUNTIME_SAFETY,
    }
    parts["planned_peak"] = sum(parts.values())
    return parts


def mib(n: int) -> float:
    return n / 1048576.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Kimi-VL text generation through the C low-RAM runtime")
    ap.add_argument("model_dir", help="packed runtime directory containing trunk/expert stores + tokenizer assets")
    ap.add_argument("prompt")
    ap.add_argument("--system", default="You are a helpful assistant")
    ap.add_argument("--binary", default=default_binary())
    ap.add_argument("--cache-mib", type=int, default=512,
                    help="hard routed-expert cache budget")
    ap.add_argument("--ram-mib", type=int, default=4096,
                    help="V8 total known-working-set budget; configuration is rejected if the conservative plan exceeds it")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="0 = greedy; official generation_config uses 0.2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show-tokens", action="store_true")
    args = ap.parse_args()

    if args.cache_mib <= 0 or args.ram_mib <= 0 or args.max_new <= 0:
        raise SystemExit("cache, RAM budget and max-new must be positive")
    if args.temperature < 0:
        raise SystemExit("temperature must be >= 0")

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
    plan = planned_bytes(len(prompt_ids), args.max_new, args.cache_mib)
    budget = args.ram_mib * 1024 * 1024
    if plan["planned_peak"] > budget:
        detail = ", ".join(f"{k}={mib(v):.1f}MiB" for k, v in plan.items() if k != "planned_peak")
        raise SystemExit(
            f"RAM plan rejected: planned_peak={mib(plan['planned_peak']):.1f}MiB > "
            f"budget={args.ram_mib}MiB ({detail}). Reduce prompt/max-new/cache or raise --ram-mib."
        )

    print(
        f"[kvl] prompt tokens={len(prompt_ids)} cache={args.cache_mib} MiB "
        f"RAM plan={mib(plan['planned_peak']):.1f}/{args.ram_mib} MiB "
        f"max_new={args.max_new} temperature={args.temperature}",
        file=sys.stderr,
    )
    if args.show_tokens:
        print("[kvl] prompt ids:", " ".join(map(str, prompt_ids)), file=sys.stderr)
        for k, v in plan.items():
            if k != "planned_peak":
                print(f"[kvl] RAM {k}={mib(v):.2f} MiB", file=sys.stderr)

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
    token_times: list[float] = []
    start = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("TOKEN "):
                now = time.monotonic()
                token = int(line.split()[1])
                generated.append(token)
                token_times.append(now)
                if args.show_tokens:
                    dt = now - (token_times[-2] if len(token_times) > 1 else start)
                    print(f"[kvl] token={token} dt={dt:.3f}s", file=sys.stderr)
        rc = proc.wait()
        end = time.monotonic()
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
    if token_times:
        first = token_times[0] - start
        decode_intervals = [b - a for a, b in zip(token_times, token_times[1:])]
        avg_decode = sum(decode_intervals) / len(decode_intervals) if decode_intervals else 0.0
        print(
            f"[kvl] timing first_token={first:.3f}s "
            f"avg_next={avg_decode:.3f}s total={end-start:.3f}s generated={len(generated)}",
            file=sys.stderr,
        )
    if args.show_tokens:
        print("[kvl] generated ids:", " ".join(map(str, generated)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
