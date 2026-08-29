#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from kimi_image import write_patches
from kimi_tokenizer import build_encoding, decode_generated, encode_image_chat
from kvl_memory_plan import MIB, as_mib, planned_text_breakdown, planned_text_bytes

EOS_ID = 163585
IM_END_ID = 163586

def _exe(name: str) -> str:
    if os.name == "nt":
        release = Path("build") / "Release" / f"{name}.exe"
        return str(release if release.exists() else Path("build") / f"{name}.exe")
    return str(Path("build") / name)

def main() -> int:
    ap = argparse.ArgumentParser(description="Run image chat through MoonViT C + Kimi text C")
    ap.add_argument("model_dir", help="packed runtime directory")
    ap.add_argument("image")
    ap.add_argument("prompt")
    ap.add_argument("--system", default="You are a helpful assistant")
    ap.add_argument("--vision-binary", default=_exe("kvl_vision"))
    ap.add_argument("--generate-binary", default=_exe("kvl_generate_vl"))
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show-tokens", action="store_true")
    ap.add_argument(
        "--prompt-ids-out",
        help="optional research/debug path to persist the exact multimodal prompt token ids",
    )
    args = ap.parse_args()

    if args.cache_mib <= 0 or args.ram_mib <= 0 or args.max_new <= 0:
        raise SystemExit("cache, RAM budget and max-new must be positive")
    if args.temperature < 0:
        raise SystemExit("temperature must be >= 0")

    model = Path(args.model_dir)
    required = [
        "trunk.bin", "trunk.idx", "experts.bin", "experts.idx",
        "vision.bin", "vision.idx", "tiktoken.model", "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    missing = [n for n in required if not (model / n).is_file()]
    if missing:
        raise SystemExit("missing runtime files: " + ", ".join(missing))
    if not Path(args.image).is_file():
        raise SystemExit(f"image not found: {args.image}")

    enc, _, special = build_encoding(model)
    media_pad_id = special.get("<|media_pad|>")
    if media_pad_id is None:
        raise SystemExit("tokenizer has no <|media_pad|> special token")

    with tempfile.TemporaryDirectory(prefix="kvl-v9-") as td:
        temp = Path(td)
        patches_path = temp / "patches.f32"
        media_path = temp / "media.f32"
        ids_path = temp / "prompt.ids"

        gh, gw = write_patches(model, args.image, patches_path)
        media_tokens = (gh // 2) * (gw // 2)
        prompt_ids = encode_image_chat(enc, args.prompt, media_tokens, args.system)
        actual_media = sum(1 for x in prompt_ids if x == media_pad_id)
        if actual_media != media_tokens:
            raise SystemExit(
                f"media token expansion mismatch: expected {media_tokens}, encoded {actual_media}"
            )
        if args.prompt_ids_out:
            research_ids = Path(args.prompt_ids_out)
            research_ids.parent.mkdir(parents=True, exist_ok=True)
            research_ids.write_text("\n".join(map(str, prompt_ids)) + "\n", encoding="ascii")

        plan = planned_text_breakdown(len(prompt_ids), args.max_new, args.cache_mib)
        planned = plan["total"]
        budget = args.ram_mib * MIB
        if planned > budget:
            raise SystemExit(
                f"RAM plan rejected: text phase {as_mib(planned):.1f} MiB > "
                f"budget {args.ram_mib} MiB; state={as_mib(plan['compressed_state']):.1f} "
                f"seq_ws={as_mib(plan['sequence_workspace']):.1f} "
                f"mla_ws={as_mib(plan['streaming_mla_workspace']):.1f} "
                f"cache={as_mib(plan['expert_cache']):.1f}; reduce cache/max-new/context or "
                f"raise --ram-mib"
            )

        print(
            f"[kvl-vl] grid={gh}x{gw} media_tokens={media_tokens} prompt_tokens={len(prompt_ids)} "
            f"text_RAM_plan={as_mib(planned):.1f}/{args.ram_mib} MiB "
            f"state={as_mib(plan['compressed_state']):.1f} "
            f"seq_ws={as_mib(plan['sequence_workspace']):.1f} "
            f"mla_ws={as_mib(plan['streaming_mla_workspace']):.1f} "
            f"cache={as_mib(plan['expert_cache']):.1f} "
            f"global={as_mib(plan['global_bf16']):.1f} "
            f"layer_peak={as_mib(plan['peak_layer_bf16']):.1f} "
            f"safety={as_mib(plan['misc_safety']):.1f} prefill=exact-head-streaming",
            file=sys.stderr,
        )

        t0 = time.monotonic()
        vision_cmd = [
            args.vision_binary,
            str(model / "vision.bin"), str(model / "vision.idx"),
            str(patches_path), str(gh), str(gw), str(media_path),
        ]
        vr = subprocess.run(vision_cmd)
        if vr.returncode != 0:
            return vr.returncode
        t_vision = time.monotonic() - t0

        ids_path.write_text("\n".join(map(str, prompt_ids)) + "\n", encoding="ascii")
        cmd = [
            args.generate_binary,
            str(model / "trunk.bin"), str(model / "trunk.idx"),
            str(model / "experts.bin"), str(model / "experts.idx"),
            str(ids_path), str(media_path), str(args.cache_mib * MIB),
            str(args.max_new), str(args.temperature), str(args.seed),
        ]

        generated: list[int] = []
        token_times: list[float] = []
        text_start = time.monotonic()
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
                    prev = token_times[-2] if len(token_times) > 1 else text_start
                    print(f"[kvl-vl] token={token} dt={now-prev:.3f}s", file=sys.stderr)
        rc = proc.wait()
        text_end = time.monotonic()
        if rc != 0:
            return rc

    text = decode_generated(enc, generated, {EOS_ID, IM_END_ID})
    print(text)
    first = token_times[0] - text_start if token_times else 0.0
    intervals = [b-a for a, b in zip(token_times, token_times[1:])]
    avg_next = sum(intervals) / len(intervals) if intervals else 0.0
    print(
        f"[kvl-vl] timing vision={t_vision:.3f}s first_text_token={first:.3f}s "
        f"avg_next={avg_next:.3f}s text_total={text_end-text_start:.3f}s "
        f"generated={len(generated)}",
        file=sys.stderr,
    )
    if args.show_tokens:
        print("[kvl-vl] generated ids:", " ".join(map(str, generated)), file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
