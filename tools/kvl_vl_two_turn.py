#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from kimi_image import write_patches
from kimi_tokenizer import build_encoding, decode_generated, encode_image_chat
from kvl_memory_plan import MIB, as_mib, planned_text_breakdown

EOS_ID = 163585
IM_END_ID = 163586
STOP_IDS = {EOS_ID, IM_END_ID}

@dataclass
class TurnResult:
    ids: list[int]
    text: str
    first: float
    avg_next: float
    total: float

def _exe(name: str) -> str:
    if os.name == "nt":
        release = Path("build") / "Release" / f"{name}.exe"
        return str(release if release.exists() else Path("build") / f"{name}.exe")
    return str(Path("build") / name)

def _body_before_stop(ids: list[int]) -> list[int]:
    out: list[int] = []
    for token in ids:
        if token in STOP_IDS:
            break
        out.append(token)
    return out

def _plan_or_die(prompt_n: int, max_new: int, cache_mib: int, ram_mib: int, label: str) -> None:
    plan = planned_text_breakdown(prompt_n, max_new, cache_mib)
    if plan["total"] > ram_mib * MIB:
        raise SystemExit(
            f"{label} RAM plan rejected: {as_mib(plan['total']):.1f} MiB > {ram_mib} MiB"
        )
    print(
        f"{label}_PLAN prompt_tokens={prompt_n} total_mib={as_mib(plan['total']):.2f} "
        f"state_mib={as_mib(plan['compressed_state']):.2f} "
        f"seq_ws_mib={as_mib(plan['sequence_workspace']):.2f} "
        f"mla_ws_mib={as_mib(plan['streaming_mla_workspace']):.2f} "
        f"cache_mib={as_mib(plan['expert_cache']):.2f}",
        flush=True,
    )

def _run_turn(
    *,
    label: str,
    model: Path,
    ids_path: Path,
    media_path: Path,
    prompt_ids: list[int],
    generate_binary: str,
    cache_mib: int,
    max_new: int,
    temperature: float,
    seed: int,
    encoding,
    show_tokens: bool,
) -> TurnResult:
    ids_path.write_text("\n".join(map(str, prompt_ids)) + "\n", encoding="ascii")
    cmd = [
        generate_binary,
        str(model / "trunk.bin"), str(model / "trunk.idx"),
        str(model / "experts.bin"), str(model / "experts.idx"),
        str(ids_path), str(media_path), str(cache_mib * MIB),
        str(max_new), str(temperature), str(seed),
    ]

    generated: list[int] = []
    token_times: list[float] = []
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("TOKEN "):
            continue
        now = time.monotonic()
        token = int(line.split()[1])
        generated.append(token)
        token_times.append(now)
        if show_tokens:
            prev = token_times[-2] if len(token_times) > 1 else start
            print(f"{label}_TOKEN id={token} dt={now-prev:.3f}s", flush=True)
    rc = proc.wait()
    end = time.monotonic()
    if rc != 0:
        raise SystemExit(f"{label} generator failed rc={rc}")

    first = token_times[0] - start if token_times else 0.0
    intervals = [b - a for a, b in zip(token_times, token_times[1:])]
    avg_next = sum(intervals) / len(intervals) if intervals else 0.0
    text = decode_generated(encoding, generated, STOP_IDS)
    return TurnResult(generated, text, first, avg_next, end - start)

def main() -> int:
    ap = argparse.ArgumentParser(description="Two-turn Vietnamese Kimi-VL benchmark")
    ap.add_argument("model_dir")
    ap.add_argument("image")
    ap.add_argument("prompt1")
    ap.add_argument("prompt2")
    ap.add_argument("--system", default="Bạn là một trợ lý hữu ích. Luôn trả lời bằng tiếng Việt.")
    ap.add_argument("--vision-binary", default=_exe("kvl_vision"))
    ap.add_argument("--generate-binary", default=_exe("kvl_generate_vl"))
    ap.add_argument("--cache-mib", type=int, default=512)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--max-new-turn1", type=int, default=32)
    ap.add_argument("--max-new-turn2", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show-tokens", action="store_true")
    args = ap.parse_args()

    if min(args.cache_mib, args.ram_mib, args.max_new_turn1, args.max_new_turn2) <= 0:
        raise SystemExit("cache/RAM/max-new values must be positive")
    if args.temperature < 0:
        raise SystemExit("temperature must be >= 0")

    model = Path(args.model_dir)
    image = Path(args.image)
    required = [
        "trunk.bin", "trunk.idx", "experts.bin", "experts.idx",
        "vision.bin", "vision.idx", "tiktoken.model", "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    missing = [n for n in required if not (model / n).is_file()]
    if missing:
        raise SystemExit("missing runtime files: " + ", ".join(missing))
    if not image.is_file():
        raise SystemExit(f"image not found: {image}")

    enc, _, special = build_encoding(model)
    media_pad_id = special.get("<|media_pad|>")
    im_end_id = special.get("<|im_end|>")
    if media_pad_id is None or im_end_id is None:
        raise SystemExit("required chat special token missing")
    if im_end_id != IM_END_ID:
        raise SystemExit(f"unexpected <|im_end|> id {im_end_id}")

    wall_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="kvl-two-turn-") as td:
        temp = Path(td)
        patches_path = temp / "patches.f32"
        media_path = temp / "media.f32"

        patch_start = time.monotonic()
        gh, gw = write_patches(model, image, patches_path)
        patch_seconds = time.monotonic() - patch_start
        media_tokens = (gh // 2) * (gw // 2)

        turn1_prompt = encode_image_chat(enc, args.prompt1, media_tokens, args.system)
        actual_media = sum(1 for x in turn1_prompt if x == media_pad_id)
        if actual_media != media_tokens:
            raise SystemExit(
                f"media token mismatch: expected {media_tokens}, encoded {actual_media}"
            )
        _plan_or_die(
            len(turn1_prompt), args.max_new_turn1, args.cache_mib, args.ram_mib, "TURN1"
        )

        vision_start = time.monotonic()
        vr = subprocess.run([
            args.vision_binary,
            str(model / "vision.bin"), str(model / "vision.idx"),
            str(patches_path), str(gh), str(gw), str(media_path),
        ])
        vision_seconds = time.monotonic() - vision_start
        if vr.returncode != 0:
            return vr.returncode

        turn1 = _run_turn(
            label="TURN1", model=model, ids_path=temp / "turn1.ids", media_path=media_path,
            prompt_ids=turn1_prompt, generate_binary=args.generate_binary,
            cache_mib=args.cache_mib, max_new=args.max_new_turn1,
            temperature=args.temperature, seed=args.seed, encoding=enc,
            show_tokens=args.show_tokens,
        )

        turn1_body = _body_before_stop(turn1.ids)
        followup_markup = (
            f"<|im_user|>user<|im_middle|>{args.prompt2}<|im_end|>"
            f"<|im_assistant|>assistant<|im_middle|>"
        )
        followup_ids = enc.encode(followup_markup, allowed_special="all")
        turn2_prompt = turn1_prompt + turn1_body + [IM_END_ID] + followup_ids
        _plan_or_die(
            len(turn2_prompt), args.max_new_turn2, args.cache_mib, args.ram_mib, "TURN2"
        )

        turn2 = _run_turn(
            label="TURN2", model=model, ids_path=temp / "turn2.ids", media_path=media_path,
            prompt_ids=turn2_prompt, generate_binary=args.generate_binary,
            cache_mib=args.cache_mib, max_new=args.max_new_turn2,
            temperature=args.temperature, seed=args.seed, encoding=enc,
            show_tokens=args.show_tokens,
        )

    wall_total = time.monotonic() - wall_start
    turn1_stopped = bool(turn1.ids and turn1.ids[-1] in STOP_IDS)
    turn2_stopped = bool(turn2.ids and turn2.ids[-1] in STOP_IDS)

    print(f"CHAT_MODE=turn2_full_history_reprefill vision_reused=yes persistent_state=no")
    print(f"GRID={gh}x{gw} MEDIA_TOKENS={media_tokens}")
    print(f"PROMPT1={args.prompt1}")
    print(f"TURN1_ANSWER={turn1.text}")
    print("TURN1_IDS=" + " ".join(map(str, turn1.ids)))
    print(
        f"TURN1_TIMING first_token={turn1.first:.3f}s avg_next={turn1.avg_next:.3f}s "
        f"text_total={turn1.total:.3f}s generated={len(turn1.ids)} stopped={turn1_stopped}"
    )
    print(f"PROMPT2={args.prompt2}")
    print(f"TURN2_ANSWER={turn2.text}")
    print("TURN2_IDS=" + " ".join(map(str, turn2.ids)))
    print(
        f"TURN2_TIMING first_token={turn2.first:.3f}s avg_next={turn2.avg_next:.3f}s "
        f"text_total={turn2.total:.3f}s generated={len(turn2.ids)} stopped={turn2_stopped}"
    )
    print(
        f"CHAT_TIMING image_preprocess={patch_seconds:.3f}s vision={vision_seconds:.3f}s "
        f"text_both_turns={turn1.total + turn2.total:.3f}s wall_total={wall_total:.3f}s"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
