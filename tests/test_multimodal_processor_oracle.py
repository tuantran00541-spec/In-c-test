#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kimi_image import preprocess_image  # noqa: E402
from kimi_tokenizer import build_encoding, encode_image_chat, format_image_chat  # noqa: E402

REPO = "moonshotai/Kimi-VL-A3B-Instruct"
SPECIAL_NAMES = [
    "<|im_start|>",
    "<|im_system|>",
    "<|im_user|>",
    "<|im_assistant|>",
    "<|im_middle|>",
    "<|media_start|>",
    "<|media_content|>",
    "<|media_pad|>",
    "<|media_end|>",
    "<|im_end|>",
]


def _ids(x) -> list[int]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        x = x[0]
    return [int(v) for v in x]


def _positions(ids: list[int], token_id: int) -> list[int]:
    return [i for i, v in enumerate(ids) if v == token_id]


def run_case(processor, model_dir: Path, image_path: Path, prompt: str, label: str) -> None:
    enc, _, special = build_encoding(model_dir)
    tokenizer = processor.tokenizer

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # This is the exact text seen before KimiVLProcessor expands the one Jinja media pad.
    before = tokenizer(rendered, return_attention_mask=True)
    before_ids = _ids(before["input_ids"])
    before_mask = _ids(before["attention_mask"])

    image = Image.open(image_path).convert("RGB")
    out = processor(images=[image], text=[rendered], return_tensors="pt")
    after_ids = _ids(out["input_ids"])
    after_mask = _ids(out["attention_mask"])
    grid = tuple(int(v) for v in out["image_grid_hws"][0].tolist())
    merge = processor.image_processor.merge_kernel_size
    media_tokens = (grid[0] * grid[1]) // (int(merge[0]) * int(merge[1]))

    # The official text model uses ordinary sequential positions. During generation, the
    # HF implementation derives them from attention_mask.cumsum(-1)-1; with this unpadded
    # single example that must be exactly 0..N-1.
    position_ids = np.cumsum(np.asarray(after_mask, dtype=np.int64)) - 1
    position_ids[np.asarray(after_mask) == 0] = 1
    expected_positions = np.arange(len(after_ids), dtype=np.int64)
    assert np.array_equal(position_ids, expected_positions), (position_ids, expected_positions)

    # Compare the lightweight runtime frontend against the real AutoProcessor, not a hand-made
    # media expansion. Before expansion, the official Jinja template contains exactly one pad.
    custom_before = format_image_chat(prompt, 1)
    assert custom_before == rendered, (custom_before, rendered)
    custom_after_ids = encode_image_chat(enc, prompt, media_tokens)
    assert custom_after_ids == after_ids, (custom_after_ids, after_ids)
    assert before_mask == [1] * len(before_ids)
    assert after_mask == [1] * len(after_ids)

    # Re-check the released image frontend on the actual regression image too.
    custom_patches, custom_grid = preprocess_image(model_dir, image_path)
    official_patches = out["pixel_values"].detach().cpu().float().numpy().reshape(-1, 3 * 14 * 14)
    assert tuple(custom_grid) == grid, (custom_grid, grid)
    assert custom_patches.shape == official_patches.shape
    diff = custom_patches.astype(np.float64) - official_patches.astype(np.float64)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    rms = float(math.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    assert max_abs == 0.0, max_abs

    media_id = int(tokenizer.convert_tokens_to_ids("<|media_pad|>"))
    before_media = _positions(before_ids, media_id)
    after_media = _positions(after_ids, media_id)
    assert len(before_media) == 1, before_media
    assert len(after_media) == media_tokens, (len(after_media), media_tokens)

    print(f"===== {label} =====")
    print(f"image={image_path.name} size={image.size[0]}x{image.size[1]} grid={grid[0]}x{grid[1]} media_tokens={media_tokens}")
    print("--- rendered chat before processor expansion ---")
    print(rendered)
    print("--- token counts ---")
    print(f"before={len(before_ids)} after={len(after_ids)} attention_ones={sum(after_mask)}")
    print(f"before_ids={before_ids}")
    print(f"after_ids={after_ids}")
    print(f"attention_mask={after_mask}")
    print(f"position_ids={position_ids.tolist()}")
    print(f"media_pad_positions={after_media}")
    print(f"pixel_preprocess_max_abs={max_abs:.9g} rms={rms:.9g}")
    print("--- special token positions after expansion ---")
    for name in SPECIAL_NAMES:
        token_id = tokenizer.convert_tokens_to_ids(name)
        # Unknown strings map to [UNK] in this tokenizer; only report names that really exist.
        known_id = special.get(name)
        if known_id is None:
            print(f"{name}: MISSING")
        else:
            assert int(token_id) == int(known_id), (name, token_id, known_id)
            print(f"{name}: id={known_id} positions={_positions(after_ids, int(known_id))}")
    print(f"PASS {label}: exact AutoProcessor multimodal token sequence matches runtime frontend")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("vi_image")
    ap.add_argument("en_image")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)

    run_case(
        processor,
        model_dir,
        Path(args.vi_image),
        "Hãy nhìn ảnh này và trả lời hoàn toàn bằng tiếng Việt. Mô tả ngắn gọn nhân vật và biểu cảm trong ảnh trong một câu.",
        "VI",
    )
    run_case(
        processor,
        model_dir,
        Path(args.en_image),
        "Look at this image and answer only in English. Describe the character and facial expression in one short sentence.",
        "EN",
    )


if __name__ == "__main__":
    main()
