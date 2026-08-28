#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kimi_tokenizer import (  # noqa: E402
    build_encoding,
    encode_chat,
    encode_image_chat,
    format_image_chat,
    format_text_chat,
)

REPO = "moonshotai/Kimi-VL-A3B-Instruct"


def unwrap(x):
    if hasattr(x, "keys"):
        x = x["input_ids"]
    return list(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    args = ap.parse_args()

    enc, _, _ = build_encoding(args.model_dir)
    ref = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

    samples = [
        "Hello, world!",
        "Xin chào Việt Nam :v",
        "你好，世界。",
        "Numbers 123456 and punctuation...?!",
        "line one\nline two\n",
    ]
    for text in samples:
        got = enc.encode(text, allowed_special="all")
        expected = ref.encode(text)
        assert got == expected, (text, got, expected)
        assert enc.decode(got) == ref.decode(expected)

    prompt = "Viết đúng một từ: chào"
    messages = [{"role": "user", "content": prompt}]
    ref_text = ref.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    got_text = format_text_chat(prompt)
    assert got_text == ref_text, (got_text, ref_text)
    got_ids = encode_chat(enc, prompt)
    ref_ids = unwrap(ref.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    assert got_ids == ref_ids, (got_ids, ref_ids)

    # The official processor first renders exactly one media pad in the Jinja template, then
    # expands that pad to grid_h*grid_w/4 copies before tokenization. Three is arbitrary here
    # and tests the expansion semantics independent of a particular image size.
    image_prompt = "Mô tả ảnh này."
    media_tokens = 3
    image_messages = [{"role": "user", "content": [
        {"type": "image", "url": "unused"},
        {"type": "text", "text": image_prompt},
    ]}]
    ref_image_text = ref.apply_chat_template(image_messages, tokenize=False, add_generation_prompt=True)
    ref_image_text = ref_image_text.replace("<|media_pad|>", "<|media_pad|>" * media_tokens, 1)
    got_image_text = format_image_chat(image_prompt, media_tokens)
    assert got_image_text == ref_image_text, (got_image_text, ref_image_text)
    got_image_ids = encode_image_chat(enc, image_prompt, media_tokens)
    ref_image_ids = ref.encode(ref_image_text)
    assert got_image_ids == ref_image_ids, (got_image_ids, ref_image_ids)

    print(
        f"PASS: tokenizer + text/image chat templates match official oracle; "
        f"text_tokens={len(got_ids)} image_tokens={len(got_image_ids)}"
    )


if __name__ == "__main__":
    main()
