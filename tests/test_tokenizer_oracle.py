#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kimi_tokenizer import build_encoding, encode_chat, format_text_chat  # noqa: E402

REPO = "moonshotai/Kimi-VL-A3B-Instruct"


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
    ref_ids = ref.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ref_ids, "keys"):
        ref_ids = ref_ids["input_ids"]
    assert got_ids == list(ref_ids), (got_ids, ref_ids)

    print(f"PASS: tokenizer + text chat template match official oracle; chat_tokens={len(got_ids)}")


if __name__ == "__main__":
    main()
