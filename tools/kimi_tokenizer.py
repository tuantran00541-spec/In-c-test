#!/usr/bin/env python3
"""Small tokenizer helper matching moonshotai/Kimi-VL-A3B-Instruct.

Inference stays in C. This module only reproduces the official tiktoken regex/ranks,
special-token table, and text-only chat template so V7 can turn strings into token IDs
and generated token IDs back into UTF-8 text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import tiktoken
from tiktoken.load import load_tiktoken_bpe

PAT_STR = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

DEFAULT_SYSTEM = "You are a helpful assistant"


def build_encoding(model_dir: str | Path) -> tuple[tiktoken.Encoding, int, dict[str, int]]:
    model_dir = Path(model_dir)
    ranks = load_tiktoken_bpe(str(model_dir / "tiktoken.model"))
    cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    added = {
        int(k): v["content"] if isinstance(v, dict) else str(v)
        for k, v in cfg.get("added_tokens_decoder", {}).items()
    }
    n_base = len(ranks)
    # Mirrors tokenization_moonshot.py: 256 reserved specials plus two trailing slots.
    special = {
        added.get(i, f"<|reserved_token_{i}|>"): i
        for i in range(n_base, n_base + 258)
    }
    enc = tiktoken.Encoding(
        name="kimi-vl-a3b-instruct",
        pat_str=PAT_STR,
        mergeable_ranks=ranks,
        special_tokens=special,
    )
    return enc, n_base, special


def format_text_chat(prompt: str, system: str = DEFAULT_SYSTEM) -> str:
    # Exact text-only specialization of the official Jinja chat template.
    return (
        f"<|im_system|>system<|im_middle|>{system}<|im_end|>"
        f"<|im_user|>user<|im_middle|>{prompt}<|im_end|>"
        f"<|im_assistant|>assistant<|im_middle|>"
    )


def encode_chat(enc: tiktoken.Encoding, prompt: str, system: str = DEFAULT_SYSTEM) -> list[int]:
    return enc.encode(format_text_chat(prompt, system), allowed_special="all")


def decode_generated(enc: tiktoken.Encoding, ids: Iterable[int], stop_ids: set[int] | None = None) -> str:
    kept: list[int] = []
    stop_ids = stop_ids or set()
    for token in ids:
        if token in stop_ids:
            break
        kept.append(token)
    return enc.decode(kept)
