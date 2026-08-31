#!/usr/bin/env python3
"""Semantic answer scoring helpers for Kimi-VL hard regression evidence.

The hard gate asks for a structured marker (for example FINAL=37), but intelligence
retention and instruction-format compliance are separate measurements. This module
therefore reports both. Marker-bearing conclusions have precedence; conservative
fallbacks are used only when no marker is present. Conflicting conclusions are
reported as ambiguous rather than guessed.
"""
from __future__ import annotations

import re


def _norm_word(value: str) -> str:
    return re.sub(r"[^a-z]+", "", value.lower())


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def score_semantic_answer(text: str, *, marker: str, answer_kind: str, expected: str) -> dict:
    if answer_kind not in {"integer", "word"}:
        raise ValueError(f"unsupported answer kind: {answer_kind}")
    marker_re = re.escape(marker)
    if answer_kind == "integer":
        value_re = r"-?\d+"
        normalize = lambda s: str(int(s))
        expected_norm = str(int(expected))
    else:
        value_re = r"[A-Za-z]+"
        normalize = _norm_word
        expected_norm = _norm_word(expected)

    # Exact requested surface is a format metric, not the only semantic metric.
    strict = re.findall(rf"(?im)^\s*{marker_re}\s*=\s*({value_re})\s*[.!]?\s*$", text)
    strict_norm = _dedupe([normalize(x) for x in strict])
    format_ok = len(strict_norm) == 1

    # If the marker is present at all, trust marker-bearing conclusions over prose.
    marked = re.findall(rf"(?i)\b{marker_re}\b\s*[:=]\s*({value_re})\b", text)
    marked_norm = _dedupe([normalize(x) for x in marked])
    if marked_norm:
        extracted = marked_norm[0] if len(marked_norm) == 1 else None
        return {
            "semantic_correct": extracted == expected_norm,
            "format_ok": format_ok,
            "extracted": extracted,
            "expected": expected_norm,
            "source": "marker",
            "ambiguous": len(marked_norm) != 1,
            "candidates": marked_norm,
        }

    # Conservative fallback: inspect only the tail, where a conclusion normally lives.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-3:])
    phrases = r"(?:answer|result|final answer|kết quả|ket qua|đáp án|dap an)"
    fallback = re.findall(rf"(?i){phrases}\s*(?:is|là|la|:|=)?\s*({value_re})\b", tail)
    fallback_norm = _dedupe([normalize(x) for x in fallback])

    # Also permit a bare final line, but never search arbitrary body numbers/colors.
    if lines:
        bare = re.fullmatch(rf"\s*({value_re})\s*[.!]?\s*", lines[-1])
        if bare:
            fallback_norm = _dedupe(fallback_norm + [normalize(bare.group(1))])

    extracted = fallback_norm[0] if len(fallback_norm) == 1 else None
    return {
        "semantic_correct": extracted == expected_norm,
        "format_ok": False,
        "extracted": extracted,
        "expected": expected_norm,
        "source": "tail_fallback" if fallback_norm else "none",
        "ambiguous": len(fallback_norm) > 1,
        "candidates": fallback_norm,
    }
