#!/usr/bin/env python3

from __future__ import annotations

HIDDEN = 2048
LAYERS = 27
HEADS = 16
QK_NOPE = 128
QK_ROPE = 64
VALUE_HEAD = 128
KV_LATENT = 512
KV_LATENT_PLUS_ROPE = KV_LATENT + QK_ROPE
DENSE_INTERMEDIATE = 11264
FP32_BYTES = 4
BF16_BYTES = 2
MIB = 1024 * 1024

GLOBAL_BF16_BYTES = 2 * (163840 * HIDDEN * BF16_BYTES) + HIDDEN * BF16_BYTES

LAYER_MAJOR_PROMPT_BUFFERS = 3

PEAK_LAYER_BF16_BYTES = 3 * DENSE_INTERMEDIATE * HIDDEN * BF16_BYTES

RUNTIME_MISC_SAFETY = 128 * MIB

# Exact non-global text-trunk payload measured from the released Kimi-VL pack is
# 1705.66796875 MiB. The runtime accepts an integer MiB cache budget, so 1706 MiB
# is the smallest budget that can retain all always-used layer tensors.
FULL_TRUNK_CACHE_MIB = 1706


def parse_trunk_cache_request(value: str | int) -> int | None:
    """Return an explicit MiB budget, or None for planner-managed 'auto'."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError("trunk cache must be >= 0")
        return value
    raw = str(value).strip().lower()
    if raw == "auto":
        return None
    try:
        parsed = int(raw, 10)
    except ValueError as exc:
        raise ValueError("trunk cache must be 'auto' or a non-negative integer MiB value") from exc
    if parsed < 0:
        raise ValueError("trunk cache must be >= 0")
    return parsed


def planned_text_breakdown(
    prompt_tokens: int,
    max_new: int,
    cache_mib: int,
    trunk_cache_mib: int = 0,
) -> dict[str, int]:
    if prompt_tokens <= 0 or max_new <= 0 or cache_mib <= 0 or trunk_cache_mib < 0:
        raise ValueError(
            "prompt_tokens, max_new and cache_mib must be positive; trunk_cache_mib must be >= 0"
        )

    capacity = prompt_tokens + max_new
    compressed_state = LAYERS * capacity * KV_LATENT_PLUS_ROPE * FP32_BYTES
    sequence_workspace = (
        LAYER_MAJOR_PROMPT_BUFFERS * prompt_tokens * HIDDEN * FP32_BYTES
    )

    streaming_mla_workspace = (
        prompt_tokens * (KV_LATENT_PLUS_ROPE + QK_NOPE + VALUE_HEAD + 1) * FP32_BYTES
    )
    streaming_mla_workspace += (
        (QK_NOPE + QK_ROPE) * FP32_BYTES
        + (KV_LATENT + QK_ROPE) * FP32_BYTES
        + (QK_NOPE + VALUE_HEAD) * FP32_BYTES
        + QK_ROPE * FP32_BYTES
        + (HEADS * VALUE_HEAD) * FP32_BYTES
        + VALUE_HEAD * 8
    )

    parts = {
        "global_bf16": GLOBAL_BF16_BYTES,
        "compressed_state": compressed_state,
        "sequence_workspace": sequence_workspace,
        "streaming_mla_workspace": streaming_mla_workspace,
        "expert_cache": cache_mib * MIB,
        "trunk_cache": trunk_cache_mib * MIB,
        "peak_layer_bf16": PEAK_LAYER_BF16_BYTES,
        "misc_safety": RUNTIME_MISC_SAFETY,
    }
    parts["total"] = sum(parts.values())
    return parts


def resolve_trunk_cache_mib(
    prompt_tokens: int,
    max_new: int,
    cache_mib: int,
    ram_mib: int,
    requested: str | int = "auto",
) -> int:
    """Resolve a trunk-cache request without violating the conservative RAM plan.

    Explicit integer budgets are returned unchanged and are validated by the caller's
    final plan. In auto mode, use as much of the known 1706 MiB full-trunk budget as
    the existing hard RAM contract can safely accommodate. The planner's normal peak
    layer and misc safety allowances remain counted, so auto never borrows from them.
    """
    if ram_mib <= 0:
        raise ValueError("ram_mib must be positive")
    parsed = parse_trunk_cache_request(requested)
    if parsed is not None:
        return parsed

    base = planned_text_breakdown(prompt_tokens, max_new, cache_mib, 0)
    available = ram_mib * MIB - base["total"]
    if available <= 0:
        return 0
    return min(FULL_TRUNK_CACHE_MIB, available // MIB)


def planned_text_bytes(
    prompt_tokens: int, max_new: int, cache_mib: int, trunk_cache_mib: int = 0
) -> int:
    return planned_text_breakdown(prompt_tokens, max_new, cache_mib, trunk_cache_mib)["total"]


def as_mib(n: int) -> float:
    return n / MIB
