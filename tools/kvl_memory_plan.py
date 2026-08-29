#!/usr/bin/env python3
"""Pure-Python RAM accounting for the Kimi-VL V9 text phase."""
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

# Global tensors held for the complete text phase: embedding, lm_head, final RMSNorm.
GLOBAL_BF16_BYTES = 2 * (163840 * HIDDEN * BF16_BYTES) + HIDDEN * BF16_BYTES

# Long-context prefill aliases r1->attn, n2->n1 and y->x once each source value is dead,
# so only three [prompt,H] FP32 matrices are simultaneously allocated.
LAYER_MAJOR_PROMPT_BUFFERS = 3

# Largest simultaneously-loaded non-expert trunk block is layer-0 dense gate/up/down.
PEAK_LAYER_BF16_BYTES = 3 * DENSE_INTERMEDIATE * HIDDEN * BF16_BYTES

# Small allocations, allocator/alignment overhead, prompt/media arrays and headroom not otherwise
# modeled. Vision is excluded because it runs in a separate subprocess before text inference.
RUNTIME_MISC_SAFETY = 128 * MIB


def planned_text_breakdown(prompt_tokens: int, max_new: int, cache_mib: int) -> dict[str, int]:
    if prompt_tokens <= 0 or max_new <= 0 or cache_mib <= 0:
        raise ValueError("prompt_tokens, max_new and cache_mib must be positive")

    capacity = prompt_tokens + max_new
    compressed_state = LAYERS * capacity * KV_LATENT_PLUS_ROPE * FP32_BYTES
    sequence_workspace = (
        LAYER_MAJOR_PROMPT_BUFFERS * prompt_tokens * HIDDEN * FP32_BYTES
    )

    # Exact head-wise streaming MLA prefill keeps one shared [S,R+DR] latent/RoPE set,
    # one head's [S,DN] K(no-PE), one head's [S,DV] V and one score per prompt token.
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
        "peak_layer_bf16": PEAK_LAYER_BF16_BYTES,
        "misc_safety": RUNTIME_MISC_SAFETY,
    }
    parts["total"] = sum(parts.values())
    return parts


def planned_text_bytes(prompt_tokens: int, max_new: int, cache_mib: int) -> int:
    return planned_text_breakdown(prompt_tokens, max_new, cache_mib)["total"]


def as_mib(n: int) -> float:
    return n / MIB
