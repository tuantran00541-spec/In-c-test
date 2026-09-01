#!/usr/bin/env python3

from __future__ import annotations

from kvl_memory_plan import (
    FULL_TRUNK_CACHE_MIB,
    MIB,
    parse_trunk_cache_request,
    planned_text_breakdown,
    resolve_trunk_cache_mib,
)

def main() -> int:
    for context in (8192, 16384, 32768):
        for cache_mib in (512, 1024, 1536, 2048):
            plan = planned_text_breakdown(context, 32, cache_mib)
            total_mib = plan["total"] / MIB
            print(
                f"plan context={context} cache={cache_mib} total={total_mib:.2f} MiB "
                f"state={plan['compressed_state']/MIB:.2f} "
                f"seq_ws={plan['sequence_workspace']/MIB:.2f} "
                f"mla_ws={plan['streaming_mla_workspace']/MIB:.2f}"
            )

    p8 = planned_text_breakdown(8192, 32, 512)
    p16 = planned_text_breakdown(16384, 32, 512)
    p16_big = planned_text_breakdown(16384, 32, 2048)
    p32 = planned_text_breakdown(32768, 32, 512)

    assert p8["compressed_state"] < p16["compressed_state"] < p32["compressed_state"]
    assert p8["sequence_workspace"] * 2 == p16["sequence_workspace"]
    assert p16["sequence_workspace"] * 2 == p32["sequence_workspace"]
    assert p16["streaming_mla_workspace"] < 53 * MIB

    assert p16["total"] < 4096 * MIB
    assert p16_big["total"] > 4096 * MIB
    assert p32["total"] > 4096 * MIB

    # Auto trunk cache must preserve the same hard RAM contract. Short/common prompts
    # can retain the full 1706 MiB non-global trunk; long contexts automatically give
    # RAM back to KV/workspaces instead of failing or exceeding the budget.
    short_auto = resolve_trunk_cache_mib(2048, 96, 512, 4096, "auto")
    long_auto = resolve_trunk_cache_mib(8192, 96, 512, 4096, "auto")
    very_long_auto = resolve_trunk_cache_mib(16384, 96, 512, 4096, "auto")
    assert short_auto == FULL_TRUNK_CACHE_MIB
    assert 0 < very_long_auto < long_auto < FULL_TRUNK_CACHE_MIB

    for context, resolved in ((2048, short_auto), (8192, long_auto), (16384, very_long_auto)):
        auto_plan = planned_text_breakdown(context, 96, 512, resolved)
        assert auto_plan["total"] <= 4096 * MIB
        if resolved < FULL_TRUNK_CACHE_MIB:
            one_more = planned_text_breakdown(context, 96, 512, resolved + 1)
            assert one_more["total"] > 4096 * MIB

    assert resolve_trunk_cache_mib(2048, 96, 512, 4096, "0") == 0
    assert resolve_trunk_cache_mib(2048, 96, 512, 4096, "256") == 256
    assert parse_trunk_cache_request("AUTO") is None
    try:
        parse_trunk_cache_request("-1")
    except ValueError:
        pass
    else:
        raise AssertionError("negative trunk cache request was accepted")

    print(
        f"auto_trunk_cache short={short_auto}MiB long={long_auto}MiB "
        f"very_long={very_long_auto}MiB hard_cap=4096MiB"
    )
    print("long_context_planner PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
