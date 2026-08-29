#!/usr/bin/env python3
"""Synthetic long-context RAM-planner regression checks; allocates no model tensors."""
from __future__ import annotations

from kvl_memory_plan import MIB, planned_text_breakdown


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

    # Phase-one target: exact 16k + conservative 512 MiB expert cache fits the
    # existing 4 GiB text budget according to the now-complete planner.
    assert p16["total"] < 4096 * MIB

    # A 2 GiB expert cache is still useful for short prompts but must not be
    # silently accepted under a 4 GiB budget at 16k.
    assert p16_big["total"] > 4096 * MIB

    # 32k is deliberately not promised by this phase under the 4 GiB budget.
    assert p32["total"] > 4096 * MIB

    print("long_context_planner PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
