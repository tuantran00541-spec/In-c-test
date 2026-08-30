Kimi-VL compression lab quickstart

1. Build bounded per-expert activation reservoirs from an instrumented routed-token trace:

   python tools/kimi_expert_reservoir.py trace.jsonl reservoirs --capacity 128 --hidden 2048 --seed 1

2. Attach pinned BF16 gate/up/down tensors for one expert to its reservoir NPZ, then simulate:

   python tools/kimi_compression_lab.py expert.npz --bits 8,6,5,4 --group-size 128 --output expert.json

3. Once multiple expert sensitivity JSON files exist, choose a projected mixed-bit assignment:

   python tools/kimi_mixed_bit_plan.py expert-*.json --budget-bytes BYTES --output plan.json

These tools are offline research utilities. They do not implement a native low-bit runtime format, do not physically compact weights, and do not prove end-to-end model quality or speed.
