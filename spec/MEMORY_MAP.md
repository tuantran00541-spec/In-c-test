# Kimi-VL-A3B low-RAM memory map v0

Derived from the official config and implementation dimensions.
The arithmetic totals 16.407656048B parameters, which corresponds to ~32.815 GB at BF16
and sanity-checks against the official ~32.8 GB checkpoint size.

| Group | Params | BF16 GiB | v0 policy |
|---|---:|---:|---|
| Token embedding | 335.544M | 0.625 | resident initially; later row-cache candidate |
| LM head | 335.544M | 0.625 | resident |
| Attention, 27 layers | 371.603M | 0.692 | resident |
| Norms | 0.113M | ~0 | resident |
| Dense layer-0 MLP | 69.206M | 0.129 | resident |
| Routers | 3.410M | 0.006 | resident |
| Shared experts | 449.839M | 0.838 | resident |
| Routed experts | 14.394851B | 26.812 | SSD stream |
| Vision tower | 416.866M | 0.776 | temporary/on-demand |
| Vision projector | 30.680M | 0.057 | temporary/on-demand |

Text non-routed trunk = ~2.916 GiB BF16.
Vision+projector = ~0.834 GiB BF16 temporary.

## Routed expert geometry

- 26 MoE layers (layer 0 is dense)
- 64 routed experts/layer
- top-6 selected/token/layer
- hidden = 2048
- expert intermediate = 1408
- one expert = 3 * 2048 * 1408 = 8,650,752 parameters
- one expert BF16 = 16.50 MiB
- one expert MXFP4 (17/32 B/weight) = 4.383 MiB
- full routed pool MXFP4 = ~7.122 GiB
- 100% cache-miss expert traffic/token MXFP4 = ~0.668 GiB

## Milestones

1. Pack BF16 routed experts without changing numeric values.
2. Prove direct-I/O expert reads reproduce resident BF16 bytes exactly.
3. Implement router + one resident expert reference path.
4. Add expert source abstraction and streaming top-6 path.
5. Compare streamed logits with PyTorch reference.
6. Only then introduce MXFP4 expert quantization/direct GEMV.
7. Add vision as a separate temporary arena after text path is correct.
