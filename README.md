# kimi-vl-lowram v3

CPU-first low-RAM inference prototype for `moonshotai/Kimi-VL-A3B-Instruct`, borrowing the
storage discipline of `kimi-k3-in-c` while keeping model-specific math separate.

**Real-weight V3 status: PASS.** On 2026-08-27 a complete four-token causal prefill through
Kimi-VL decoder layer 1 ran in C using official weights:

```text
RMSNorm -> MLA/RoPE/causal attention -> residual
        -> RMSNorm -> streamed top-6 MoE -> residual
```

The final layer output matched the manual PyTorch oracle with `1.1920929e-07` max absolute
error. The routed expert pack was 1.031 GiB while the test cache was hard-capped at 256 MiB,
forcing real direct-I/O reads and evictions. See `spec/REAL_V3_RESULT.md`.

## What works

- **V0:** safetensors routed-expert packer -> aligned `experts.bin` + `experts.idx`.
- **V1:** hard-budget LRU, `EMPTY/INFLIGHT/VALID`, offset-sorted concurrent `getmany()`
  reads, portable direct I/O, cache metrics.
- **V2:** Kimi/DeepSeek-style router, BF16 streamed expert GEMV, SiLU-GLU, routed weighted
  sum and resident shared expert. Official layer-1 MoE weights pass the Torch oracle.
- **V3:** RMSNorm + Kimi-VL MLA + exact RoPE permutation + causal attention + residuals +
  V2 streamed MoE. A real four-token decoder-layer probe passes end-to-end.
- Synthetic V0/V1/V2/V3 regressions are available under `tests/`; V3 was also exercised
  under AddressSanitizer and UndefinedBehaviorSanitizer during development.
- `tools/fetch_real_v2.py` and `tools/fetch_real_v3.py` resolve only the checkpoint shard(s)
  required for a requested layer instead of snapshotting the full ~32.8 GB model.

For the current official checkpoint, the complete decoder layer 1 (205 tensors) resolves to
one shard: `model-00003-of-00007.safetensors`. This is discovered from the checkpoint index,
not hardcoded.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Local regression tests

```sh
python tests/test_pack_roundtrip.py
python tests/test_cache_roundtrip.py --build-dir build
python tests/test_moe_oracle.py --build-dir build
python tests/test_layer_oracle.py --build-dir build
```

## Pull only the real weights needed for V3

Install Python-side dependencies:

```sh
pip install huggingface_hub safetensors torch numpy
```

Resolve the exact shard without downloading it:

```sh
python tools/fetch_real_v3.py D:/models/Kimi-VL-real-v3 --layer 1 --metadata-only
```

Then download only the resolved shard(s):

```sh
python tools/fetch_real_v3.py D:/models/Kimi-VL-real-v3 --layer 1
```

Pack routed experts and create a four-token full-layer oracle:

```sh
python tools/pack_experts.py \
  D:/models/Kimi-VL-real-v3 \
  D:/models/Kimi-VL-layer1 \
  --layer 1

python tools/dump_real_layer_reference.py \
  D:/models/Kimi-VL-real-v3 \
  D:/models/Kimi-VL-layer1/layer1-v3.fixture \
  --layer 1 --seq-len 4
```

Run the C layer probe with a 256 MiB expert-cache cap:

```sh
build/kvl_layer_probe \
  D:/models/Kimi-VL-layer1/experts.bin \
  D:/models/Kimi-VL-layer1/experts.idx \
  D:/models/Kimi-VL-layer1/layer1-v3.fixture \
  268435456
```

See:

- `spec/V2_NUMERICS.md`
- `spec/REAL_V2_RESULT.md`
- `spec/V3_DECODER_LAYER.md`
- `spec/REAL_V3_RESULT.md`

## Next: V4

V4 moves from a single layer to a **real text-decoder path**. The immediate targets are:

1. pack/load the always-active trunk cleanly instead of embedding resident tensors in a
   test fixture;
2. chain multiple real decoder layers, then all 27;
3. add persistent compressed MLA KV state for incremental decoding;
4. add token embeddings, final RMSNorm and LM head so the C runtime can produce real logits;
5. only after the text path is stable, add quantized expert kernels and MoonViT vision.
