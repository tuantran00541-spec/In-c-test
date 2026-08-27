# kimi-vl-lowram v4

CPU-first low-RAM inference prototype for `moonshotai/Kimi-VL-A3B-Instruct`, borrowing the
storage discipline of `kimi-k3-in-c` while keeping model-specific math separate.

**Real-weight V4 status: PASS.** On 2026-08-27 a four-token prefill was chained through
official decoder layers 0 -> 1 in C. Layer 0 is the model's dense first layer; layer 1 is
sparse MoE. Resident tensors were read from a new aligned `trunk.bin/trunk.idx`, routed
experts from `experts.bin/experts.idx`, and both paths used direct I/O.

The final two-layer output matched the PyTorch oracle with `1.1920929e-07` max absolute
error under a 256 MiB expert-cache cap. See `spec/REAL_V4_RESULT.md`.

## What works

- **V0:** safetensors routed-expert packer -> aligned `experts.bin` + `experts.idx`.
- **V1:** hard-budget LRU with `EMPTY/INFLIGHT/VALID`, offset-sorted concurrent `getmany()`
  reads, portable direct I/O, and cache metrics.
- **V2:** real Kimi/DeepSeek-style router + BF16 streamed MoE; official layer-1 weights pass.
- **V3:** RMSNorm + Kimi-VL MLA/RoPE/causal attention + residuals + streamed MoE; one full
  official decoder layer passes on a four-token sequence.
- **V4:** a separate aligned trunk backing store plus a real multi-layer stack. Official
  layer 0 (dense) -> layer 1 (MoE) passes end-to-end with both trunk and expert direct I/O.
- **Bounded-shard conversion:** V4 downloaded layer 0's source shard, packed it, deleted the
  raw shard, then downloaded/packed layer 1. The converter therefore does not need the full
  checkpoint resident on disk at once.
- Synthetic regressions exist for every milestone; V3/V4 paths were also exercised with
  AddressSanitizer and UndefinedBehaviorSanitizer during development.

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
python tests/test_stack_oracle.py --build-dir build
```

## Runtime backing-store layout

```text
trunk.bin / trunk.idx
  attention projections
  RMSNorms
  dense layer-0 MLP
  MoE routers
  shared experts
  (global embeddings/final norm/LM head are reserved for V5)

experts.bin / experts.idx
  routed expert gate/up/down matrices
```

Records are 4096-byte aligned. The current V4 probe loads one layer's trunk records,
computes the layer, frees them, and advances. A later optimization can replace these
independent reads with layer-contiguous ring slots without changing the numerical path.

## Real V4 result

The passing official-weight run used:

```text
trunk data      0.213 GiB   (layers 0 and 1 only)
expert data     1.031 GiB   (64 BF16 routed experts for layer 1)
expert cache    256 MiB
unique experts  17 across four tokens
expert reads    280.50 MiB
expert evictions 2

C vs Torch:
dense_layer_max  5.9604645e-08
router_ids        OK
max_weight_abs    8.9406967e-08
final_max         1.1920929e-07
final_rms         1.765652e-08
trunk_direct_io   yes
expert_direct_io  yes
```

The real workflow is `.github/workflows/real-v4.yml` and the recorded baseline is
`spec/REAL_V4_RESULT.md`.

## Next: V5 logits

V5 generalizes the stack into an arbitrary 27-layer decoder loop, extends the bounded-shard
packer across the whole text model, and adds global token embeddings, final RMSNorm and LM
head. The target milestone is **real full-prefill text logits from the C runtime**.

After full-prefill logits match the oracle, incremental MLA/KV state, tokenizer/sampling,
quantized expert kernels and MoonViT vision can be added one at a time without mixing their
failure modes.
