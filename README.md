# kimi-vl-lowram v6

CPU-first low-RAM inference prototype for `moonshotai/Kimi-VL-A3B-Instruct`, borrowing the
storage discipline of `kimi-k3-in-c` while keeping model-specific math separate.

**Real-weight V6 status: PASS.** The complete official Kimi-VL text decoder now supports
persistent token-by-token execution in C from aligned SSD-backed runtime stores using
compressed MLA history.

The accepted full-model test compares:

```text
A. causal prefill of token ids [1, 1008]
B. token 1 -> persistent compressed MLA state -> token 1008 incremental decode
```

through all 27 decoder layers, final RMSNorm and the 163,840-way LM head. With the official
`rms_norm_eps=1e-5`, incremental decoding matched causal prefill with
`7.62939453e-06` maximum hidden-state error and `1.90734863e-06` maximum logit error; both
paths selected argmax token `1609`. See `spec/REAL_V6_RESULT.md`.

## Milestones

- **V0:** safetensors routed-expert packer -> aligned `experts.bin` + `experts.idx`.
- **V1:** hard-budget LRU with `EMPTY/INFLIGHT/VALID`, offset-sorted concurrent `getmany()`
  reads, portable direct I/O and cache metrics.
- **V2:** real Kimi/DeepSeek-style router + BF16 streamed MoE; official layer-1 weights pass.
- **V3:** RMSNorm + Kimi-VL MLA/RoPE/causal attention + residuals + streamed MoE; one full
  official decoder layer passes on a four-token sequence.
- **V4:** separate aligned trunk backing store plus a real two-layer stack; official dense
  layer 0 -> sparse layer 1 passes with both stores using direct I/O.
- **V5:** complete 27-layer official text model -> final RMSNorm -> 163,840 logits in C.
- **V6:** persistent incremental decode using compressed MLA latent+RoPE history; full
  two-token 27-layer execution matches causal prefill and produces the same next argmax.

## Complete runtime pack

```text
trunk.bin       2.916 GiB
trunk.idx      18,240 bytes
experts.bin    26.812 GiB
experts.idx   133,168 bytes

routed expert records = 1664 = 26 x 64
```

`tools/pack_full_text.py` downloads official source shards with a bounded working set. Some
layers cross shard boundaries, so the converter may retain two source shards at once; a
source shard is deleted as soon as no unfinished tensor depends on it. In the passing V6
workflow all seven large source shards were gone after conversion.

## V6 compressed MLA state

V6 has two incremental implementations:

1. an expanded K/V state used only as a correctness reference;
2. the intended compressed MLA state, which stores normalized latent KV plus the RoPE
   component and performs the K/V projection algebra during attention.

For the released Kimi-VL dimensions:

```text
latent KV rank                  512
RoPE component                   64
compressed FP32 payload
  per layer / historical token 2304 bytes
```

An official production-dimension layer-1 test reduced four-token state from 81,960 bytes
expanded to 9,248 bytes compressed (~8.862x) while matching causal prefill within
`4.42378223e-09` max absolute error.

For the final full-model two-token test, all 27 compressed states together allocated
125,280 bytes.

## Full V6 real run

Official semantics and execution:

```text
rms_norm_eps         1.0e-05
tokens                    2
worst layer              26
worst token                1
hidden max abs     7.62939453e-06
logits max abs     1.90734863e-06
logits RMS         4.55979847e-07
prefill argmax          1609
incremental argmax      1609
trunk direct I/O         yes
expert direct I/O        yes
```

The hard 512 MiB routed-expert cache was heavily exercised:

```text
cache slots          31
physical reads       595
expert bytes read 9817.50 MiB
expert evictions     564
read failures          0
```

The cache's compute-side hit count is measured after `getmany()` prefetch; the physical-read
and eviction counters show that the complete routed working set was not resident.

The measured GitHub Actions disk throughput is runner-specific and is not a laptop
performance claim.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Regression tests

```sh
python tests/test_pack_roundtrip.py
python tests/test_cache_roundtrip.py --build-dir build
python tests/test_moe_oracle.py --build-dir build
python tests/test_layer_oracle.py --build-dir build
python tests/test_stack_oracle.py --build-dir build
./build/kvl_mla_incremental_probe
```

Real workflows:

```text
.github/workflows/real-v5.yml              full one-token logits
.github/workflows/real-v6-mla.yml          official compressed-MLA layer probe
.github/workflows/real-v6-full-decode.yml  full 27-layer incremental equivalence
```

## Runtime backing stores

```text
trunk.bin / trunk.idx
  token embeddings
  attention projections
  RMSNorms
  dense layer-0 MLP
  MoE routers
  shared experts
  final RMSNorm
  LM head

experts.bin / experts.idx
  routed expert gate/up/down matrices
```

Records are 4096-byte aligned. Trunk tensors are loaded/released as needed; routed experts
use the hard-budget LRU/prefetch cache.

## Next: V7 actual text generation

V6 closes the incremental-decoder correctness milestone. The shortest route to visible text
is now tokenizer + generation plumbing rather than another model-math block:

```text
prompt
  -> tokenizer
  -> prefill / compressed MLA state
  -> logits
  -> greedy or sampled token
  -> incremental decode
  -> EOS / stop handling
  -> decoded text
```

After this baseline CLI generates real text, performance work can proceed independently:
BF16 MLA-state compression, routed-expert quantization/direct AVX2 kernels, smarter expert
prefetch/cache policy and finally MoonViT vision integration.
