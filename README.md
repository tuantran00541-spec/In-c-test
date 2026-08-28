# kimi-vl-lowram v5

CPU-first low-RAM inference prototype for `moonshotai/Kimi-VL-A3B-Instruct`, borrowing the
storage discipline of `kimi-k3-in-c` while keeping model-specific math separate.

**Real-weight V5 status: PASS.** The complete official Kimi-VL text model now forwards in C
from aligned SSD-backed runtime stores:

```text
token embedding
  -> 27 decoder layers
  -> final RMSNorm
  -> 163,840-way LM head
  -> logits
```

For token id `1`, the C runtime matched the packed PyTorch oracle with `2.8252602e-05`
maximum absolute logit error and selected the same argmax token, `1008`. Both trunk and
expert backing stores used direct I/O. See `spec/REAL_V5_RESULT.md`.

## Milestones

- **V0:** safetensors routed-expert packer -> aligned `experts.bin` + `experts.idx`.
- **V1:** hard-budget LRU with `EMPTY/INFLIGHT/VALID`, offset-sorted concurrent `getmany()`
  reads, portable direct I/O and cache metrics.
- **V2:** real Kimi/DeepSeek-style router + BF16 streamed MoE; official layer-1 weights pass.
- **V3:** RMSNorm + Kimi-VL MLA/RoPE/causal attention + residuals + streamed MoE; one full
  official decoder layer passes on a four-token sequence.
- **V4:** separate aligned trunk backing store plus a real two-layer stack; official dense
  layer 0 -> sparse layer 1 passes with both stores using direct I/O.
- **V5:** bounded conversion of the complete official text checkpoint, all 27 decoder layers,
  global embedding/final norm/LM head, and a complete C logits forward.

## Full V5 runtime pack

```text
trunk.bin       2.916 GiB
trunk.idx      18,240 bytes
experts.bin    26.812 GiB
experts.idx   133,168 bytes

routed expert records = 1664 = 26 x 64
```

`tools/pack_full_text.py` downloads official source shards with a bounded working set. A few
decoder layers cross a shard boundary, so the converter may keep two source shards at once;
once no unfinished tensor depends on a shard it is deleted. In the passing V5 workflow all
seven large safetensor source shards were removed by the end of conversion.

## V5 real run

The C run used a hard 512 MiB routed-expert cache:

```text
cache slots       31
MoE selections    156
physical reads    156
expert bytes read 2574.00 MiB
expert evictions  125
read failures     0

worst layer       26
layer max abs     1.0681152e-04
logits max abs    2.8252602e-05
logits RMS        4.0282628e-06
argmax             1008
reference argmax   1008
trunk direct I/O   yes
expert direct I/O  yes
```

The measured disk throughput in CI is runner-specific and is not a laptop benchmark.

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
```

The real full-text workflow is `.github/workflows/real-v5.yml`.

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

Records are 4096-byte aligned. The current implementation loads the needed trunk records,
computes with them and releases them; routed experts use the hard-budget LRU/prefetch cache.

## Next: V6 incremental decoding

V5 proves a complete one-token text forward, but it is not generation yet. V6 adds persistent
MLA state and compares token-by-token decoding against causal multi-token prefill.

The correctness plan is deliberately staged:

1. **V6a:** an expanded K/V cache used as a reference implementation. Incremental outputs
   must match the already-validated causal prefill path.
2. **V6b:** replace the expanded reference state with the intended compressed MLA state
   (latent KV + RoPE component) and prove it produces the same outputs.
3. Run a real multi-token 27-layer oracle and compare next-token logits.
4. Add tokenizer/sampling only after incremental logits match.

Quantized expert kernels and MoonViT vision remain separate later milestones so they do not
hide decoder-correctness bugs.
