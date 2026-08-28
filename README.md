# kimi-vl-lowram v7

CPU-first low-RAM inference prototype for `moonshotai/Kimi-VL-A3B-Instruct`, borrowing the
storage discipline of `kimi-k3-in-c` while keeping model-specific math separate.

**Real-weight V7 status: PASS.** The complete official Kimi-VL text decoder now generates
coherent autoregressive text through the custom C runtime from aligned SSD-backed stores.
Official-compatible tiktoken BPE/chat templating, greedy/temperature sampling, persistent
compressed MLA state and hard-budget streamed MoE are wired end-to-end.

Accepted V7 workflow output for the prompt
`Reply with exactly one short word: hello`:

```text
Hello! How can I assist
```

See `spec/REAL_V7_RESULT.md`.

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
- **V7:** official-compatible tokenizer/chat template + C generation loop + sampling/EOS;
  the SSD-backed runtime generates coherent text from a real user prompt.

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
source shard is deleted as soon as no unfinished tensor depends on it. In the passing V7
workflow all seven large source shards were gone after conversion.

## V7 text CLI

Fetch tokenizer assets and prepare the runtime pack, then run:

```sh
python tools/kvl_chat.py /path/to/packed-model "Hello" \
  --binary ./build/kvl_generate \
  --cache-mib 512 \
  --max-new 32 \
  --temperature 0
```

`tools/kimi_tokenizer.py` reconstructs the released Moonshot tiktoken vocabulary/special-token
mapping. The regression oracle compares English, Vietnamese, Chinese, punctuation/newlines
and the released chat template against the official `AutoTokenizer`.

## V6/V7 compressed MLA state

The intended incremental state stores normalized latent KV plus the RoPE component and uses
absorbed MLA algebra instead of retaining expanded historical per-head K/V vectors.

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

## V7 accepted run

```text
prompt tokens                24
new tokens                    6
compressed MLA state       1.78 MiB
expert cache               512 MiB
cache slots                  31
physical expert reads      4524
expert bytes read        74646 MiB
expert evictions           4493
read failures                 0
trunk direct I/O              yes
expert direct I/O             yes
```

The workflow intentionally uses one-token incremental execution for prompt prefill to reuse
the already-proven V6 path. On that 2-thread GitHub runner, the 24-token prompt prefill took
roughly 264 seconds and subsequent decode was about 10.8 seconds/token. Those numbers are not
a laptop performance claim; they identify the optimization target for V8.

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
python tests/test_tokenizer_oracle.py /path/to/tokenizer-assets
```

Real workflows:

```text
.github/workflows/real-v5.yml              full one-token logits
.github/workflows/real-v6-mla.yml          official compressed-MLA layer probe
.github/workflows/real-v6-full-decode.yml  full 27-layer incremental equivalence
.github/workflows/real-v7-text.yml         official tokenizer + real text generation
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

## Release path: V8 -> V9

The target is not merely a numerical prototype. The first intended complete release is V9.

### V8 — practical laptop text runtime

- faster/batched prompt prefill rather than one-token-at-a-time prefill;
- optimized CPU BF16/quantized GEMV (AVX2 where available);
- production Windows direct-I/O path and MSVC-friendly build flags;
- hard total-RAM planning, not only routed-expert cache budgeting;
- practical model-pack/install CLI and benchmarking;
- preserve V5/V6/V7 numerical/text baselines as regression gates.

### V9 — complete Kimi-VL multimodal runtime

- pack/load MoonViT + multimodal projector;
- image preprocessing/patch embedding/MoonViT forward;
- image-token merge into the text sequence;
- phase-based vision arena so vision memory can be released before long text decode;
- end-to-end text+image CLI on the same low-RAM SSD-backed runtime;
- Windows/Linux release workflow and reproducible benchmark report.
