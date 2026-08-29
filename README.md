# kimi-vl-lowram v9

CPU-first low-RAM inference runtime for `moonshotai/Kimi-VL-A3B-Instruct`, using aligned
SSD/NVMe-backed weight stores, streamed sparse-MoE experts, compressed MLA state and a native
C MoonViT + text decoder.

**Real-weight V9 status: PASS.** The runtime now executes end-to-end image + text generation
from the released Kimi-VL checkpoint without loading the complete ~30 GiB packed model into
RAM. The final pinned two-image acceptance on `main` passed with BF16 media injection,
layer-major batched multimodal prefill and direct I/O enabled for both trunk and expert stores.

See [`spec/REAL_V9_RESULT.md`](spec/REAL_V9_RESULT.md) for the final evidence, exact pinned
checkpoint revision, outputs, cache metrics and the documented upstream-model semantic caveat.

## What is complete

- **V0:** safetensors routed-expert packer -> aligned `experts.bin` + `experts.idx`.
- **V1:** hard-budget LRU/prefetch expert cache with portable direct I/O and metrics.
- **V2:** real Kimi/DeepSeek-style router + BF16 streamed MoE.
- **V3:** RMSNorm + Kimi-VL MLA/RoPE/causal attention + residuals + streamed MoE.
- **V4:** separate aligned trunk store plus real multi-layer execution.
- **V5:** complete 27-layer text tower -> final RMSNorm -> 163,840 logits in C.
- **V6:** persistent incremental decode with compressed MLA latent+RoPE history.
- **V7:** official-compatible tokenizer/chat template + native C generation/sampling loop.
- **V8:** batched prompt prefill, RAM planning, AVX2 path, Windows direct-I/O support and
  practical packing/frontend tooling.
- **V9:** official-compatible image preprocessing, MoonViT + projector, BF16 media-token
  merge, batched multimodal prefill and end-to-end image chat.

## Packed V9 model

Real released weights pack to approximately:

```text
trunk.bin       2.916 GiB
experts.bin    26.812 GiB
vision.bin      0.834 GiB

routed expert records = 1664 = 26 x 64
```

Records are 4096-byte aligned. Trunk tensors are loaded/released as needed; routed experts use
a hard-budget cache. With `--cache-mib 512`, the real V9 acceptance uses 31 BF16 expert slots
(~511.5 MiB arena) rather than keeping all routed experts resident.

No checkpoint or packed model weight is stored in git.

## Quick start V9

### 1. Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKVL_USE_AVX2=ON
cmake --build build --config Release
```

The C runtime supports Linux direct I/O and the Windows `FILE_FLAG_NO_BUFFERING` path. The
Python frontend automatically looks under `build/Release/` for Windows executables when using
a multi-config CMake generator.

### 2. Pack the official checkpoint

Download/prepare the normal complete seven-shard Hugging Face snapshot of
`moonshotai/Kimi-VL-A3B-Instruct`, then run:

```sh
python tools/pack_full_model.py /path/to/Kimi-VL-A3B-Instruct /path/to/kimi-vl-v9-packed
```

The packer creates the text trunk, routed-expert store, MoonViT/projector store and required
frontend/tokenizer assets. Conversion uses a bounded source-shard working set; source model
weights are not copied into this repository.

### 3. Chat with an image

```sh
python tools/kvl_vl_chat.py /path/to/kimi-vl-v9-packed image.jpg \
  "Describe this image." \
  --cache-mib 512 \
  --ram-mib 4096 \
  --max-new 32 \
  --temperature 0
```

The frontend preprocesses the image, runs the native C MoonViT/projector, releases the vision
phase working set, constructs the official-style multimodal chat sequence and then runs the
27-layer C text decoder.

For text-only chat, `tools/kvl_chat.py` remains available.

## V9 runtime architecture

### Weight residency

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

vision.bin / vision.idx
  MoonViT patch embedding
  27 vision blocks
  final/projector normalization
  multimodal projector
```

### Low-RAM execution

- model weights remain SSD/NVMe resident;
- routed experts are streamed through a bounded LRU/prefetch cache;
- MLA history stores compressed latent KV + RoPE state instead of expanded historical K/V;
- multimodal prompt prefill is layer-major/batched rather than token-major;
- image features cross into the decoder through a BF16 media boundary;
- vision and text run as separate phases so large vision temporaries can be released before
  long text decode;
- `tools/kvl_vl_chat.py` rejects a request up front when its planned text working set exceeds
  `--ram-mib`.

For the released text dimensions, compressed MLA historical state is based on latent rank 512
plus RoPE component 64, or 2304 FP32 bytes per layer per historical token.

## Final real-weight acceptance

The final main-branch workflow is:

```text
.github/workflows/real-v9-user-two-images.yml
```

Pinned model revision:

```text
398eede0903cd983a2bfa0cc634e9ac1d843f375
```

Final run `33228249344` passed both real user image fixtures with:

```text
image grid                       14 x 14
media tokens                         49
expert cache                    512 MiB
cache slots                          31
prefill                 batch-layer-major-media
media boundary                      BF16
trunk direct I/O                     yes
expert direct I/O                    yes
expert read failures                   0
```

English control output:

```text
The character has large, expressive eyes and a surprised or shocked facial expression.
```

The Vietnamese fixture no longer falls into the historical "image not provided" failure mode,
but the released model itself shows prompt/language-dependent semantic inconsistency on that
fixture. A separate released-BF16 teacher-forced diagnostic verified the C-generated
24-token trajectory **24/24**, with `FIRST_DIVERGENCE=-1`; therefore the repository records
that discrepancy as upstream model behavior rather than hiding it as a runtime error. Details
are in `spec/REAL_V9_RESULT.md`.

## Regression tests and real workflows

Core local tests include:

```sh
python tests/test_pack_roundtrip.py
python tests/test_cache_roundtrip.py --build-dir build
python tests/test_moe_oracle.py --build-dir build
python tests/test_layer_oracle.py --build-dir build
python tests/test_stack_oracle.py --build-dir build
./build/kvl_mla_incremental_probe
python tests/test_tokenizer_oracle.py /path/to/tokenizer-assets
```

Historical real-weight gates remain useful regression evidence:

```text
.github/workflows/real-v5.yml              full one-token logits
.github/workflows/real-v6-mla.yml          official compressed-MLA layer probe
.github/workflows/real-v6-full-decode.yml  full 27-layer incremental equivalence
.github/workflows/real-v7-text.yml         official tokenizer + real text generation
.github/workflows/real-v9-user-two-images.yml  final pinned multimodal runtime acceptance
```

V9 is the completed Kimi-VL target for this repository; later architecture experiments should
remain separate so they do not weaken the V9 real-weight baselines.
