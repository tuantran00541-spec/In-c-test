# kimi-vl-lowram v2

CPU-first low-RAM inference prototype for `moonshotai/Kimi-VL-A3B-Instruct`, borrowing the
storage discipline of `kimi-k3-in-c` while keeping model-specific math separate.

V2 is the first numerical milestone: a complete **one-token Kimi/DeepSeek-style MoE
forward** now runs through the streamed expert cache and is checked against a PyTorch
oracle.

## What works

- V0: safetensors routed-expert packer -> aligned `experts.bin` + `experts.idx`.
- V1: hard-budget LRU, `EMPTY/INFLIGHT/VALID`, offset-sorted concurrent `getmany()` reads,
  portable direct I/O, cache metrics.
- V2: FP32 router, sigmoid + correction-bias `noaux_tc` top-k, unbiased normalized routing
  weights, BF16 direct GEMV, SiLU-GLU, routed weighted sum, resident shared MLP.
- Synthetic oracle uses **64 experts / top-6**, matching released Kimi-VL routing
  cardinality, while reducing matrix dimensions so CI/tests are tiny.
- One-layer real-checkpoint fixture generator avoids instantiating the full 32.8 GB model.
- `tools/fetch_real_v2.py` downloads only the checkpoint shards that contain one requested
  decoder layer's MLP tensors instead of snapshotting the whole model.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Tests

```sh
python tests/test_pack_roundtrip.py
python tests/test_cache_roundtrip.py --build-dir build
python tests/test_moe_oracle.py --build-dir build
```

The V2 oracle currently reaches about `1.5e-8` max absolute error on its deterministic
fixture after the BF16 bytes have gone through safetensors -> packer -> direct-I/O cache.

Sanitizer build used during development:

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan
python tests/test_moe_oracle.py --build-dir build-asan
```

## Pull only the real weights needed for V2

Install the small Python-side dependencies first:

```sh
pip install huggingface_hub safetensors torch
```

Resolve the exact shards for decoder layer 1 without downloading large files:

```sh
python tools/fetch_real_v2.py D:/models/Kimi-VL-real-v2 --layer 1 --metadata-only
```

Then download only those resolved shards:

```sh
python tools/fetch_real_v2.py D:/models/Kimi-VL-real-v2 --layer 1
```

The script first fetches `config.json` and `model.safetensors.index.json`, matches
`language_model.model.layers.1.mlp.*`, writes `real_v2_download_plan.json`, then downloads
only the safetensors shards containing those tensors. The official checkpoint is split
into seven shards of roughly 5 GB each (the last is smaller), so this avoids automatically
pulling the full ~32.8 GB repository.

## Validate one real model layer

This does **not** instantiate Transformers or load the whole model into RAM:

```sh
python tools/pack_experts.py \
  D:/models/Kimi-VL-real-v2 \
  D:/models/Kimi-VL-layer1 \
  --layer 1

python tools/dump_real_moe_reference.py \
  D:/models/Kimi-VL-real-v2 \
  D:/models/Kimi-VL-layer1/layer1.fixture \
  --layer 1

build/kvl_moe_probe \
  D:/models/Kimi-VL-layer1/experts.bin \
  D:/models/Kimi-VL-layer1/experts.idx \
  D:/models/Kimi-VL-layer1/layer1.fixture \
  1200000000
```

See `spec/V2_NUMERICS.md` for the numerical contract and its deliberate precision limits.

## Next: V3

V3 should add **resident tensor packing + one real decoder layer**:

- RMSNorm;
- Kimi-VL MLA projections/rope path;
- streamed V2 MoE;
- residual connection;
- compare a whole decoder layer against a lightweight PyTorch oracle.

Only after a real decoder layer matches should routed experts be converted to MXFP4/Q4.
