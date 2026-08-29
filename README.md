# kimi-vl-lowram v9

CPU-first low-RAM inference runtime for `moonshotai/Kimi-VL-A3B-Instruct`, using aligned
SSD/NVMe-backed weight stores, streamed sparse-MoE experts, compressed MLA state and a native
C MoonViT + text decoder.

The exact V9 image + text path is real-weight validated on the pinned Kimi-VL checkpoint. The
recommended consumer-machine format keeps trunk/router/shared/vision weights BF16 and stores
only routed MoE experts as validated row-wise Q8, reducing the routed-expert store from about
26.8 GiB to **13.438 GiB**.

> **Windows laptop users:** start with
> [`docs/USER_GUIDE_WINDOWS.md`](docs/USER_GUIDE_WINDOWS.md). It covers prerequisites,
> one-command setup, pinned model download/packing, preflight, text-only inference, real
> image+text inference, RAM/cache tuning and troubleshooting.

## Validated model

```text
repository: moonshotai/Kimi-VL-A3B-Instruct
revision:   398eede0903cd983a2bfa0cc634e9ac1d843f375
```

Do not mix weights and tokenizer/frontend assets from unrelated revisions when reproducing the
validated path. `tools/prepare_kimi_vl_q8.py` pins both to the revision above and records it in
`SOURCE_REVISION.txt`.

## Current packed Q8 layout

Official released weights pack to approximately:

```text
trunk.bin       2.916 GiB   BF16 trunk/router/shared/global weights
experts.bin    13.438 GiB   row-wise Q8 routed experts
vision.bin      0.834 GiB   BF16 MoonViT + projector
--------------------------------
weight total   ~17.188 GiB
```

The model is not loaded completely into RAM. Large stores remain on SSD/NVMe; routed experts
are streamed through a hard-budget cache. No checkpoint or packed model weight is stored in
Git.

## Fastest Windows path

Install Git, 64-bit Python 3.11+, CMake, and Visual Studio 2022 Build Tools with the C++ desktop
workload. Then in PowerShell:

```powershell
git clone https://github.com/tuantran00541-spec/In-c-test.git
cd In-c-test
git switch research/v9-two-turn-vi-chat
powershell -ExecutionPolicy Bypass -File .\tools\windows_setup_q8.ps1
```

The setup helper:

1. creates `.venv`;
2. installs CPU-only PyTorch and user-facing Python dependencies;
3. builds the native C runtime with AVX2;
4. downloads the exact pinned model revision;
5. packs MoonViT and the Q8 text runtime with a bounded source-shard working set;
6. deletes consumed source shards by default;
7. runs `tools/kvl_doctor.py` before declaring the runtime ready.

The final packed runtime defaults to:

```text
packed\kimi-vl-a3b-q8
```

## First real image + text run

Keep the first diagnostic short and deterministic:

```powershell
.\.venv\Scripts\python.exe .\tools\kvl_vl_chat.py `
  .\packed\kimi-vl-a3b-q8 `
  "C:\path\to\image.jpg" `
  "Look at this image and describe the character and facial expression in one short English sentence." `
  --cache-mib 512 `
  --ram-mib 4096 `
  --max-new 8 `
  --temperature 0 `
  --seed 1 `
  --show-tokens
```

For a 16 GiB laptop, `--cache-mib 512 --ram-mib 4096 --max-new 8` is the recommended first
configuration. Close other loaded LLM runtimes before testing.

A healthy text run should eventually report direct-I/O status such as:

```text
trunk_direct_io=yes expert_direct_io=yes
```

and `tools/kvl_vl_chat.py` prints vision time, first-token latency, average next-token latency,
generated IDs when requested, and a conservative text working-set plan.

For text-only testing:

```powershell
.\.venv\Scripts\python.exe .\tools\kvl_chat.py `
  .\packed\kimi-vl-a3b-q8 `
  "2 + 2 bằng bao nhiêu? Trả lời thật ngắn." `
  --cache-mib 512 --ram-mib 4096 --max-new 8 --temperature 0 --show-tokens
```

See the Windows guide for the manual install/pack path and troubleshooting.

## Low-RAM pack path

If you do not use the PowerShell helper, the recommended pack command is:

```powershell
python .\tools\prepare_kimi_vl_q8.py `
  .\checkpoints\kimi-vl-work `
  .\packed\kimi-vl-a3b-q8
```

Unlike the old full-snapshot-first workflow, this path does not require all seven source
safetensor shards to coexist for the entire conversion. `pack_full_text.py` completes layers as
soon as their required shard set is available and deletes consumed shards unless
`--keep-source-shards` is requested.

Run the cheap structural preflight before inference:

```powershell
python .\tools\kvl_doctor.py .\packed\kimi-vl-a3b-q8 --build-dir .\build
```

## V9 runtime architecture

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
  routed expert gate/up/down matrices (Q8 in recommended user pack)

vision.bin / vision.idx
  MoonViT patch embedding
  27 vision blocks
  final/projector normalization
  multimodal projector
```

Low-RAM execution properties:

- model weights remain SSD/NVMe resident;
- routed experts are streamed through a bounded LRU/prefetch cache;
- Linux direct I/O and Windows `FILE_FLAG_NO_BUFFERING` paths are supported;
- MLA history stores compressed latent + RoPE state rather than expanded historical K/V;
- multimodal prompt prefill is layer-major/batched;
- image features cross into the decoder through a BF16 media boundary;
- vision and text run as separate phases so vision working memory can be released before text
  decode;
- the Python frontend rejects text configurations whose conservative known-working-set plan
  exceeds `--ram-mib`.

For the released text dimensions, the exact compressed MLA target history uses latent rank 512
plus RoPE component 64.

## Real-weight evidence

The Q8 official short gate runs the real pinned Kimi-VL weights, not a toy model. The current
validated gate covers:

- Q8 official packing with a streamed source-shard working set;
- exact four-token layer-major target verification with `logits_max=0` and `state_max=0`;
- direct I/O enabled for trunk and expert stores;
- exact English multimodal token/output parity;
- a Vietnamese token-trajectory preservation fixture;
- synthetic scalar/AVX2 and Windows MSVC portability gates.

On the pinned English image fixture the accepted generated IDs are:

```text
1008 6162 924 4393 11 98717 11002 316 261 21478 528 54275 37632 11276 13 163586
```

with output:

```text
The character has large, expressive eyes and a surprised or shocked facial expression.
```

The Vietnamese fixture is deliberately treated as a **trajectory-preservation** gate rather
than a semantic-quality claim. The released model can produce semantically poor Vietnamese
behavior for that fixture even when the C runtime follows the accepted target trajectory.

Additional historical evidence lives in [`spec/REAL_V9_RESULT.md`](spec/REAL_V9_RESULT.md).

## Self-speculative research status

This research branch also contains `kvl_self_spec_sweep`, a same-model speculative decoder lab
using an INT8 MLA draft state and selective routed-expert skipping. The exact target verifier
and target commit remain exact.

The latest focused real-weight sweep found that keeping routed layer 22 while skipping the other
12 layers in the aggressive mask restores **4/4** draft acceptance and reduces draft expert
traffic to about **2084 MiB / 252 reads**, but the measured full cycle was still **0.978x** the
serial baseline on that runner. In other words: close, but **not a speedup**.

Therefore self-speculation is intentionally **not** wired into the user-facing chat command.
`tools/kvl_vl_chat.py` uses the exact V9 generation path so a first laptop test remains directly
comparable with the validated baseline.

## Development/regression probes

Useful local probes include:

```text
kvl_mla_streaming_probe
kvl_spec_q8_lab_probe
kvl_exact_block_verify
kvl_self_spec_sweep            research only
```

The synthetic workflow builds scalar/AVX2 Linux and AVX2 Windows MSVC variants. The official
short workflow additionally packs and runs the pinned real model.

## Repository policy

- Model/checkpoint/packed weight files stay outside Git.
- Exact V9 real-weight baselines should not be weakened by later architecture experiments.
- New acceleration ideas remain research-only until they preserve the target trajectory/state
  and demonstrate an actual end-to-end win.
