# Kimi-VL low-RAM C runtime — Windows real-model user guide

This guide is the recommended path for running the repository on a normal Windows laptop with
CPU-only inference and limited RAM. It covers installation, downloading the exact validated
Kimi-VL checkpoint revision, packing the low-RAM Q8 runtime, preflight checks, text-only chat,
image + text chat, RAM/cache tuning, and troubleshooting.

The stable inference frontend in this guide uses the exact production V9 decoder. The
self-speculative decoding sweep in `src/self_spec_sweep.c` is a research benchmark and is **not**
enabled by `tools/kvl_vl_chat.py` unless it is explicitly integrated and revalidated later.

## 1. What you are about to run

Model:

```text
moonshotai/Kimi-VL-A3B-Instruct
validated revision:
398eede0903cd983a2bfa0cc634e9ac1d843f375
```

Runtime storage format used by this guide:

```text
trunk / attention / routers / shared experts / LM head    BF16
routed MoE experts                                       row-wise Q8
MoonViT + multimodal projector                           BF16
compressed MLA history                                   FP32 exact target state
```

Typical packed weight sizes from the validated checkpoint are approximately:

```text
trunk.bin       2.916 GiB
experts.bin    13.4 GiB     (Q8 routed experts)
vision.bin      0.834 GiB
--------------------------------
packed weights ~17.2 GiB
```

The model is not loaded completely into RAM. Large weights remain on SSD/NVMe; routed experts
are streamed through a hard-budget cache. Windows uses the native no-buffering/direct-I/O path
when the filesystem and alignment permit it.

## 2. Recommended machine setup

For the first real test:

- Windows 10/11 x64.
- 64-bit Python 3.11 or newer.
- CMake.
- Visual Studio 2022 Build Tools with **Desktop development with C++** / MSVC x64 tools.
- Git.
- Local SSD/NVMe, preferably NTFS.
- At least roughly 40–50 GiB free during preparation. More is better.
- 16 GiB system RAM is a practical target for the low-RAM path.

Keep the packed runtime on a normal local SSD path. Avoid putting the large `.bin` files inside
OneDrive/sync folders, network drives, or compressed folders for the first test.

For a 16 GiB laptop, close LM Studio and any other loaded LLM before running this runtime.

## 3. Clone the current test branch

Open PowerShell:

```powershell
git clone https://github.com/tuantran00541-spec/In-c-test.git
cd In-c-test
git switch research/v9-two-turn-vi-chat
git pull
```

Do not download model weights into the Git history. The repository already ignores
`checkpoints/`, `packed/`, `*.safetensors`, `*.bin`, and `*.idx`.

## 4. Easiest Windows setup — one PowerShell command

The repository contains `tools/windows_setup_q8.ps1`. It creates a virtual environment,
installs CPU-only PyTorch and helper dependencies, builds the C runtime with AVX2, downloads the
pinned model revision with a bounded source-shard working set, packs the Q8 runtime, and runs a
preflight doctor.

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\windows_setup_q8.ps1
```

Default output locations:

```text
checkpoints\kimi-vl-work       temporary/download working directory
packed\kimi-vl-a3b-q8          final runtime directory
.venv                           Python environment
build                           CMake build directory
```

The packer does **not** need all seven source shards to remain resident simultaneously. It packs
completed layers and deletes consumed source shards as it progresses. This is the recommended
path on a consumer SSD.

To keep all downloaded source shards instead:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\windows_setup_q8.ps1 -KeepSourceShards
```

Keeping source shards is not required for inference and uses substantially more disk space.

## 5. Manual setup — if you want every step visible

### 5.1 Create Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install CPU-only PyTorch:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Install the remaining user-facing dependencies:

```powershell
python -m pip install -r requirements-user.txt
```

### 5.2 Build the native C runtime

```powershell
cmake -S . -B build -DKVL_USE_AVX2=ON
cmake --build build --config Release --parallel 2
```

On a normal Visual Studio multi-config generator the Windows executables should appear under:

```text
build\Release\kvl_vision.exe
build\Release\kvl_generate.exe
build\Release\kvl_generate_vl.exe
```

If CMake says it cannot find a C/C++ compiler, install Visual Studio 2022 Build Tools and select
the C++ desktop workload, then reopen PowerShell.

### 5.3 Download and pack the pinned Q8 runtime

```powershell
python .\tools\prepare_kimi_vl_q8.py `
  .\checkpoints\kimi-vl-work `
  .\packed\kimi-vl-a3b-q8
```

This command pins both model weights and tokenizer/frontend files to:

```text
398eede0903cd983a2bfa0cc634e9ac1d843f375
```

The final runtime also contains `SOURCE_REVISION.txt` so you can verify provenance later.

Interrupted Hugging Face downloads can normally be resumed by running the same command again
with the same working directory.

## 6. Preflight before spending time on inference

Run:

```powershell
python .\tools\kvl_doctor.py .\packed\kimi-vl-a3b-q8 --build-dir .\build
```

A healthy setup ends with:

```text
PASS: runtime structure and native binaries look ready for inference
```

The doctor checks:

- required runtime files exist;
- trunk/vision/expert indexes have the expected format magic;
- the recorded checkpoint revision matches the validated revision;
- `kvl_vision` and `kvl_generate_vl` were built;
- packed weight sizes are readable.

It does not load the full model and is therefore cheap to run.

## 7. First real image + text test

Use a normal JPG or PNG stored on the local disk. For the very first test, keep generation short
and deterministic:

```powershell
python .\tools\kvl_vl_chat.py `
  .\packed\kimi-vl-a3b-q8 `
  "C:\path\to\your-image.jpg" `
  "Look at this image and describe the character and facial expression in one short English sentence." `
  --cache-mib 512 `
  --ram-mib 4096 `
  --max-new 8 `
  --temperature 0 `
  --seed 1 `
  --show-tokens
```

Why start with English? The pinned runtime has an exact English real-weight acceptance case,
while the released model itself has shown prompt/language-dependent semantic inconsistency on a
Vietnamese fixture. English is therefore the cleaner first diagnostic for separating runtime
problems from model behavior.

After the first test works, try Vietnamese:

```powershell
python .\tools\kvl_vl_chat.py `
  .\packed\kimi-vl-a3b-q8 `
  "C:\path\to\your-image.jpg" `
  "Hãy nhìn ảnh và mô tả ngắn gọn nhân vật cùng biểu cảm bằng tiếng Việt." `
  --cache-mib 512 `
  --ram-mib 4096 `
  --max-new 16 `
  --temperature 0 `
  --seed 1 `
  --show-tokens
```

The frontend performs these phases:

1. preprocess image into MoonViT patches;
2. run native C MoonViT/projector;
3. release vision-phase working memory;
4. build the official-style multimodal prompt with media tokens;
5. run the native 27-layer C text decoder;
6. stream generated token IDs and decode them to text.

## 8. What healthy stderr should look like

Useful lines include values similar to:

```text
[kvl-vl] grid=... media_tokens=... prompt_tokens=...
[kvl-vl] ... text_RAM_plan=.../4096 MiB ...
...
trunk_direct_io=yes expert_direct_io=yes
[kvl-vl] token=... dt=...s
[kvl-vl] timing vision=... first_text_token=... avg_next=... generated=...
```

`trunk_direct_io=yes expert_direct_io=yes` confirms the native direct/no-buffering paths are
active for the text stores.

Do not compare your seconds/token directly with GitHub Actions numbers. CPU model, power limits,
SSD, thermals, OS caching, and background processes differ. The important first-local-test gates
are: no crash, no uncontrolled RAM growth, valid media processing, tokens are produced, and the
runtime reports direct I/O when supported.

## 9. Text-only test

The same packed runtime can run without an image:

```powershell
python .\tools\kvl_chat.py `
  .\packed\kimi-vl-a3b-q8 `
  "2 + 2 bằng bao nhiêu? Trả lời thật ngắn." `
  --cache-mib 512 `
  --ram-mib 4096 `
  --max-new 8 `
  --temperature 0 `
  --seed 1 `
  --show-tokens
```

Text-only is useful when diagnosing whether a problem belongs to MoonViT/image preprocessing or
the text decoder/storage path.

## 10. Settings recommended for a 16 GiB laptop

Start here:

```text
--cache-mib 512
--ram-mib   4096
--max-new   8 or 16
--temperature 0
```

`--cache-mib` is the routed-expert cache hard budget. Raising it can reduce expert rereads but
uses more RAM. Lowering it saves memory but may increase SSD traffic. The validated baseline uses
512 MiB, so use 512 first.

`--ram-mib` is a conservative frontend planning budget for known text-phase allocations. The
frontend refuses a request if its plan exceeds the value. It is not a claim that total process
RSS can never exceed that exact number because runtime libraries, OS buffers, and allocator
overhead exist outside the planner.

`--max-new` controls generated length. Keep it small during first testing; increase after the
runtime is stable on your machine.

`--temperature 0` selects greedy deterministic generation and is best for debugging.

## 11. Optional OpenMP tuning

The CI real-weight gates currently use a small OpenMP thread count. On a hybrid laptop CPU,
more threads are not automatically faster because the workload is also SSD/memory-bandwidth
sensitive.

For the first run you can leave the environment untouched. Later, compare small values such as:

```powershell
$env:OMP_NUM_THREADS="2"
$env:OMP_DYNAMIC="FALSE"
```

or:

```powershell
$env:OMP_NUM_THREADS="4"
$env:OMP_DYNAMIC="FALSE"
```

Measure on the actual laptop before deciding which is best.

## 12. Common failures

### `cmake` is not recognized

Install CMake for Windows and ensure it is on `PATH`, then open a new PowerShell window.

### No C/C++ compiler / Visual Studio generator error

Install **Visual Studio 2022 Build Tools** with **Desktop development with C++**. Make sure the
x64 MSVC toolchain and a Windows SDK are selected.

### PowerShell blocks `.ps1`

Use the repository command with process-level bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\windows_setup_q8.ps1
```

This does not require permanently changing the machine-wide policy.

### Hugging Face download stops halfway

Run the same preparation command again with the same `checkpoints\kimi-vl-work` directory. Do
not delete the working directory unless you intentionally want to restart downloads.

### `RAM plan rejected`

Reduce in this order:

1. `--max-new`;
2. prompt/context length;
3. `--cache-mib` (for example 512 -> 384 or 256).

Only raise `--ram-mib` if you intentionally want the planner to permit a larger known working
set and the machine has enough real free RAM.

### Windows RAM approaches 100%

Stop other model runtimes first. Do not run LM Studio with another model resident at the same
time. Retry with a short prompt, `--max-new 4` or `8`, and a smaller expert cache if necessary.
Record Task Manager peak memory and the complete stderr for debugging.

### `trunk_direct_io=no` or `expert_direct_io=no`

First move the packed directory to a normal local NTFS SSD path and retry. Avoid cloud-sync,
network, compressed, and unusual virtual filesystem paths. The runtime can fall back when the
OS/filesystem does not accept the direct-I/O mode, but that is not the preferred low-RAM path.

### Image path works but output says image is missing / semantics look wrong

Confirm MoonViT actually ran and media tokens were reported. Then test the English diagnostic
prompt. The pinned released model has a documented Vietnamese semantic caveat even when the C
trajectory matches the released model. Do not treat every poor-language answer as a storage or
vision-runtime failure.

### Process is simply slow

CPU-only execution of this model is expected to be heavy. Use `--show-tokens` and report the
printed `vision`, `first_text_token`, `avg_next`, expert-cache statistics, and direct-I/O flags.
Those numbers let us identify whether the bottleneck is vision, prefill, decode compute, or SSD
expert streaming.

## 13. What to send back after your first laptop run

For a useful performance/debug report, copy these items:

```text
1. CPU model and RAM size
2. Windows version
3. SSD/NVMe model if known
4. exact git commit: git rev-parse HEAD
5. SOURCE_REVISION.txt contents
6. full command you ran
7. text_RAM_plan line
8. trunk_direct_io / expert_direct_io line
9. vision time
10. first_text_token time
11. avg_next time
12. generated token count / generated IDs if --show-tokens was enabled
13. peak RAM from Task Manager
14. final generated text
```

Do not send the model `.bin`/`.safetensors` files.

## 14. Cleaning up disk after a successful pack

If the default streaming preparer completed successfully, consumed source shards were already
deleted. Once `kvl_doctor.py` passes and you have successfully run inference, the working
checkpoint directory can be removed if you do not need it for repacking:

```powershell
Remove-Item -Recurse -Force .\checkpoints\kimi-vl-work
```

Keep:

```text
packed\kimi-vl-a3b-q8
build\Release
.venv
repository source
```

The packed runtime can be regenerated from the pinned Hugging Face checkpoint at any time.

## 15. Research self-speculative decoder

The branch also contains `kvl_self_spec_sweep`, which evaluates a same-model draft + exact target
verification design with INT8 MLA draft history and selected routed-expert skipping. This is for
benchmarking candidate acceleration masks. It is deliberately separate from the user-facing
exact generator until acceptance and speed are validated broadly enough.

For your first real laptop test, **do not use the self-spec sweep as the chat binary**. Use
`tools/kvl_vl_chat.py` exactly as shown above so the result is comparable with the pinned V9
acceptance path.
