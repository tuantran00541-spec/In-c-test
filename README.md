# Kimi-VL low-RAM CPU runtime

A CPU-first, low-RAM inference runtime for the official open-weight
`moonshotai/Kimi-VL-A3B-Instruct` model.

The project is designed for normal consumer laptops where the full model should **not** live in
RAM. Large weight stores remain on SSD/NVMe, routed MoE experts are streamed through a bounded
cache, MLA history is compressed, and the text/vision inference cores are native C.

The recommended user format keeps trunk/router/shared/vision weights in BF16 and stores routed
MoE experts as validated row-wise Q8.

> **Primary tested use case:** Windows 10/11 x64, CPU-only, local SSD/NVMe, about 16 GiB system
> RAM. Linux is also supported by the native runtime, including direct-I/O paths.

---

## 1. What is included

Current user-facing features:

- native C text decoder for Kimi-VL-A3B-Instruct;
- native C MoonViT + multimodal projector;
- Q8 routed-MoE expert store;
- sparse expert streaming from SSD/NVMe;
- bounded expert cache;
- automatic bounded trunk cache;
- compressed MLA KV/history state;
- Linux direct I/O and native Windows no-buffering/direct-I/O;
- AVX2 kernels on supported x86 CPUs;
- text-only command-line chat;
- image + text command-line chat;
- conservative RAM planner;
- local Anthropic-compatible API;
- local OpenAI-compatible API;
- SSE token streaming over HTTP;
- Windows one-command build/download/pack helper;
- structural runtime doctor/preflight tool.

The project deliberately does **not** store model weights in Git.

---

## 2. Validated model

The runtime is built around this exact official checkpoint:

```text
repository: moonshotai/Kimi-VL-A3B-Instruct
revision:   398eede0903cd983a2bfa0cc634e9ac1d843f375
```

Do not casually mix tokenizer files, configuration files, or weights from a different model
revision with the validated runtime.

`tools/prepare_kimi_vl_q8.py` pins the revision above and writes it into:

```text
SOURCE_REVISION.txt
```

inside the final packed runtime.

---

## 3. Packed runtime layout

Typical packed weight sizes from the validated checkpoint are approximately:

```text
trunk.bin       2.916 GiB   BF16 trunk/router/shared/global weights
experts.bin    13.4 GiB     row-wise Q8 routed experts
vision.bin      0.834 GiB   BF16 MoonViT + projector
--------------------------------
weight total   ~17.2 GiB
```

The final runtime directory also contains indexes and tokenizer/preprocessor assets, for example:

```text
packed/kimi-vl-a3b-q8/
  trunk.bin
  trunk.idx
  experts.bin
  experts.idx
  vision.bin
  vision.idx
  tiktoken.model
  tokenizer_config.json
  preprocessor_config.json
  SOURCE_REVISION.txt
```

The whole ~17.2 GiB packed model is **not** loaded into RAM at once.

---

## 4. Recommended Windows requirements

For the easiest path install:

- Windows 10/11 x64;
- 64-bit Python 3.11 or newer;
- Git;
- CMake;
- Visual Studio 2022 Build Tools;
- **Desktop development with C++** workload / x64 MSVC toolchain;
- a local SSD/NVMe, preferably NTFS.

Recommended practical resources:

```text
RAM:          ~16 GiB or more
free disk:    at least ~40-50 GiB during preparation
final pack:   ~17.2 GiB of model weights
```

Keep the large runtime files on a normal local SSD path for the first test. Avoid OneDrive,
network drives, compressed folders, or unusual virtual filesystems.

If another LLM is loaded in LM Studio/Ollama/etc., unload it before testing this runtime.

---

# QUICK START — WINDOWS

## 5. Clone the repository

Open PowerShell:

```powershell
git clone https://github.com/tuantran00541-spec/In-c-test.git
cd In-c-test
git switch main
git pull
```

You do **not** need a research branch for the current user-facing runtime.

---

## 6. One-command Windows setup

The easiest supported path is:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\windows_setup_q8.ps1
```

The script performs the complete preparation sequence:

1. creates `.venv`;
2. upgrades pip;
3. installs CPU-only PyTorch;
4. installs user-facing Python dependencies;
5. configures the native runtime with AVX2;
6. builds Release binaries;
7. downloads the exact pinned Kimi-VL revision;
8. packs the low-RAM Q8 runtime;
9. deletes consumed source shards by default;
10. runs `kvl_doctor.py`.

Default locations:

```text
.venv
build
checkpoints\kimi-vl-work
packed\kimi-vl-a3b-q8
```

### Keep the original downloaded source shards

If you want to keep the BF16/safetensor source shards after packing:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\windows_setup_q8.ps1 -KeepSourceShards
```

Keeping them is **not required for inference** and consumes substantially more disk space.

For a normal first installation, allowing the packer to delete consumed source shards is the
recommended path.

---

# MANUAL INSTALLATION

## 7. Create the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install CPU-only PyTorch:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Install the remaining dependencies:

```powershell
python -m pip install -r requirements-user.txt
```

The user requirements include the Hugging Face downloader, safetensors, NumPy, Pillow and
tiktoken.

---

## 8. Build the native C runtime

```powershell
cmake -S . -B build -DKVL_USE_AVX2=ON
cmake --build build --config Release --parallel 2
```

Typical Windows binaries are produced under:

```text
build\Release\kvl_generate.exe
build\Release\kvl_generate_vl.exe
build\Release\kvl_vision.exe
```

If CMake cannot find a compiler, install Visual Studio 2022 Build Tools with the C++ desktop
workload and reopen PowerShell.

---

## 9. Download and pack the model manually

```powershell
.\.venv\Scripts\python.exe .\tools\prepare_kimi_vl_q8.py `
  .\checkpoints\kimi-vl-work `
  .\packed\kimi-vl-a3b-q8
```

The preparation path uses a bounded source-shard working set rather than requiring every source
shard to remain on disk for the entire conversion.

If a Hugging Face download is interrupted, rerun the same command with the same work directory.
The downloader can normally resume instead of starting from zero.

To retain all source shards:

```powershell
.\.venv\Scripts\python.exe .\tools\prepare_kimi_vl_q8.py `
  .\checkpoints\kimi-vl-work `
  .\packed\kimi-vl-a3b-q8 `
  --keep-source-shards
```

---

# VERIFY THE INSTALLATION

## 10. Run the runtime doctor

Before spending time on real inference:

```powershell
.\.venv\Scripts\python.exe .\tools\kvl_doctor.py `
  .\packed\kimi-vl-a3b-q8 `
  --build-dir .\build
```

A healthy setup should end with:

```text
PASS: runtime structure and native binaries look ready for inference
```

The doctor checks the expected stores/indexes, model revision provenance, build outputs and basic
runtime structure without performing a long full-model generation.

Do not delete your source checkpoint working directory until the pack has completed and the
doctor passes.

---

# TEXT CHAT

## 11. First text-only test

Use a short deterministic prompt first:

```powershell
.\.venv\Scripts\python.exe .\tools\kvl_chat.py `
  .\packed\kimi-vl-a3b-q8 `
  "2 + 2 bằng bao nhiêu? Trả lời thật ngắn." `
  --cache-mib 512 `
  --ram-mib 4096 `
  --max-new 8 `
  --temperature 0 `
  --seed 1 `
  --show-tokens
```

Once this works, increase `--max-new` and use normal chat prompts.

### Important runtime options

`--cache-mib 512`

Routed-expert cache budget. The validated baseline uses 512 MiB. Larger values can reduce expert
rereads but consume more RAM.

`--trunk-cache-mib auto`

This is the default. The frontend automatically uses as much safe non-global trunk cache as it
can fit under the requested RAM plan, up to the full useful cache size. You normally do not need
to specify this option.

Use this only for debugging/streaming-baseline comparisons:

```text
--trunk-cache-mib 0
```

`--ram-mib 4096`

Conservative known-working-set planning budget for the text phase. The frontend rejects a
configuration if its calculated plan exceeds this value.

This is a planning contract, not a promise that Windows Task Manager RSS can never exceed exactly
4096 MiB; Python/runtime libraries, allocators and OS bookkeeping also consume memory.

`--max-new`

Maximum generated token count. Start with 8-16 while validating the machine.

`--temperature 0`

Greedy deterministic generation. Best for first tests and performance comparisons.

`--show-tokens`

Print token IDs and per-token timing information useful for debugging and benchmarking.

---

# IMAGE + TEXT CHAT

## 12. First multimodal test

Use a normal local JPG/PNG:

```powershell
.\.venv\Scripts\python.exe .\tools\kvl_vl_chat.py `
  .\packed\kimi-vl-a3b-q8 `
  "C:\path\to\image.jpg" `
  "Look at this image and describe it in one short sentence." `
  --cache-mib 512 `
  --ram-mib 4096 `
  --max-new 8 `
  --temperature 0 `
  --seed 1 `
  --show-tokens
```

Then try a Vietnamese prompt if desired:

```powershell
.\.venv\Scripts\python.exe .\tools\kvl_vl_chat.py `
  .\packed\kimi-vl-a3b-q8 `
  "C:\path\to\image.jpg" `
  "Hãy nhìn ảnh và mô tả ngắn gọn nội dung bằng tiếng Việt." `
  --cache-mib 512 `
  --ram-mib 4096 `
  --max-new 16 `
  --temperature 0 `
  --seed 1 `
  --show-tokens
```

The VL frontend performs:

```text
image
  -> Python image preprocessing
  -> native MoonViT/projector
  -> media embeddings
  -> official-style multimodal prompt
  -> native Kimi text decoder
  -> generated tokens
  -> decoded text
```

Vision and text are separate phases so vision working memory can be released before text decode.

---

# LOCAL HTTP API

## 13. Start the local API server

After the packed runtime and Release binaries exist:

```powershell
.\.venv\Scripts\python.exe .\tools\kvl_api.py `
  .\packed\kimi-vl-a3b-q8 `
  --host 127.0.0.1 `
  --port 8000 `
  --api-key local-kimi `
  --ram-mib 4096 `
  --cache-mib 512
```

Default model ID:

```text
kimi-vl-a3b-instruct-q8
```

Default local addresses:

```text
Anthropic base URL: http://127.0.0.1:8000
OpenAI base URL:    http://127.0.0.1:8000/v1
```

The API key may be supplied through either `x-api-key` or `Authorization: Bearer ...` depending
on the client contract.

---

## 14. Available endpoints

| Purpose | Endpoint |
| --- | --- |
| Health | `GET /healthz` |
| Model discovery | `GET /v1/models` |
| Anthropic Messages | `POST /v1/messages` |
| Anthropic token count | `POST /v1/messages/count_tokens` |
| OpenAI chat completions | `POST /v1/chat/completions` |

Anthropic and OpenAI chat endpoints support SSE text streaming.

The HTTP contract is tested on both Windows and Linux in GitHub Actions.

---

## 15. Claude Code / Anthropic-compatible shell smoke test

Point an Anthropic-compatible client at the local server:

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8000"
$env:ANTHROPIC_API_KEY = "local-kimi"

claude --model kimi-vl-a3b-instruct-q8
```

### Current Claude Code limitation

The present API revision is intended for local chat/routing/streaming tests. It can accept tool
history as readable conversation context, but it does **not yet emit native tool-use calls**.
Therefore Claude Code is currently useful as a shell/UI smoke test, not yet as a fully reliable
coding agent backed by Kimi.

---

## 16. OpenAI-compatible clients and harnesses

Use:

```text
base_url = http://127.0.0.1:8000/v1
api_key  = local-kimi
model    = kimi-vl-a3b-instruct-q8
```

Example PowerShell request:

```powershell
$headers = @{ Authorization = "Bearer local-kimi" }
$body = @{
  model = "kimi-vl-a3b-instruct-q8"
  messages = @(
    @{ role = "user"; content = "Xin chào. Trả lời thật ngắn." }
  )
  max_tokens = 32
  temperature = 0
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/chat/completions" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

### Current serving boundary

The Python API server remains alive, but each inference request currently launches the validated
`kvl_generate` CLI once. Requests are serialized so two clients cannot accidentally launch two
multi-GiB inference jobs at the same time and exhaust RAM.

A future serving optimization can make the C model engine persistent across requests without
changing these HTTP URLs.

The current HTTP API is **text-only**. Use `kvl_vl_chat.py` for image inference until the
multimodal HTTP contract is added.

---

# RAM, CACHE AND SSD BEHAVIOR

## 17. Recommended starting configuration for a 16 GiB laptop

Start with:

```text
expert cache:       512 MiB
RAM plan:           4096 MiB
trunk cache:        auto
max-new:            8-16 for first test
temperature:        0 for diagnostics
```

The automatic trunk-cache planner tries to retain reusable trunk tensors in RAM while still
respecting the configured text working-set budget.

For short contexts under the standard 4096 MiB plan, it can use up to roughly the full useful
non-global trunk cache (~1706 MiB). As context/workspace requirements grow, the planner reduces
that cache automatically rather than blindly exceeding the budget.

Large model stores remain SSD/NVMe resident. Routed experts continue to stream through the
bounded expert cache.

---

## 18. Direct I/O

The runtime supports:

```text
Linux:    direct-I/O path
Windows:  FILE_FLAG_NO_BUFFERING-based native path
```

Healthy text runs should report direct-I/O status such as:

```text
trunk_direct_io=yes expert_direct_io=yes
```

If direct I/O is unavailable, move the packed runtime to a normal local SSD/NTFS path and avoid
network/cloud/compressed folders.

---

# PERFORMANCE NOTES

## 19. What has actually been accelerated

Current performance work includes:

- Q8 routed experts;
- native Q8 execution;
- AVX2 expert kernels;
- SSD-resident sparse-MoE streaming;
- hard-budget expert caching;
- layer-aware expert pinning/hysteresis research;
- compressed MLA history;
- bounded reusable trunk caching;
- direct/no-buffering weight I/O;
- layer-major/batched prompt prefill;
- native MoonViT path.

The largest measured text-decode improvement so far came from eliminating repeated trunk reads.
In a controlled hosted-Windows full-model A/B, the same generated token sequence and expert-I/O
trajectory were preserved while median average-next-token time changed from:

```text
streamed trunk:  7.3450 s/token
cached trunk:    2.9605 s/token
ratio:           2.481x
reduction:       ~59.7%
```

The cached run eliminated about **1705.67 MiB of repeated trunk reads per decode forward** after
the cache was populated.

These are **hosted runner measurements, not a prediction for your laptop**. CPU, SSD, thermals,
power limits and memory bandwidth can change the result substantially.

A previous scalar-vs-AVX2 controlled test improved first-token time substantially, but did not
materially improve decode by itself. That result helped identify storage/cache traffic as the
larger decode bottleneck.

Always benchmark on the actual target machine.

---

# BENCHMARKING YOUR MACHINE

## 20. What to record

For a useful local performance report, record:

```text
CPU model
RAM size
SSD/NVMe model if known
Windows/Linux version
git rev-parse HEAD
SOURCE_REVISION.txt
exact command used
prompt length
RAM-plan line
trunk cache resident/hit/read statistics
expert cache/read statistics
trunk_direct_io / expert_direct_io
vision time (VL only)
first-token time
average next-token time
generated token count
peak RAM from Task Manager / system monitor
final output
```

Do not compare two runs unless prompt, temperature, seed, max-new, cache budgets and model files
are controlled.

For deterministic performance comparisons use:

```text
--temperature 0
--seed 1
--show-tokens
```

---

# CLEANUP AFTER SUCCESSFUL PACKING

## 21. What can be deleted

After all of the following are true:

1. packing completed successfully;
2. `kvl_doctor.py` passes;
3. at least one real text or VL inference run succeeds;

then the temporary checkpoint working directory may be removed if you do not need to repack:

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

If you used the default streaming preparer, many consumed source shards may already have been
deleted automatically during packing.

Do **not** delete the source checkpoint before the final pack and doctor check complete.

---

# TROUBLESHOOTING

## 22. `cmake` is not recognized

Install CMake and ensure it is in `PATH`, then open a new PowerShell window.

---

## 23. CMake cannot find an MSVC/C++ compiler

Install Visual Studio 2022 Build Tools and select:

```text
Desktop development with C++
```

Make sure the x64 MSVC tools and Windows SDK are installed.

---

## 24. PowerShell blocks `.ps1`

Use:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\windows_setup_q8.ps1
```

This avoids requiring a permanent machine-wide policy change.

---

## 25. Hugging Face download stopped midway

Rerun the same preparation command using the same checkpoint work directory. Do not delete the
partial working directory unless you intentionally want to redownload everything.

---

## 26. `RAM plan rejected`

Reduce memory pressure in roughly this order:

1. shorten the prompt/context;
2. reduce `--max-new`;
3. allow `--trunk-cache-mib auto` to manage itself;
4. reduce `--cache-mib` from 512 to 384 or 256 if necessary.

Only increase `--ram-mib` when the machine genuinely has enough free RAM and you intentionally
want a larger known working set.

---

## 27. RAM climbs close to 100%

Stop other loaded LLM runtimes first. Retry with:

```text
--max-new 4 or 8
--cache-mib 256 or 384
--ram-mib 4096
```

Also keep trunk cache on `auto` rather than manually forcing a value too large for the request.

---

## 28. `trunk_direct_io=no` / `expert_direct_io=no`

Move the packed runtime to a normal local SSD/NTFS location and retry.

Avoid:

- OneDrive/sync folders;
- network shares;
- compressed folders;
- unusual virtual filesystems.

---

## 29. Text works but image chat fails

Run text-only `kvl_chat.py` first. If text succeeds, check:

- `vision.bin` and `vision.idx` exist;
- the image path is valid;
- Pillow can read the file;
- `kvl_vision` was built;
- `kvl_vl_chat.py` reports a valid image grid/media-token count.

This separates MoonViT/preprocessing failures from the text decoder/storage path.

---

## 30. The model is simply slow

This is a ~16.4B-total-parameter sparse-MoE model running CPU-only with large weights resident on
SSD. Performance is expected to depend heavily on CPU, SSD, DDR bandwidth, thermals and cache
behavior.

Use `--show-tokens` and collect the printed timing/cache/direct-I/O statistics before assuming a
specific component is at fault.

---

# RUNTIME ARCHITECTURE

## 31. High-level data path

```text
Official Kimi-VL checkpoint
        |
        v
low-RAM packer
        |
        +--> trunk.bin / trunk.idx       BF16
        +--> experts.bin / experts.idx   Q8 routed experts
        +--> vision.bin / vision.idx     BF16

Text request
        |
        v
tokenizer/frontend
        |
        v
native C decoder
        |
        +--> reusable bounded trunk cache
        +--> compressed MLA state
        +--> router
        +--> bounded SSD expert cache
        +--> Q8 routed MoE kernels
        |
        v
LM head -> generated tokens

Image request
        |
        v
image preprocessing
        |
        v
native MoonViT + projector
        |
        v
media embeddings -> native text decoder
```

The released text model has 27 decoder layers. Layer 0 is dense; layers 1-26 use routed MoE with
64 routed experts and top-6 routing per token.

---

# CURRENT LIMITATIONS

## 32. Things that are not finished yet

- HTTP image input is not yet exposed; use `kvl_vl_chat.py` for vision.
- Native Anthropic/OpenAI tool-call emission is not yet implemented.
- The local API process is persistent, but the underlying C generator is currently launched per
  inference request; model/cache persistence across HTTP requests is a future serving
  optimization.
- CPU-only MoonViT still has additional optimization headroom.
- Absolute GitHub Actions timings must not be treated as target-laptop benchmark numbers.
- Experimental acceleration branches are not automatically promoted into the user path until
  they preserve correctness and demonstrate a real end-to-end win.

---

# ADDITIONAL DOCUMENTATION

## 33. More detailed references

Windows-specific guide:

[`docs/USER_GUIDE_WINDOWS.md`](docs/USER_GUIDE_WINDOWS.md)

Local API details:

[`docs/local_api.md`](docs/local_api.md)

Historical real-weight validation notes:

[`spec/REAL_V9_RESULT.md`](spec/REAL_V9_RESULT.md)

The README is intended to be sufficient for normal installation and usage. The documents above
contain additional implementation/debug detail.

---

# REPOSITORY POLICY

## 34. Weight and validation policy

- Never commit model/checkpoint/packed weight files into Git.
- Keep the pinned official checkpoint revision explicit.
- Do not weaken real-weight correctness gates to make an optimization appear faster.
- Keep experimental acceleration isolated until it preserves the accepted target behavior and
  demonstrates an end-to-end benefit.
- Hosted benchmark numbers are evidence for controlled A/B comparisons, not promises for another
  machine.

---

## 35. Minimal first-run checklist

If you only want the shortest safe path:

```text
1. Clone main.
2. Run tools/windows_setup_q8.ps1.
3. Wait for kvl_doctor PASS.
4. Run kvl_chat.py with max-new 8, temperature 0.
5. Run kvl_vl_chat.py with one local JPG/PNG.
6. Confirm tokens are generated and direct-I/O flags look healthy.
7. Start kvl_api.py if you want a local HTTP endpoint.
8. Only then delete the temporary checkpoint working directory.
```

That is the current supported path for using the Kimi-VL low-RAM runtime.
