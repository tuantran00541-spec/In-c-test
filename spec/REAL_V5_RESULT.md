# Real Kimi-VL V5 result

Status: **PASS**

Date: 2026-08-28

Workflow: `.github/workflows/real-v5.yml`

GitHub Actions run: `33134882451`

V5 is the first end-to-end text-forward milestone for the complete official
`moonshotai/Kimi-VL-A3B-Instruct` language model. The runtime consumes only the packed
backing stores after conversion; the raw safetensor source shards are deleted by the
bounded-working-set converter.

## Complete runtime pack

```text
27 decoder layers packed
26 sparse-MoE layers
64 routed experts / MoE layer
1664 routed expert records

trunk.bin       2.916 GiB
trunk.idx      18,240 bytes
experts.bin    26.812 GiB
experts.idx   133,168 bytes
```

The converter downloaded the seven official checkpoint shards in filename order, packed
any layer whose complete tensor set was available, and removed source shards as soon as
no unfinished layer still depended on them. Layers that cross a shard boundary therefore
require at most a two-shard source working set. At completion the source model directory
contained only metadata/cache files (~640 KiB), while the runtime pack remained on disk.

## Oracle

The Python reference mmap'd the same `trunk.bin` and `experts.bin` consumed by C. It did
not reopen the original safetensors. For token id `1` it ran embedding -> all 27 decoder
layers -> final RMSNorm -> all 163,840 LM-head logits.

```text
expected argmax token = 1008
expected max logit    = 6.769507884979248
```

V5 intentionally uses sequence length 1 to isolate the complete weight/decoder/logit path.
The real four-token V3 test already validates Kimi-VL RoPE and causal attention. Persistent
multi-token KV state is the next milestone.

## C result

The C runtime used a hard 512 MiB routed-expert cache:

```text
cache slots       31
slot size         16.50 MiB
arena             511.50 / 512.00 MiB
MoE selections    156  (26 layers x top-6)
physical reads    156
expert bytes read 2574.00 MiB
expert evictions  125
read failures     0
```

Numerical comparison against the packed Torch oracle:

```text
layers              27
worst layer          26
layer max abs        0.00010681152
logits max abs       2.8252602e-05
logits RMS           4.0282628e-06
C argmax             1008
reference argmax     1008
trunk direct I/O     yes
expert direct I/O    yes
```

The cache report's `hit=156, miss=0` refers to compute-side `get()` calls after each
`getmany()` has prefetched the selected expert. The run still performed 156 physical
expert reads and 125 evictions; this is not a 100% cross-token cache-reuse result.

The measured 444.6 MiB/s physical expert-read rate is specific to the GitHub Actions runner
and is not a laptop performance claim.

## What V5 proves

V5 proves that the complete official Kimi-VL text weights can be converted into bounded,
aligned SSD backing stores and forwarded by the C runtime without loading the whole model
into RAM, while retaining the same top output token and close elementwise logits as the
packed PyTorch oracle.

It does **not** yet prove generation. Incremental MLA/KV state, tokenizer integration and
sampling are still required before the runtime can generate a text sequence token by token.
