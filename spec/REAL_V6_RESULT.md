# Real Kimi-VL V6 result

Status: **PASS**

Date: 2026-08-28

Workflow: `.github/workflows/real-v6-full-decode.yml`

GitHub Actions run: `33137744826` (run #2)

V6 proves persistent token-by-token decoding for the complete official
`moonshotai/Kimi-VL-A3B-Instruct` text model using compressed MLA state. The final probe
compares two execution paths over the same packed BF16 runtime stores:

```text
A. two-token causal prefill through all 27 decoder layers
B. token 0 -> persistent compressed MLA state -> token 1 incremental decode
```

Both paths include the dense first layer, all 26 sparse-MoE layers, final RMSNorm and the
163,840-way LM head.

## Official model semantics

The accepted baseline explicitly uses:

```text
hidden size          2048
attention heads        16
qk NoPE dim           128
qk RoPE dim            64
V head dim             128
KV LoRA rank           512
RoPE theta          800000
RMSNorm epsilon       1e-5
layers                  27
routed experts           64
experts/token              6
```

An earlier full run (#1) used `1e-6` in the end-to-end probe. Although its prefill and
incremental paths matched each other, it is intentionally not treated as the official V6
baseline. Run #2 prints `rms_eps=1.0e-05` in the result line and is the accepted run.

## Compressed MLA state

V6 first established an expanded K/V reference state, then replaced it with MLA-native
compressed history. Per historical token and decoder layer the compressed state stores the
normalized 512-dimensional latent plus the 64-dimensional RoPE component instead of
materialized per-head K/V.

```text
compressed payload / layer / token = (512 + 64) x 4 = 2304 bytes (FP32)
allocated state for 27 layers, capacity 2 = 125,280 bytes
```

The synthetic four-token regression matched causal prefill with
`6.98491931e-10` maximum absolute error and passed ASan/UBSan. An official layer-1 probe at
production dimensions used 9,248 bytes of compressed state versus 81,960 bytes for the
expanded reference (~8.862x smaller), with `4.42378223e-09` maximum error and direct I/O.

## Complete runtime pack

The final V6 workflow rebuilt the complete official text backing stores with the same
bounded-source converter used by V5:

```text
27 decoder layers packed
1664 routed expert records

trunk.bin       2.916 GiB
trunk.idx      18,240 bytes
experts.bin    26.812 GiB
experts.idx   133,168 bytes
```

All seven source safetensor shards were removed after their dependent tensors had been
packed; no source shard remained resident at the end of conversion.

## Full two-token result

The test sequence is token ids `[1, 1008]`; `1008` is the V5 argmax produced from token id
`1`. The incremental path maintains one persistent compressed MLA state per decoder layer.

```text
tokens                    2
rms_eps              1.0e-05
compressed_state_bytes 125280
worst layer               26
worst token                 1
hidden max abs      7.62939453e-06
logits max abs      1.90734863e-06
logits RMS          4.55979847e-07
prefill argmax             1609
incremental argmax         1609
trunk direct I/O            yes
expert direct I/O           yes
```

The routed-expert cache remained hard-capped at 512 MiB:

```text
cache slots          31
slot size         16.50 MiB
arena        511.50 / 512.00 MiB
compute requests     624
physical reads       595
expert bytes read 9817.50 MiB
expert evictions     564
read failures          0
```

The cache report's compute-side hit count occurs after `getmany()` prefetch. The 595
physical reads and 564 evictions demonstrate that the test is not keeping the full routed
expert working set resident.

The measured CI read rate is runner-specific and is not a laptop performance benchmark.

## What V6 proves

V6 proves that the complete official Kimi-VL text decoder can preserve causal history in
compressed MLA state and advance token by token through all 27 layers while matching the
full causal-prefill path to close FP32 numerical tolerance and producing the identical next
argmax token.

The remaining work for actual text generation is primarily product/runtime plumbing:
tokenizer integration, a sampling loop, EOS/stop handling and a user-facing CLI. BF16-to-
quantized expert kernels and MoonViT vision remain separate later milestones.
