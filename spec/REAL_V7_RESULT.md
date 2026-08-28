# REAL V7 RESULT — official Kimi-VL text generation

Status: **PASS**

Accepted workflow: `Real Kimi-VL V7 text generation`, run `33141354065`.

V7 closes the first usable text-generation milestone. The runtime now performs:

```text
user text
  -> official-compatible tiktoken BPE + Kimi chat template
  -> token ids
  -> C prompt prefill
  -> 27 decoder layers
  -> compressed persistent MLA state
  -> SSD-streamed routed MoE
  -> final RMSNorm + 163,840-way LM head
  -> greedy / temperature sampling
  -> incremental autoregressive decode
  -> token ids
  -> text decode
```

## Official tokenizer oracle

The V7 tokenizer implementation was compared against `AutoTokenizer` from
`moonshotai/Kimi-VL-A3B-Instruct` on English, Vietnamese, Chinese, punctuation/newlines and
the released chat template. Token IDs and decoded text matched the official tokenizer.

The accepted workflow reported:

```text
PASS: tokenizer + text chat template match official oracle; chat_tokens=29
```

## Full runtime pack

The same complete BF16 text pack used by V5/V6 was rebuilt from the official checkpoint:

```text
trunk.bin       2.916 GiB
trunk.idx      18,240 bytes
experts.bin    26.812 GiB
experts.idx   133,168 bytes
routed expert records = 1664 = 26 x 64
```

All seven source safetensor shards were deleted after their tensors were packed.

## Real generation

Prompt:

```text
Reply with exactly one short word: hello
```

The Kimi chat template produced 24 prompt tokens. `kvl_generate` then ran the complete text
model and generated six greedy tokens:

```text
19180 0 3653 691 374 8593
```

Decoded output:

```text
Hello! How can I assist
```

The response did not obey the "one word" instruction, but it is coherent model text produced
end-to-end through the custom runtime rather than an oracle or Transformers model.

## Runtime state and I/O

```text
prompt tokens          24
new tokens              6
final context           29
compressed MLA state 1.78 MiB
expert cache          512 MiB
cache slots             31
trunk direct I/O       yes
expert direct I/O      yes
physical expert reads 4524
expert bytes read   74646 MiB
expert evictions      4493
read failures            0
```

The cache-side compute hit counter is recorded after `getmany()` prefetch; physical-read and
eviction counters show that the routed expert working set was repeatedly streamed rather than
resident.

## Performance observation

This workflow intentionally prioritizes correctness over speed. Prompt prefill is currently
implemented by calling the one-token incremental path for every prompt token.

On the GitHub runner used by the accepted workflow:

```text
24-token prompt prefill  ~264 s
subsequent decode token  ~10.8 s/token
```

These numbers are runner- and thread-count-specific and are not a laptop performance claim.
They do identify V8's primary bottlenecks: token-by-token prefill, scalar/double-accumulation
BF16 GEMV and repeated trunk/expert I/O.

## V7 conclusion

V7 proves that the official Kimi-VL text model can be converted to the project's aligned
SSD-backed format and can generate coherent autoregressive text using the custom C decoder
under a hard routed-expert cache budget.

Next milestone: V8 practical laptop runtime — optimized prefill/kernels, Windows portability,
hard total-memory planning and production-oriented packaging — while retaining V5/V6/V7 as
correctness baselines.
