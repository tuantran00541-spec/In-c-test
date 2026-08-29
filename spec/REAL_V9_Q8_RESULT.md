# Kimi-VL V9 routed-expert Q8 result

Q8 is an opt-in routed-expert storage and compute path for the completed V9 runtime. The
release BF16 format remains the default. Router, shared experts, attention/trunk weights,
LM head and MoonViT/projector remain BF16; only the 1664 routed expert gate/up/down matrices
use symmetric signed-int8 weights with one FP32 scale per output row.

Pinned checkpoint revision used for the full image benchmark:

```text
moonshotai/Kimi-VL-A3B-Instruct
398eede0903cd983a2bfa0cc634e9ac1d843f375
```

Primary full Q8 benchmark evidence: GitHub Actions run `33230203100` on the research branch.
The exact V9 English image/prompt produced the same 16 generated token IDs as the BF16 V9
baseline:

```text
1008 6162 924 4393 11 98717 11002 316 261 21478 528 54275 37632 11276 13 163586
```

Decoded output was also identical:

```text
The character has large, expressive eyes and a surprised or shocked facial expression.
```

Measured on the same two-thread GitHub runner class:

| metric | BF16 V9 | routed-expert Q8 |
| --- | ---: | ---: |
| expert store | 26.812 GiB | 13.438 GiB |
| 512 MiB expert-cache slots | 31 | 61 |
| first text token | 134.568 s | 41.512 s |
| average following token | 10.867 s | 7.592 s |
| text total for 16 tokens | 297.575 s | 155.406 s |
| physical expert traffic | 88.65 GiB | 31.43 GiB |
| direct-I/O failures | 0 | 0 |

This corresponds to roughly 3.24x faster first-token latency, 1.43x faster subsequent-token
latency and 1.91x lower total text time in that benchmark. These GitHub-runner numbers are
regression evidence, not a laptop performance guarantee.

Pre-merge numerical gates were layered rather than relying only on decoded text: the synthetic
row-wise Q8 kernel test stayed below 2% relative RMSE, a released routed expert stayed below 4%
full-MLP relative RMSE, and a released router + top-6 routed experts + shared expert stayed below
5% relative RMS with router IDs preserved. Windows MSVC/AVX2/direct-I/O compile/runtime checks
also passed on the research branch.

The exact Vietnamese V9 fixture is retained as a quantization-preservation gate. Its known
language/prompt-dependent semantic behavior is documented separately in the V9 result; Q8 must
not be credited with fixing or causing that upstream-model behavior. The Q8 merge criterion is
that its generated trajectory remains equal to the accepted BF16 trajectory for the same pinned
fixture.

To create a complete Q8 runtime from an already-downloaded official checkpoint:

```sh
python tools/pack_full_model.py /path/to/Kimi-VL-A3B-Instruct /path/to/kimi-vl-v9-q8 \
  --expert-format q8
```

For bounded-download conversion, `tools/pack_full_text.py` also accepts `--expert-format q8`.
No checkpoint or packed model weights are committed to git.
