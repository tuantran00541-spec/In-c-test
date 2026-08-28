#!/usr/bin/env python3
"""Check a generated prefix against released Kimi-VL greedy logits in one causal prefill.

The prompt, projected BF16 media, and candidate generated-token prefix are supplied explicitly.
The released decoder implementation runs layer-by-layer with a bounded working set, and logits
at the relevant causal positions report the first token where official greedy differs.
"""
from __future__ import annotations

import argparse
import gc
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from dump_official_streamed_text_prefill import (
    REPO, GLOBAL, MEDIA_PAD, EMBED, FINAL_NORM, LM_HEAD,
    Trunk, Experts, layer_state_dict, repair_meta_rotary, official_rms,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_dir", type=pathlib.Path)
    ap.add_argument("prompt_ids", type=pathlib.Path)
    ap.add_argument("media_u16", type=pathlib.Path)
    ap.add_argument("prefix_ids", type=pathlib.Path,
                    help="candidate generated token IDs to check, whitespace separated")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--logit-chunk", type=int, default=8192)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, int(__import__('os').environ.get('OMP_NUM_THREADS', '2'))))

    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True, revision=args.revision)
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    decoder_cls = get_class_from_dynamic_module(
        "modeling_kimi_vl.DeepseekV3DecoderLayer", REPO, revision=args.revision
    )

    prompt = [int(x) for x in args.prompt_ids.read_text().split()]
    prefix = [int(x) for x in args.prefix_ids.read_text().split()]
    if not prompt:
        raise SystemExit("empty prompt")
    if not prefix:
        raise SystemExit("empty generated prefix")
    media_pos = [i for i,t in enumerate(prompt) if t == MEDIA_PAD]
    if not media_pos:
        raise SystemExit("prompt has no media pads")

    prompt_len = len(prompt)
    # Prefix[:-1] is teacher-forced input. Hidden at prompt_len-1 predicts prefix[0], etc.
    seq_ids = prompt + prefix[:-1]
    S = len(seq_ids)
    H = int(tc.hidden_size)

    tr = Trunk(args.packed_dir)
    ex = Experts(args.packed_dir)
    embed = tr.get(GLOBAL, EMBED)
    x = embed[torch.tensor(seq_ids, dtype=torch.long)].clone().unsqueeze(0)
    del embed

    bits = np.fromfile(args.media_u16, dtype=np.uint16)
    if bits.size != len(media_pos) * H:
        raise SystemExit(
            f"media u16 size={bits.size}, expected {len(media_pos)*H} "
            f"for {len(media_pos)} media positions"
        )
    media = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(len(media_pos),H)
    x[0, torch.tensor(media_pos)] = media
    assert x.dtype == torch.bfloat16

    mask2 = torch.ones((1,S), dtype=torch.long)
    attention_mask = _prepare_4d_causal_attention_mask(mask2, (1,S), x, 0)
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0)

    print(
        f"teacher-forced official: prompt={prompt_len} seq={S} "
        f"predicts={len(prefix)} media={len(media_pos)} prefix={prefix}", flush=True
    )
    for L in range(int(tc.num_hidden_layers)):
        with torch.device("meta"):
            layer_mod = decoder_cls(tc, L)
        sd = layer_state_dict(tr, ex, tc, L)
        miss = layer_mod.load_state_dict(sd, strict=True, assign=True)
        if miss.missing_keys or miss.unexpected_keys:
            raise RuntimeError(miss)
        repair_meta_rotary(layer_mod.self_attn.rotary_emb, tc, S)
        layer_mod.eval()
        with torch.inference_mode():
            x = layer_mod(x, attention_mask=attention_mask,
                          position_ids=position_ids, use_cache=False)[0]
        if x.dtype != torch.bfloat16:
            raise RuntimeError(f"layer {L} returned {x.dtype}")
        print(f"layer {L:02d} done", flush=True)
        del layer_mod, sd
        gc.collect()

    z = official_rms(x, tr.get(GLOBAL, FINAL_NORM), float(tc.rms_norm_eps))
    targets = z[0, torch.arange(prompt_len-1, prompt_len-1+len(prefix))]
    lm = tr.get(GLOBAL, LM_HEAD)
    best_val = torch.full((len(prefix),), -float("inf"), dtype=torch.float32)
    best_id = torch.full((len(prefix),), -1, dtype=torch.long)
    prefix_logits = torch.full((len(prefix),), float("nan"), dtype=torch.float32)

    for a in range(0, int(tc.vocab_size), args.logit_chunk):
        b = min(int(tc.vocab_size), a + args.logit_chunk)
        logits = F.linear(targets, lm[a:b]).float()
        vals, inds = torch.max(logits, dim=1)
        take = vals > best_val
        best_val[take] = vals[take]
        best_id[take] = inds[take] + a
        for j, tid in enumerate(prefix):
            if a <= tid < b:
                prefix_logits[j] = logits[j, tid-a]

    official = best_id.tolist()
    matches = [int(a == b) for a,b in zip(official, prefix)]
    first_div = next((i for i,m in enumerate(matches) if not m), -1)
    print("CANDIDATE_IDS=" + " ".join(map(str,prefix)), flush=True)
    print("OFFICIAL_GREEDY_IDS=" + " ".join(map(str,official)), flush=True)
    print("MATCH_FLAGS=" + " ".join(map(str,matches)), flush=True)
    print("CANDIDATE_LOGITS=" + " ".join(f"{v:.7g}" for v in prefix_logits.tolist()), flush=True)
    print("OFFICIAL_MAX_LOGITS=" + " ".join(f"{v:.7g}" for v in best_val.tolist()), flush=True)
    print(f"FIRST_DIVERGENCE={first_div}", flush=True)
    if first_div >= 0:
        print(
            f"DIVERGENCE_DETAIL index={first_div} candidate={prefix[first_div]} "
            f"official={official[first_div]} candidate_logit={prefix_logits[first_div].item():.7g} "
            f"official_logit={best_val[first_div].item():.7g}", flush=True
        )


if __name__ == "__main__":
    main()
