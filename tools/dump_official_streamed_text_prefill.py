#!/usr/bin/env python3
"""Official-code, low-RAM Kimi-VL text prefill oracle over packed runtime stores.

This intentionally instantiates exactly one released DeepseekV3DecoderLayer at a time on
meta, assigns the packed BF16 tensors without copying the whole checkpoint resident, runs
that layer with the checkpoint's own eager forward implementation, then discards it.
It accepts already-projected BF16 media embeddings so vision is causally out of the test.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import struct

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

REPO = "moonshotai/Kimi-VL-A3B-Instruct"
GLOBAL = 0xFFFFFFFF
MEDIA_PAD = 163605

TMAGIC = b"KVLTRNK1"
THDR = struct.Struct("<8s4I2Q")
TREC = struct.Struct("<8I3Q")
EMAGIC = b"KVLXPRT1"
EHDR = struct.Struct("<8sIIIIIIQQ")
EREC = struct.Struct("<IIQQQQQQQQQ")

EMBED=1; FINAL_NORM=2; LM_HEAD=3
INORM=10; PNORM=11; Q=12; KVA=13; KVAN=14; KVB=15; O=16
DG=20; DU=21; DD=22; ROUTER=30; RBIAS=31; SG=32; SU=33; SD=34


def bf16_map(path: pathlib.Path, off: int, shape: tuple[int, ...]) -> torch.Tensor:
    n = int(np.prod(shape))
    a = np.memmap(path, dtype=np.uint16, mode="c", offset=int(off), shape=(n,))
    return torch.from_numpy(a).view(torch.bfloat16).reshape(shape)


class Trunk:
    def __init__(self, root: pathlib.Path):
        self.path = root / "trunk.bin"
        raw = (root / "trunk.idx").read_bytes()
        h = THDR.unpack_from(raw, 0)
        if h[0] != TMAGIC or h[1] != 1:
            raise RuntimeError("bad trunk index")
        self.r = {}
        off = h[5]
        for _ in range(h[3]):
            z = TREC.unpack_from(raw, off); off += TREC.size
            self.r[(z[0], z[1])] = z

    def get(self, layer: int, kind: int) -> torch.Tensor:
        z = self.r[(layer, kind)]
        nd = z[3]
        shape = tuple(int(z[4+i]) for i in range(nd))
        return bf16_map(self.path, z[8], shape)


class Experts:
    def __init__(self, root: pathlib.Path):
        self.path = root / "experts.bin"
        raw = (root / "experts.idx").read_bytes()
        h = EHDR.unpack_from(raw, 0)
        if h[0] != EMAGIC or h[1] != 1:
            raise RuntimeError("bad expert index")
        self.r = {}
        off = h[7]
        for _ in range(h[5]):
            z = EREC.unpack_from(raw, off); off += EREC.size
            self.r[(z[0], z[1])] = z

    def get(self, layer: int, expert: int, hidden: int, inter: int):
        z = self.r[(layer, expert)]
        base = z[2]
        return (
            bf16_map(self.path, base + z[5], (inter, hidden)),
            bf16_map(self.path, base + z[7], (inter, hidden)),
            bf16_map(self.path, base + z[9], (hidden, inter)),
        )


def layer_state_dict(tr: Trunk, ex: Experts, tc, layer: int) -> dict[str, torch.Tensor]:
    sd = {
        "input_layernorm.weight": tr.get(layer, INORM),
        "post_attention_layernorm.weight": tr.get(layer, PNORM),
        "self_attn.q_proj.weight": tr.get(layer, Q),
        "self_attn.kv_a_proj_with_mqa.weight": tr.get(layer, KVA),
        "self_attn.kv_a_layernorm.weight": tr.get(layer, KVAN),
        "self_attn.kv_b_proj.weight": tr.get(layer, KVB),
        "self_attn.o_proj.weight": tr.get(layer, O),
    }
    is_moe = layer >= int(tc.first_k_dense_replace) and layer % int(tc.moe_layer_freq) == 0
    if not is_moe:
        sd.update({
            "mlp.gate_proj.weight": tr.get(layer, DG),
            "mlp.up_proj.weight": tr.get(layer, DU),
            "mlp.down_proj.weight": tr.get(layer, DD),
        })
        return sd

    sd["mlp.gate.weight"] = tr.get(layer, ROUTER)
    sd["mlp.gate.e_score_correction_bias"] = tr.get(layer, RBIAS)
    sd["mlp.shared_experts.gate_proj.weight"] = tr.get(layer, SG)
    sd["mlp.shared_experts.up_proj.weight"] = tr.get(layer, SU)
    sd["mlp.shared_experts.down_proj.weight"] = tr.get(layer, SD)
    H = int(tc.hidden_size); I = int(tc.moe_intermediate_size)
    for eid in range(int(tc.n_routed_experts)):
        g,u,d = ex.get(layer, eid, H, I)
        p = f"mlp.experts.{eid}."
        sd[p+"gate_proj.weight"] = g
        sd[p+"up_proj.weight"] = u
        sd[p+"down_proj.weight"] = d
    return sd


def official_rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata_dir", type=pathlib.Path)
    ap.add_argument("packed_dir", type=pathlib.Path)
    ap.add_argument("prompt_ids", type=pathlib.Path)
    ap.add_argument("media_u16", type=pathlib.Path)
    ap.add_argument("out_npz", type=pathlib.Path)
    ap.add_argument("--logit-chunk", type=int, default=8192)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, int(__import__('os').environ.get('OMP_NUM_THREADS', '2'))))

    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    decoder_cls = get_class_from_dynamic_module("modeling_kimi_vl.DeepseekV3DecoderLayer", REPO)

    tr = Trunk(args.packed_dir); ex = Experts(args.packed_dir)
    ids = [int(x) for x in args.prompt_ids.read_text().split()]
    S = len(ids); H = int(tc.hidden_size)
    if S != 136:
        raise SystemExit(f"expected exact VI 136-token prompt, got {S}")
    positions = [i for i,t in enumerate(ids) if t == MEDIA_PAD]
    if positions != list(range(15,64)):
        raise SystemExit(f"unexpected media positions: {positions}")

    embed = tr.get(GLOBAL, EMBED)
    x = embed[torch.tensor(ids, dtype=torch.long)].clone().unsqueeze(0)
    del embed
    bits = np.fromfile(args.media_u16, dtype=np.uint16)
    if bits.size != 49 * H:
        raise SystemExit(f"media u16 size={bits.size}, expected {49*H}")
    media = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(49,H)
    x[0, torch.tensor(positions)] = media
    assert x.dtype == torch.bfloat16

    attn_mask_2d = torch.ones((1,S), dtype=torch.long)
    attention_mask = _prepare_4d_causal_attention_mask(attn_mask_2d, (1,S), x, 0)
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0)
    last_by_layer = []

    print(f"official streamed prefill: seq={S} hidden={H} media={len(positions)} dtype={x.dtype}", flush=True)
    for L in range(int(tc.num_hidden_layers)):
        with torch.device("meta"):
            layer_mod = decoder_cls(tc, L)
        sd = layer_state_dict(tr, ex, tc, L)
        miss = layer_mod.load_state_dict(sd, strict=True, assign=True)
        if miss.missing_keys or miss.unexpected_keys:
            raise RuntimeError(miss)
        # Rotary caches created on meta by the meta constructor must be rebuilt on CPU.
        layer_mod.self_attn.rotary_emb._set_cos_sin_cache(S, torch.device("cpu"), torch.bfloat16)
        layer_mod.eval()
        with torch.inference_mode():
            x = layer_mod(x, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)[0]
        if x.dtype != torch.bfloat16:
            raise RuntimeError(f"layer {L} returned {x.dtype}")
        last = x[0,-1].float().cpu().clone()
        last_by_layer.append(last)
        print(f"layer {L:02d}: last_rms={last.square().mean().sqrt().item():.8g} last_max={last.abs().max().item():.8g}", flush=True)
        del layer_mod, sd
        gc.collect()

    z = official_rms(x, tr.get(GLOBAL, FINAL_NORM), float(tc.rms_norm_eps))
    last_z = z[0,-1]
    lm = tr.get(GLOBAL, LM_HEAD)
    best_id = -1; best_val = -float("inf")
    top_vals = []
    for a in range(0, int(tc.vocab_size), args.logit_chunk):
        b = min(int(tc.vocab_size), a + args.logit_chunk)
        logits = F.linear(last_z, lm[a:b]).float()
        v, i = torch.max(logits, dim=0)
        vv=float(v); ii=a+int(i)
        if vv > best_val:
            best_val=vv; best_id=ii
        # retain a small global top list for debugging
        kk=min(8, logits.numel()); vals, inds=torch.topk(logits, kk)
        top_vals.extend((float(vv2), a+int(ii2)) for vv2,ii2 in zip(vals,inds))
    top_vals=sorted(top_vals, reverse=True)[:16]
    print(f"OFFICIAL_FIRST_TOKEN={best_id} logit={best_val:.9g}", flush=True)
    print("OFFICIAL_TOP16=" + " ".join(f"{i}:{v:.7g}" for v,i in top_vals), flush=True)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        layer_last=torch.stack(last_by_layer).numpy().astype(np.float32),
        final_last=last_z.float().cpu().numpy().astype(np.float32),
        first_token=np.array([best_id], dtype=np.int32),
        first_logit=np.array([best_val], dtype=np.float32),
    )


if __name__ == "__main__":
    main()
