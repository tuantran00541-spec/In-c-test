#!/usr/bin/env python3
"""Exact architecture arithmetic for Kimi-VL-A3B-style configs.
Does not load model weights and uses only the Python standard library.
"""
import argparse, json, pathlib

GIB = 1024**3

def fmt(n): return f"{n/1e9:.6f} B"
def gib(n, bpp): return n*bpp/GIB

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("model_dir", type=pathlib.Path)
    args=ap.parse_args()
    cfg=json.loads((args.model_dir/"config.json").read_text(encoding="utf-8"))
    t=cfg["text_config"]; v=cfg["vision_config"]
    H=t["hidden_size"]; V=t["vocab_size"]; L=t["num_hidden_layers"]
    nh=t["num_attention_heads"]; qn=t["qk_nope_head_dim"]; qr=t["qk_rope_head_dim"]
    vr=t["v_head_dim"]; kr=t["kv_lora_rank"]; E=t["n_routed_experts"]
    K=t["num_experts_per_tok"]; I=t["moe_intermediate_size"]; D=t["intermediate_size"]
    S=t["n_shared_experts"]; first=t["first_k_dense_replace"]

    embed=V*H; lm=V*H; final_norm=H
    attn=H*(nh*(qn+qr)) + H*(kr+qr) + kr + kr*(nh*(qn+vr)) + (nh*vr)*H
    layer_norms=2*H
    dense_mlp=3*H*D
    router=H*E + E
    shared_mlp=3*H*(I*S)
    expert=3*H*I
    moe_layers=sum(1 for i in range(L) if i>=first and i%t["moe_layer_freq"]==0)
    routed=moe_layers*E*expert
    trunk=embed+lm+final_norm + L*(attn+layer_norms) + first*dense_mlp + moe_layers*(router+shared_mlp)

    vh=v["hidden_size"]; vi=v["intermediate_size"]; vl=v["num_hidden_layers"]
    p=v["patch_size"]; ph,pw=v["init_pos_emb_height"],v["init_pos_emb_width"]
    patch=vh*3*p*p + vh + ph*pw*vh
    vis_layer=(4*vh) + (vh*vi+vi + vi*vh+vh) + (vh*(3*vh)+3*vh + vh*vh+vh)
    vision=patch + vl*vis_layer + 2*vh
    mk=v["merge_kernel_size"]; proj_h=vh*mk[0]*mk[1]
    projector=2*vh + proj_h*proj_h+proj_h + proj_h*H+H

    groups=[
      ("token embedding",embed),("lm_head",lm),("attention all layers",L*attn),
      ("decoder norms",L*layer_norms+final_norm),("dense layer-0 MLP",first*dense_mlp),
      ("routers",moe_layers*router),("shared experts",moe_layers*shared_mlp),
      ("ROUTED EXPERTS",routed),("vision tower",vision),("vision projector",projector)]
    print(f"MoE layers={moe_layers}, experts/layer={E}, top-k={K}")
    print(f"one routed expert = {expert:,} params")
    print("\n%-24s %14s %11s" % ("group","params","BF16 GiB"))
    print("-"*54)
    for name,n in groups: print("%-24s %14s %11.3f"%(name,fmt(n),gib(n,2)))
    total=sum(n for _,n in groups)
    print("-"*54)
    print("%-24s %14s %11.3f"%("TOTAL",fmt(total),gib(total,2)))
    mxfp4=17/32
    active_expert=moe_layers*K*expert
    print(f"\nProposed expert MXFP4 store: {gib(routed,mxfp4):.3f} GiB")
    print(f"One MXFP4 expert: {expert*mxfp4/1024**2:.3f} MiB")
    print(f"100% miss expert I/O/token: {gib(active_expert,mxfp4):.3f} GiB")
    print(f"Resident non-routed text trunk BF16: {gib(trunk,2):.3f} GiB")
    print(f"Vision+projector BF16 temporary: {gib(vision+projector,2):.3f} GiB")

    idx=args.model_dir/"model.safetensors.index.json"
    if idx.exists():
        j=json.loads(idx.read_text(encoding="utf-8"))
        total_size=j.get("metadata",{}).get("total_size")
        if total_size:
            print(f"Safetensors index total: {total_size/1e9:.3f} GB ({total_size/GIB:.3f} GiB)")
            print(f"Arithmetic delta: {(total*2-total_size)/1e6:+.3f} MB")

if __name__=="__main__": main()
