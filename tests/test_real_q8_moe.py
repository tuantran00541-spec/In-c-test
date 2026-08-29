#!/usr/bin/env python3
"""Released Kimi-VL layer-1 router + top-6 routed experts + shared expert Q8 oracle."""
from __future__ import annotations
import argparse, json, pathlib, struct, subprocess, tempfile
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MAGIC=b'KVLV2OR1'
HDR=struct.Struct('<8s10If10Q')


def bf16_bytes(t: torch.Tensor) -> bytes:
    return t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()

def add(blob: bytearray,data: bytes): off=len(blob); blob+=data; return off

def load_many(model_dir: pathlib.Path,names: list[str]):
    wm=json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    grouped={}
    for n in names: grouped.setdefault(wm[n],[]).append(n)
    out={}
    for shard,ns in grouped.items():
        with safe_open(model_dir/shard,framework='pt',device='cpu') as f:
            for n in ns: out[n]=f.get_tensor(n)
    return out

def router_topk(x,rw,bias,k,n_group,topk_group,norm,scale):
    scores=torch.sigmoid(F.linear(x.float(),rw.float()))
    choice=scores+bias.float()
    E=choice.numel(); per=E//n_group
    if n_group>1:
        g=choice.view(n_group,per)
        gs=torch.topk(g,k=min(2,per),dim=-1,sorted=False).values.sum(dim=-1)
        keep=torch.topk(gs,k=topk_group,sorted=False).indices
        mask=torch.zeros(n_group,dtype=torch.bool); mask[keep]=True
        choice=torch.where(mask[:,None],g,torch.zeros_like(g)).reshape(-1)
    ids=torch.topk(choice,k=k,sorted=False).indices
    weights=scores[ids]
    if k>1 and norm: weights=weights/(weights.sum()+1e-20)
    weights=weights*scale
    return ids,weights

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('model_dir',type=pathlib.Path); ap.add_argument('q8_dir',type=pathlib.Path); ap.add_argument('--build-dir',type=pathlib.Path,default=pathlib.Path('build')); ap.add_argument('--layer',type=int,default=1); args=ap.parse_args()
    exe=args.build_dir/('kvl_moe_probe.exe' if __import__('os').name=='nt' else 'kvl_moe_probe')
    if not exe.exists():
        p=args.build_dir/'Release'/'kvl_moe_probe.exe'
        if p.exists(): exe=p
    cfg=json.loads((args.model_dir/'config.json').read_text())['text_config']
    H=int(cfg['hidden_size']); I=int(cfg['moe_intermediate_size']); E=int(cfg['n_routed_experts']); K=int(cfg['num_experts_per_tok'])
    n_group=int(cfg.get('n_group',1)); topk_group=int(cfg.get('topk_group',n_group)); norm=int(bool(cfg.get('norm_topk_prob',True))); scale=float(cfg.get('routed_scaling_factor',1.0))
    L=args.layer; p=f'language_model.model.layers.{L}.mlp'
    rn=p+'.gate.weight'; bn=p+'.gate.e_score_correction_bias'; sgn=p+'.shared_experts.gate_proj.weight'; sun=p+'.shared_experts.up_proj.weight'; sdn=p+'.shared_experts.down_proj.weight'
    base=load_many(args.model_dir,[rn,bn,sgn,sun,sdn]); rw=base[rn].float(); bias=base[bn].float(); sg=base[sgn]; su=base[sun]; sd=base[sdn]; SI=int(sg.shape[0])
    assert rw.shape==(E,H) and sg.shape==(SI,H) and sd.shape==(H,SI)
    rng=np.random.default_rng(260829)
    xs=[rng.normal(0,s,size=H).astype(np.float32) for s in (0.08,0.25,0.7)]
    worst=0.0
    with tempfile.TemporaryDirectory(prefix='kvl-real-q8-moe-') as td:
        td=pathlib.Path(td)
        for ci,xnp in enumerate(xs):
            x=torch.from_numpy(xnp); ids,w=router_topk(x,rw,bias,K,n_group,topk_group,norm,scale)
            expert_names=[]
            for eid in ids.tolist():
                e=f'{p}.experts.{eid}'
                expert_names += [e+'.gate_proj.weight',e+'.up_proj.weight',e+'.down_proj.weight']
            ew=load_many(args.model_dir,expert_names)
            routed=torch.zeros(H,dtype=torch.float32)
            for eid,weight in zip(ids.tolist(),w.tolist()):
                e=f'{p}.experts.{eid}'; g=ew[e+'.gate_proj.weight'].float(); u=ew[e+'.up_proj.weight'].float(); d=ew[e+'.down_proj.weight'].float()
                routed += F.linear(F.silu(F.linear(x,g))*F.linear(x,u),d)*weight
            shared=F.linear(F.silu(F.linear(x,sg.float()))*F.linear(x,su.float()),sd.float())
            expected=(routed+shared).float().contiguous()

            blob=bytearray(HDR.size)
            off_x=add(blob,xnp.astype('<f4').tobytes()); off_rw=add(blob,rw.numpy().astype('<f4').tobytes()); off_b=add(blob,bias.numpy().astype('<f4').tobytes())
            off_sg=add(blob,bf16_bytes(sg)); off_su=add(blob,bf16_bytes(su)); off_sd=add(blob,bf16_bytes(sd)); off_ids=add(blob,ids.to(torch.int32).numpy().tobytes()); off_w=add(blob,w.float().numpy().tobytes()); off_out=add(blob,expected.numpy().astype('<f4').tobytes())
            blob[:HDR.size]=HDR.pack(MAGIC,1,H,I,SI,E,K,n_group,topk_group,L,norm,scale,off_x,off_rw,off_b,off_sg,off_su,off_sd,off_ids,off_w,off_out,len(blob))
            fp=td/f'case{ci}.bin'; fp.write_bytes(blob)
            budget=K*8671232
            run=subprocess.run([str(exe),str(args.q8_dir/'experts.bin'),str(args.q8_dir/'experts.idx'),str(fp),str(budget)],text=True,capture_output=True)
            print(run.stdout,end=''); print(run.stderr,end='')
            if run.returncode: raise SystemExit(f'Q8 MoE probe failed case {ci}: {run.returncode}')
            import re
            m=re.search(r'moe_rel_rms=([0-9.eE+-]+)',run.stdout); assert m,run.stdout
            rel=float(m.group(1)); worst=max(worst,rel); print(f'REAL_MOE_CASE={ci} REL_RMS={rel:.9g} IDS={ids.tolist()}')
    print(f'REAL_MOE_WORST_REL_RMS={worst:.9g}')
    assert worst<0.05
    print('PASS: released Kimi top-6 routed+shared MoE survives Q8 lab gate')

if __name__=='__main__': main()
