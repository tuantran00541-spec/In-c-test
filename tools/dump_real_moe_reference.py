#!/usr/bin/env python3
"""Create a V2 one-token MoE oracle from *real* local Kimi-VL weights without loading the model.

Memory behavior: read router+bias, choose top-k, then materialize only those routed experts
plus the resident shared expert for one layer. It never instantiates Transformers or the
32.8 GB model graph.

The oracle intentionally evaluates BF16 weights as float32 matrices. That matches V2's
current numerical contract (FP32 activations + BF16 storage). Native BF16 activation
rounding is a later decoder-level correctness milestone.
"""
import argparse, json, pathlib, struct
import torch
import torch.nn.functional as F
from safetensors import safe_open

MAGIC=b"KVLV2OR1"
HDR=struct.Struct("<8s10If10Q")

def bf16_bytes(t):
    return t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('model_dir',type=pathlib.Path)
    ap.add_argument('out_fixture',type=pathlib.Path)
    ap.add_argument('--layer',type=int,default=1)
    ap.add_argument('--seed',type=int,default=260827)
    args=ap.parse_args()
    cfg=json.loads((args.model_dir/'config.json').read_text())['text_config']
    index=json.loads((args.model_dir/'model.safetensors.index.json').read_text())['weight_map']
    L=args.layer
    prefix=f'language_model.model.layers.{L}.mlp'

    def get(name):
        shard=index.get(name)
        if shard is None: raise KeyError(name)
        with safe_open(args.model_dir/shard,framework='pt',device='cpu') as f:
            return f.get_tensor(name)

    rw=get(prefix+'.gate.weight').float()
    bias=get(prefix+'.gate.e_score_correction_bias').float()
    H=int(cfg['hidden_size']); E=int(cfg['n_routed_experts']); K=int(cfg['num_experts_per_tok'])
    I=int(cfg['moe_intermediate_size']); SI=I*int(cfg['n_shared_experts'])
    torch.manual_seed(args.seed)
    # BF16-representable hidden values eliminate input-storage ambiguity while V2 computes FP32.
    x=(torch.randn(H)*0.20).to(torch.bfloat16).float()
    scores=torch.sigmoid(F.linear(x.float(),rw.float()))
    choice=scores+bias
    # Kimi-VL config currently n_group=topk_group=1; fail loudly if that changes.
    ng=int(cfg['n_group']); tg=int(cfg['topk_group'])
    if ng != 1 or tg != 1:
        raise SystemExit('real-oracle helper currently expects Kimi-VL n_group=topk_group=1')
    _,ids=torch.topk(choice,k=K,sorted=False)
    weights=scores.gather(0,ids)
    if cfg['norm_topk_prob']:
        weights=weights/(weights.sum()+1e-20)
    weights=weights*float(cfg['routed_scaling_factor'])

    routed=torch.zeros(H,dtype=torch.float32)
    for eid,w in zip(ids.tolist(),weights.tolist()):
        ep=f'{prefix}.experts.{eid}'
        gate=get(ep+'.gate_proj.weight').float(); up=get(ep+'.up_proj.weight').float(); down=get(ep+'.down_proj.weight').float()
        routed += F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)*w
    sp=prefix+'.shared_experts'
    sg=get(sp+'.gate_proj.weight'); su=get(sp+'.up_proj.weight'); sd=get(sp+'.down_proj.weight')
    shared=F.linear(F.silu(F.linear(x,sg.float()))*F.linear(x,su.float()),sd.float())
    expected=(routed+shared).float().contiguous()

    blob=bytearray(HDR.size)
    def add(data): off=len(blob); blob.extend(data); return off
    ox=add(x.numpy().tobytes()); orw=add(rw.numpy().tobytes()); ob=add(bias.numpy().tobytes())
    osg=add(bf16_bytes(sg)); osu=add(bf16_bytes(su)); osd=add(bf16_bytes(sd))
    oi=add(ids.to(torch.int32).numpy().tobytes()); ow=add(weights.float().numpy().tobytes()); oo=add(expected.numpy().tobytes())
    blob[:HDR.size]=HDR.pack(MAGIC,1,H,I,SI,E,K,ng,tg,L,int(cfg['norm_topk_prob']),float(cfg['routed_scaling_factor']),
                             ox,orw,ob,osg,osu,osd,oi,ow,oo,len(blob))
    args.out_fixture.parent.mkdir(parents=True,exist_ok=True); args.out_fixture.write_bytes(blob)
    print('layer',L,'top_ids',ids.tolist())
    print('weights',[round(v,8) for v in weights.tolist()])
    print('fixture',args.out_fixture,'bytes',len(blob))
    print('Only selected routed experts were loaded into Python memory.')

if __name__=='__main__': main()
