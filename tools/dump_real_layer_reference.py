#!/usr/bin/env python3
"""Create a 4-token V3 oracle for one complete Kimi-VL decoder layer.

Numerical contract: BF16 checkpoint weights are expanded to FP32 for arithmetic, while
activations remain FP32. This matches the current C V3 correctness path. It exercises
RMSNorm, eager MLA + RoPE + causal softmax, residuals, and streamed MoE. Only experts
actually selected across the short sequence are materialized by the Python oracle.
"""
import argparse, json, pathlib, struct
import torch
import torch.nn.functional as F
from safetensors import safe_open

MAGIC=b"KVLV3OR1"
HDR=struct.Struct("<8s16I3f21Q")

def bf16_bytes(t):
    return t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()

def rms(x,w,eps):
    xf=x.float()
    return w.float()*xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+eps)

def rotate_half(x):
    h=x.shape[-1]//2
    return torch.cat((-x[...,h:],x[...,:h]),dim=-1)

def rope(raw,theta):
    d=raw.shape[-1]; half=d//2
    perm=raw.view(*raw.shape[:-1],half,2).transpose(-1,-2).reshape(*raw.shape[:-1],d)
    pos=torch.arange(raw.shape[0],dtype=torch.float32)
    inv=theta**(-torch.arange(0,d,2,dtype=torch.float32)/d)
    emb=torch.cat((torch.outer(pos,inv),torch.outer(pos,inv)),dim=-1)
    while emb.ndim<perm.ndim: emb=emb.unsqueeze(1)
    return perm*emb.cos()+rotate_half(perm)*emb.sin()

def mla(x,c,w):
    S=x.shape[0]; N=c['N']; DN=c['DN']; DR=c['DR']; DV=c['DV']; R=c['R']
    q=F.linear(x,w['q'].float()).view(S,N,DN+DR)
    qn,qp=q[...,:DN],q[...,DN:]
    comp=F.linear(x,w['kva'].float()); latent,kp=comp[:,:R],comp[:,R:]
    latent=rms(latent,w['kvan'],c['eps'])
    kv=F.linear(latent,w['kvb'].float()).view(S,N,DN+DV)
    kn,v=kv[...,:DN],kv[...,DN:]
    q=torch.cat((qn,rope(qp,c['theta'])),dim=-1)
    kr=rope(kp,c['theta']).unsqueeze(1).expand(-1,N,-1)
    k=torch.cat((kn,kr),dim=-1)
    scores=torch.einsum('thd,shd->hts',q,k)*(1.0/((DN+DR)**0.5))
    mask=torch.triu(torch.full((S,S),float('-inf')),diagonal=1)
    probs=torch.softmax((scores+mask.unsqueeze(0)).float(),dim=-1)
    heads=torch.einsum('hts,shd->thd',probs,v).reshape(S,N*DV)
    return F.linear(heads,w['o'].float())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('model_dir',type=pathlib.Path)
    ap.add_argument('out_fixture',type=pathlib.Path)
    ap.add_argument('--layer',type=int,default=1)
    ap.add_argument('--seq-len',type=int,default=4)
    ap.add_argument('--seed',type=int,default=270827)
    args=ap.parse_args()
    cfg=json.loads((args.model_dir/'config.json').read_text())['text_config']
    index=json.loads((args.model_dir/'model.safetensors.index.json').read_text())['weight_map']
    L=args.layer; p=f'language_model.model.layers.{L}'
    def get(name):
        shard=index.get(name)
        if shard is None: raise KeyError(name)
        with safe_open(args.model_dir/shard,framework='pt',device='cpu') as f:
            return f.get_tensor(name)

    c=dict(S=args.seq_len,H=int(cfg['hidden_size']),N=int(cfg['num_attention_heads']),
           DN=int(cfg['qk_nope_head_dim']),DR=int(cfg['qk_rope_head_dim']),DV=int(cfg['v_head_dim']),
           R=int(cfg['kv_lora_rank']),I=int(cfg['moe_intermediate_size']),
           SI=int(cfg['moe_intermediate_size'])*int(cfg['n_shared_experts']),
           E=int(cfg['n_routed_experts']),K=int(cfg['num_experts_per_tok']),L=L,
           eps=float(cfg['rms_norm_eps']),theta=float(cfg['rope_theta']),
           scale=float(cfg['routed_scaling_factor']),norm=bool(cfg['norm_topk_prob']))
    if cfg['q_lora_rank'] is not None: raise SystemExit('V3 currently expects q_lora_rank=null')
    if int(cfg['n_group'])!=1 or int(cfg['topk_group'])!=1: raise SystemExit('V3 helper currently expects one expert group')
    w={
      'inorm':get(p+'.input_layernorm.weight'),
      'q':get(p+'.self_attn.q_proj.weight'),
      'kva':get(p+'.self_attn.kv_a_proj_with_mqa.weight'),
      'kvan':get(p+'.self_attn.kv_a_layernorm.weight'),
      'kvb':get(p+'.self_attn.kv_b_proj.weight'),
      'o':get(p+'.self_attn.o_proj.weight'),
      'pnorm':get(p+'.post_attention_layernorm.weight'),
      'router':get(p+'.mlp.gate.weight').float(),
      'bias':get(p+'.mlp.gate.e_score_correction_bias').float(),
      'sg':get(p+'.mlp.shared_experts.gate_proj.weight'),
      'su':get(p+'.mlp.shared_experts.up_proj.weight'),
      'sd':get(p+'.mlp.shared_experts.down_proj.weight')}
    torch.manual_seed(args.seed)
    x=(torch.randn(c['S'],c['H'])*0.20).to(torch.bfloat16).float()
    n1=rms(x,w['inorm'],c['eps']); a=mla(n1,c,w); r1=x+a; n2=rms(r1,w['pnorm'],c['eps'])
    scores=torch.sigmoid(F.linear(n2,w['router']))
    choice=scores+w['bias']; _,ids=torch.topk(choice,k=c['K'],dim=-1,sorted=False)
    ww=scores.gather(1,ids)
    if c['norm']: ww=ww/(ww.sum(-1,keepdim=True)+1e-20)
    ww=ww*c['scale']
    unique=sorted(set(ids.flatten().tolist()))
    experts={}
    for eid in unique:
        ep=f'{p}.mlp.experts.{eid}'
        experts[eid]=(get(ep+'.gate_proj.weight'),get(ep+'.up_proj.weight'),get(ep+'.down_proj.weight'))
    mout=[]
    for t in range(c['S']):
        y=torch.zeros(c['H'],dtype=torch.float32)
        for eid,alpha in zip(ids[t].tolist(),ww[t].tolist()):
            g,u,d=experts[eid]
            y += F.linear(F.silu(F.linear(n2[t],g.float()))*F.linear(n2[t],u.float()),d.float())*alpha
        y += F.linear(F.silu(F.linear(n2[t],w['sg'].float()))*F.linear(n2[t],w['su'].float()),w['sd'].float())
        mout.append(y)
    final=r1+torch.stack(mout)

    blob=bytearray(HDR.size)
    def add(data): off=len(blob); blob.extend(data); return off
    resident=[add(x.numpy().tobytes())]
    for key in ('inorm','q','kva','kvan','kvb','o','pnorm'): resident.append(add(bf16_bytes(w[key])))
    resident += [add(w['router'].numpy().tobytes()),add(w['bias'].numpy().tobytes())]
    for key in ('sg','su','sd'): resident.append(add(bf16_bytes(w[key])))
    exp4=[add(z.float().contiguous().numpy().tobytes()) for z in (n1,a,r1,n2)]
    oid=add(ids.to(torch.int32).contiguous().numpy().tobytes())
    oww=add(ww.float().contiguous().numpy().tobytes())
    ofinal=add(final.float().contiguous().numpy().tobytes())
    header=HDR.pack(MAGIC,1,c['S'],c['H'],c['N'],c['DN'],c['DR'],c['DV'],c['R'],c['I'],c['SI'],c['E'],c['K'],1,1,c['L'],1,
                    c['eps'],c['theta'],c['scale'],*(resident+exp4+[oid,oww,ofinal]),len(blob))
    blob[:HDR.size]=header
    args.out_fixture.parent.mkdir(parents=True,exist_ok=True); args.out_fixture.write_bytes(blob)
    print('layer',L,'seq_len',c['S'],'unique_selected_experts',len(unique),unique)
    print('top_ids',ids.tolist())
    print('fixture',args.out_fixture,'bytes',len(blob))
    print('Only experts selected by the 4-token oracle were materialized in Python.')

if __name__=='__main__': main()
