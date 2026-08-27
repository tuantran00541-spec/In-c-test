#!/usr/bin/env python3
import argparse, json, pathlib, struct, subprocess, tempfile
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

MAGIC=b"KVLV3OR1"
HDR=struct.Struct("<8s16I3f21Q")

def bf16_bytes(t):
    return t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()

def rms(x,w,eps):
    xf=x.float(); return w.float()*xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+eps)

def rotate_half(x):
    h=x.shape[-1]//2
    return torch.cat((-x[...,h:],x[...,:h]),dim=-1)

def rope(raw, theta):
    d=raw.shape[-1]; half=d//2
    perm=raw.view(*raw.shape[:-1],half,2).transpose(-1,-2).reshape(*raw.shape[:-1],d)
    pos=torch.arange(raw.shape[0],dtype=torch.float32)
    inv=theta**(-torch.arange(0,d,2,dtype=torch.float32)/d)
    freq=torch.outer(pos,inv); emb=torch.cat((freq,freq),dim=-1)
    while emb.ndim<perm.ndim: emb=emb.unsqueeze(1)
    return perm*emb.cos()+rotate_half(perm)*emb.sin()

def mla(x,c,w):
    S,H=x.shape; N=c['N']; DN=c['DN']; DR=c['DR']; DV=c['DV']; R=c['R']
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
    scores=scores+mask.unsqueeze(0)
    probs=torch.softmax(scores.float(),dim=-1)
    ho=torch.einsum('hts,shd->thd',probs,v).reshape(S,N*DV)
    return F.linear(ho,w['o'].float())

def moe(x,c,w,experts):
    scores=torch.sigmoid(F.linear(x,w['router'].float()))
    choice=scores+w['bias'].float()
    _,ids=torch.topk(choice,k=c['K'],dim=-1,sorted=False)
    ww=scores.gather(1,ids)
    if c['norm']:
        ww=ww/(ww.sum(-1,keepdim=True)+1e-20)
    ww=ww*c['scale']
    outs=[]
    for t in range(x.shape[0]):
        y=torch.zeros(c['H'],dtype=torch.float32)
        for eid,alpha in zip(ids[t].tolist(),ww[t].tolist()):
            e=experts[eid]
            y += F.linear(F.silu(F.linear(x[t],e['g'].float()))*F.linear(x[t],e['u'].float()),e['d'].float())*alpha
        y += F.linear(F.silu(F.linear(x[t],w['sg'].float()))*F.linear(x[t],w['su'].float()),w['sd'].float())
        outs.append(y)
    return torch.stack(outs),ids,ww

def make_fixture(path,c,w,x,expected,ids,ww):
    blob=bytearray(HDR.size)
    def add(data): off=len(blob); blob.extend(data); return off
    offs=[]
    offs.append(add(x.float().contiguous().numpy().tobytes()))
    for key in ('inorm','q','kva','kvan','kvb','o','pnorm'):
        offs.append(add(bf16_bytes(w[key])))
    offs += [add(w['router'].float().contiguous().numpy().tobytes()),add(w['bias'].float().contiguous().numpy().tobytes())]
    for key in ('sg','su','sd'): offs.append(add(bf16_bytes(w[key])))
    for t in expected: offs.append(add(t.float().contiguous().numpy().tobytes()))
    offs += [add(ids.to(torch.int32).contiguous().numpy().tobytes()),add(ww.float().contiguous().numpy().tobytes())]
    resident=offs[:13]; exp=offs[13:18]; oid,ow=offs[18],offs[19]
    ordered=resident+exp[:4]+[oid,ow,exp[4]]
    header=HDR.pack(MAGIC,1,c['S'],c['H'],c['N'],c['DN'],c['DR'],c['DV'],c['R'],c['I'],c['SI'],c['E'],c['K'],1,1,c['L'],1,
                    c['eps'],c['theta'],c['scale'],*ordered,len(blob))
    blob[:HDR.size]=header; path.write_bytes(blob)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--build-dir',type=pathlib.Path,required=True); args=ap.parse_args()
    torch.manual_seed(270827)
    c=dict(S=4,H=16,N=2,DN=4,DR=4,DV=4,R=6,I=8,SI=16,E=64,K=6,L=1,eps=1e-5,theta=800000.0,scale=2.446,norm=True)
    def b(shape,scale=.08): return (torch.randn(*shape)*scale).to(torch.bfloat16)
    w={
      'inorm':b((c['H'],),.2)+torch.tensor(1.0,dtype=torch.bfloat16),
      'q':b((c['N']*(c['DN']+c['DR']),c['H'])),
      'kva':b((c['R']+c['DR'],c['H'])), 'kvan':b((c['R'],),.2)+torch.tensor(1.0,dtype=torch.bfloat16),
      'kvb':b((c['N']*(c['DN']+c['DV']),c['R'])), 'o':b((c['H'],c['N']*c['DV'])),
      'pnorm':b((c['H'],),.2)+torch.tensor(1.0,dtype=torch.bfloat16),
      'router':b((c['E'],c['H'])).float(), 'bias':(torch.randn(c['E'])*.02).float(),
      'sg':b((c['SI'],c['H'])), 'su':b((c['SI'],c['H'])), 'sd':b((c['H'],c['SI']))}
    experts={e:{'g':b((c['I'],c['H'])),'u':b((c['I'],c['H'])),'d':b((c['H'],c['I']))} for e in range(c['E'])}
    x=(torch.randn(c['S'],c['H'])*.2).to(torch.bfloat16).float()
    n1=rms(x,w['inorm'],c['eps']); a=mla(n1,c,w); r1=x+a; n2=rms(r1,w['pnorm'],c['eps']); m,ids,ww=moe(n2,c,w,experts); final=r1+m
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); model=td/'model'; model.mkdir(); packed=td/'packed'; fixture=td/'v3.fixture'
        names={}
        p=f"language_model.model.layers.{c['L']}"
        names[p+'.input_layernorm.weight']=w['inorm']; names[p+'.post_attention_layernorm.weight']=w['pnorm']
        names[p+'.self_attn.q_proj.weight']=w['q']; names[p+'.self_attn.kv_a_proj_with_mqa.weight']=w['kva']; names[p+'.self_attn.kv_a_layernorm.weight']=w['kvan']; names[p+'.self_attn.kv_b_proj.weight']=w['kvb']; names[p+'.self_attn.o_proj.weight']=w['o']
        mp=p+'.mlp'; names[mp+'.gate.weight']=w['router'].to(torch.bfloat16); names[mp+'.gate.e_score_correction_bias']=w['bias'].to(torch.bfloat16)
        names[mp+'.shared_experts.gate_proj.weight']=w['sg']; names[mp+'.shared_experts.up_proj.weight']=w['su']; names[mp+'.shared_experts.down_proj.weight']=w['sd']
        for e,z in experts.items():
            ep=f'{mp}.experts.{e}'; names[ep+'.gate_proj.weight']=z['g']; names[ep+'.up_proj.weight']=z['u']; names[ep+'.down_proj.weight']=z['d']
        shard='model-00001-of-00001.safetensors'; save_file(names,str(model/shard))
        weight_map={k:shard for k in names}; (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':weight_map}))
        (model/'config.json').write_text(json.dumps({'text_config':{'num_hidden_layers':27,'n_routed_experts':64}}))
        subprocess.run(['python',str(pathlib.Path(__file__).parents[1]/'tools/pack_experts.py'),str(model),str(packed),'--layer',str(c['L'])],check=True)
        w['router']=names[mp+'.gate.weight'].float(); w['bias']=names[mp+'.gate.e_score_correction_bias'].float()
        m,ids,ww=moe(n2,c,w,experts); final=r1+m
        make_fixture(fixture,c,w,x,[n1,a,r1,n2,final],ids,ww)
        exe=args.build_dir/('kvl_layer_probe.exe' if (args.build_dir/'kvl_layer_probe.exe').exists() else 'kvl_layer_probe')
        r=subprocess.run([str(exe),str(packed/'experts.bin'),str(packed/'experts.idx'),str(fixture),'65536'],text=True,capture_output=True)
        print(r.stdout,end=''); print(r.stderr,end='')
        if r.returncode: raise SystemExit(r.returncode)

if __name__=='__main__': main()
