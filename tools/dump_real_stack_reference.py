#!/usr/bin/env python3
"""Build a V4 oracle for real Kimi-VL layers 0->1.

Resident tensors are read from our packed trunk.bin/trunk.idx, so the raw layer-0
checkpoint shard can be deleted before layer 1 is downloaded. Routed experts for layer 1
are loaded lazily from the current safetensors subset only after routing is known.
"""
import argparse, json, pathlib, struct
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MAGIC=b'KVLV4OR1'; HDR=struct.Struct('<8s18I3f6Q')
THDR=struct.Struct('<8s4I2Q'); TREC=struct.Struct('<8I3Q')
K={'input_norm':10,'post_norm':11,'q':12,'kva':13,'kvan':14,'kvb':15,'o':16,
   'dense_gate':20,'dense_up':21,'dense_down':22,'router':30,'router_bias':31,
   'shared_gate':32,'shared_up':33,'shared_down':34}

def rms(x,w,eps):
    xf=x.float(); return w.float()*xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+eps)

def rotate_half(x):
    h=x.shape[-1]//2; return torch.cat((-x[...,h:],x[...,:h]),dim=-1)

def rope(raw,theta):
    d=raw.shape[-1]; half=d//2
    perm=raw.view(*raw.shape[:-1],half,2).transpose(-1,-2).reshape(*raw.shape[:-1],d)
    pos=torch.arange(raw.shape[0],dtype=torch.float32)
    inv=theta**(-torch.arange(0,d,2,dtype=torch.float32)/d)
    emb=torch.cat((torch.outer(pos,inv),torch.outer(pos,inv)),dim=-1)
    while emb.ndim<perm.ndim: emb=emb.unsqueeze(1)
    return perm*emb.cos()+rotate_half(perm)*emb.sin()

def mla(x,c,w):
    S=x.shape[0];N=c['N'];DN=c['DN'];DR=c['DR'];DV=c['DV'];R=c['R']
    q=F.linear(x,w['q'].float()).view(S,N,DN+DR);qn,qp=q[...,:DN],q[...,DN:]
    comp=F.linear(x,w['kva'].float());latent,kp=comp[:,:R],comp[:,R:];latent=rms(latent,w['kvan'],c['eps'])
    kv=F.linear(latent,w['kvb'].float()).view(S,N,DN+DV);kn,v=kv[...,:DN],kv[...,DN:]
    q=torch.cat((qn,rope(qp,c['theta'])),dim=-1);kr=rope(kp,c['theta']).unsqueeze(1).expand(-1,N,-1);k=torch.cat((kn,kr),dim=-1)
    scores=torch.einsum('thd,shd->hts',q,k)*(1.0/((DN+DR)**0.5));mask=torch.triu(torch.full((S,S),float('-inf')),diagonal=1)
    probs=torch.softmax((scores+mask.unsqueeze(0)).float(),dim=-1);heads=torch.einsum('hts,shd->thd',probs,v).reshape(S,N*DV)
    return F.linear(heads,w['o'].float())

def dense(x,w): return F.linear(F.silu(F.linear(x,w['g'].float()))*F.linear(x,w['u'].float()),w['d'].float())

def common(x,c,w):
    n1=rms(x,w['inorm'],c['eps']); a=mla(n1,c,w);r=x+a;n2=rms(r,w['pnorm'],c['eps']);return r,n2

class Trunk:
    def __init__(self,root):
        self.bin=(root/'trunk.bin').open('rb');raw=(root/'trunk.idx').read_bytes();h=THDR.unpack_from(raw,0)
        if h[0]!=b'KVLTRNK1' or h[1]!=1: raise RuntimeError('bad trunk index')
        self.recs={};off=h[5]
        for _ in range(h[3]):
            r=TREC.unpack_from(raw,off);off+=TREC.size;self.recs[(r[0],r[1])]=r
    def get(self,L,kind):
        r=self.recs[(L,K[kind])];layer,kind_id,dtype,ndim,*tail=r;dims=tail[:4];file_off,readb,payload=tail[4:]
        self.bin.seek(file_off);data=self.bin.read(payload);arr=np.frombuffer(data,dtype=np.uint16).copy();t=torch.from_numpy(arr).view(torch.bfloat16)
        shape=tuple(int(dims[i]) for i in range(ndim));return t.reshape(shape)
    def close(self): self.bin.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('model_dir',type=pathlib.Path);ap.add_argument('packed_dir',type=pathlib.Path);ap.add_argument('out_fixture',type=pathlib.Path);ap.add_argument('--seq-len',type=int,default=4);ap.add_argument('--seed',type=int,default=270829);args=ap.parse_args()
    cfg=json.loads((args.model_dir/'config.json').read_text())['text_config'];wm=json.loads((args.model_dir/'model.safetensors.index.json').read_text())['weight_map']
    c=dict(S=args.seq_len,H=int(cfg['hidden_size']),N=int(cfg['num_attention_heads']),DN=int(cfg['qk_nope_head_dim']),DR=int(cfg['qk_rope_head_dim']),DV=int(cfg['v_head_dim']),R=int(cfg['kv_lora_rank']),DI=int(cfg['intermediate_size']),I=int(cfg['moe_intermediate_size']),SI=int(cfg['moe_intermediate_size'])*int(cfg['n_shared_experts']),E=int(cfg['n_routed_experts']),K=int(cfg['num_experts_per_tok']),eps=float(cfg['rms_norm_eps']),theta=float(cfg['rope_theta']),scale=float(cfg['routed_scaling_factor']),norm=bool(cfg['norm_topk_prob']))
    if int(cfg['first_k_dense_replace'])!=1: raise SystemExit('V4 real oracle expects dense layer 0 only')
    tr=Trunk(args.packed_dir)
    def lw(L):
        return {'inorm':tr.get(L,'input_norm'),'pnorm':tr.get(L,'post_norm'),'q':tr.get(L,'q'),'kva':tr.get(L,'kva'),'kvan':tr.get(L,'kvan'),'kvb':tr.get(L,'kvb'),'o':tr.get(L,'o')}
    w0=lw(0);w0.update({'g':tr.get(0,'dense_gate'),'u':tr.get(0,'dense_up'),'d':tr.get(0,'dense_down')})
    w1=lw(1);w1.update({'router':tr.get(1,'router').float(),'bias':tr.get(1,'router_bias').float(),'shared':{'g':tr.get(1,'shared_gate'),'u':tr.get(1,'shared_up'),'d':tr.get(1,'shared_down')}})
    torch.manual_seed(args.seed);x=(torch.randn(c['S'],c['H'])*.20).to(torch.bfloat16).float()
    r,n=common(x,c,w0);after0=r+dense(n,w0)
    r,n=common(after0,c,w1);scores=torch.sigmoid(F.linear(n,w1['router']));choice=scores+w1['bias'];_,ids=torch.topk(choice,k=c['K'],dim=-1,sorted=False);ww=scores.gather(1,ids)
    if c['norm']:ww=ww/(ww.sum(-1,keepdim=True)+1e-20)
    ww=ww*c['scale'];unique=sorted(set(ids.flatten().tolist()))
    experts={}
    for eid in unique:
        ep=f'language_model.model.layers.1.mlp.experts.{eid}'
        zs=[]
        for suffix in ('gate_proj.weight','up_proj.weight','down_proj.weight'):
            name=ep+'.'+suffix;sh=wm[name];path=args.model_dir/sh
            if not path.exists(): raise FileNotFoundError(path)
            with safe_open(path,framework='pt',device='cpu') as f: zs.append(f.get_tensor(name))
        experts[eid]={'g':zs[0],'u':zs[1],'d':zs[2]}
    outs=[]
    for t in range(c['S']):
        y=torch.zeros(c['H'],dtype=torch.float32)
        for eid,a in zip(ids[t].tolist(),ww[t].tolist()): y+=dense(n[t],experts[eid])*a
        y+=dense(n[t],w1['shared']);outs.append(y)
    final=r+torch.stack(outs);tr.close()
    blob=bytearray(HDR.size)
    def add(t,int32=False):
        off=len(blob);blob.extend((t.to(torch.int32) if int32 else t.float()).contiguous().numpy().tobytes());return off
    ox=add(x);o0=add(after0);oi=add(ids,True);ow=add(ww);of=add(final)
    header=HDR.pack(MAGIC,1,c['S'],c['H'],c['N'],c['DN'],c['DR'],c['DV'],c['R'],c['DI'],c['I'],c['SI'],c['E'],c['K'],1,1,0,2,1,c['eps'],c['theta'],c['scale'],ox,o0,oi,ow,of,len(blob));blob[:HDR.size]=header
    args.out_fixture.parent.mkdir(parents=True,exist_ok=True);args.out_fixture.write_bytes(blob)
    print('real V4 layers 0->1 seq_len',c['S'],'unique layer1 experts',len(unique),unique);print('top_ids',ids.tolist());print('fixture bytes',len(blob))

if __name__=='__main__':main()
