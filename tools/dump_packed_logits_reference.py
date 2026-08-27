#!/usr/bin/env python3
"""Generate a one-token full-text-model V5 oracle from runtime packed files.

No raw safetensors are required. Trunk and expert weights are memory-mapped from the same
BF16 runtime backing stores used by C. The fixture records the hidden state after every
decoder layer plus complete vocabulary logits so a mismatch can be localized.
"""
from __future__ import annotations
import argparse,json,pathlib,struct
import numpy as np
import torch
import torch.nn.functional as F

MAGIC=b'KVLV5OR1';FHDR=struct.Struct('<8s20I3f3Q')
TMAGIC=b'KVLTRNK1';THDR=struct.Struct('<8s4I2Q');TREC=struct.Struct('<8I3Q');GLOBAL=0xffffffff
EMBED=1;FINAL_NORM=2;LM_HEAD=3;INORM=10;PNORM=11;Q=12;KVA=13;KVAN=14;KVB=15;O=16;DG=20;DU=21;DD=22;ROUTER=30;RBIAS=31;SG=32;SU=33;SD=34
EMAGIC=b'KVLXPRT1';EHDR=struct.Struct('<8sIIIIIIQQ');EREC=struct.Struct('<IIQQQQQQQQQ')

def bf16_map(path,off,shape):
    n=int(np.prod(shape));a=np.memmap(path,dtype=np.uint16,mode='c',offset=int(off),shape=(n,));return torch.from_numpy(a).view(torch.bfloat16).reshape(shape)

class Trunk:
    def __init__(self,root):
        self.path=root/'trunk.bin';raw=(root/'trunk.idx').read_bytes();h=THDR.unpack_from(raw,0)
        if h[0]!=TMAGIC or h[1]!=1:raise RuntimeError('bad trunk index')
        self.r={};off=h[5]
        for _ in range(h[3]):
            z=TREC.unpack_from(raw,off);off+=TREC.size;self.r[(z[0],z[1])]=z
    def get(self,L,k):
        z=self.r[(L,k)];nd=z[3];shape=tuple(int(z[4+i]) for i in range(nd));return bf16_map(self.path,z[8],shape)

class Experts:
    def __init__(self,root):
        self.path=root/'experts.bin';raw=(root/'experts.idx').read_bytes();h=EHDR.unpack_from(raw,0)
        if h[0]!=EMAGIC or h[1]!=1:raise RuntimeError('bad expert index')
        self.r={};off=h[7]
        for _ in range(h[5]):
            z=EREC.unpack_from(raw,off);off+=EREC.size;self.r[(z[0],z[1])]=z
    def get(self,L,E,H,I):
        z=self.r[(L,E)];base=z[2]
        return {'g':bf16_map(self.path,base+z[5],(I,H)),'u':bf16_map(self.path,base+z[7],(I,H)),'d':bf16_map(self.path,base+z[9],(H,I))}

def rms(x,w,eps):
    xf=x.float();return w.float()*xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+eps)
def mlp(x,w):return F.linear(F.silu(F.linear(x,w['g'].float()))*F.linear(x,w['u'].float()),w['d'].float())
def attention_one(x,c,w):
    N,DN,DR,DV,R=c['N'],c['DN'],c['DR'],c['DV'],c['R'];n=rms(x,w['inorm'],c['eps'])
    _q=F.linear(n,w['q'].float()).view(N,DN+DR)
    comp=F.linear(n,w['kva'].float());lat=comp[:R];lat=rms(lat,w['kvan'],c['eps'])
    kv=F.linear(lat,w['kvb'].float()).view(N,DN+DV);v=kv[...,DN:].reshape(N*DV)
    a=F.linear(v,w['o'].float());r=x+a;return r,rms(r,w['pnorm'],c['eps'])
def topk_router(x,c,router,bias):
    scores=torch.sigmoid(F.linear(x,router.float()));choice=scores+bias.float()
    if c['groups']!=1:
        per=c['E']//c['groups'];cv=choice.view(c['groups'],per);gs=torch.topk(cv,k=min(2,per),dim=-1).values.sum(-1);keep=torch.topk(gs,k=c['topkg'],sorted=True).indices
        mask=torch.zeros(c['groups'],dtype=torch.bool);mask[keep]=True;choice=choice.masked_fill(~mask.repeat_interleave(per),0.0)
    ids=torch.topk(choice,k=c['K'],sorted=True).indices;ww=scores[ids]
    if c['norm']:ww=ww/(ww.sum()+1e-20)
    return ids,ww*c['scale']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('metadata_dir',type=pathlib.Path);ap.add_argument('packed_dir',type=pathlib.Path);ap.add_argument('out_fixture',type=pathlib.Path);ap.add_argument('--token-id',type=int,default=1);ap.add_argument('--logit-chunk',type=int,default=8192);args=ap.parse_args()
    tc=json.loads((args.metadata_dir/'config.json').read_text())['text_config']
    c={'V':int(tc['vocab_size']),'H':int(tc['hidden_size']),'N':int(tc['num_attention_heads']),'DN':int(tc['qk_nope_head_dim']),'DR':int(tc['qk_rope_head_dim']),'DV':int(tc['v_head_dim']),'R':int(tc['kv_lora_rank']),'DI':int(tc['intermediate_size']),'I':int(tc['moe_intermediate_size']),'SI':int(tc['moe_intermediate_size'])*int(tc['n_shared_experts']),'E':int(tc['n_routed_experts']),'K':int(tc['num_experts_per_tok']),'groups':int(tc['n_group']),'topkg':int(tc['topk_group']),'L':int(tc['num_hidden_layers']),'first':int(tc['first_k_dense_replace']),'freq':int(tc['moe_layer_freq']),'norm':bool(tc['norm_topk_prob']),'eps':float(tc['rms_norm_eps']),'theta':float(tc['rope_theta']),'scale':float(tc['routed_scaling_factor'])}
    if not (0<=args.token_id<c['V']):raise SystemExit('token id out of range')
    tr=Trunk(args.packed_dir);ex=Experts(args.packed_dir);embed=tr.get(GLOBAL,EMBED);x=embed[args.token_id].float().clone();del embed
    expected=[]
    for L in range(c['L']):
        w={'inorm':tr.get(L,INORM),'pnorm':tr.get(L,PNORM),'q':tr.get(L,Q),'kva':tr.get(L,KVA),'kvan':tr.get(L,KVAN),'kvb':tr.get(L,KVB),'o':tr.get(L,O)};r,n=attention_one(x,c,w)
        is_moe=L>=c['first'] and L%c['freq']==0
        if not is_moe:y=mlp(n,{'g':tr.get(L,DG),'u':tr.get(L,DU),'d':tr.get(L,DD)});route='dense'
        else:
            ids,ww=topk_router(n,c,tr.get(L,ROUTER),tr.get(L,RBIAS));y=torch.zeros(c['H'],dtype=torch.float32)
            for eid,a in zip(ids.tolist(),ww.tolist()):y+=mlp(n,ex.get(L,eid,c['H'],c['I']))*a
            y+=mlp(n,{'g':tr.get(L,SG),'u':tr.get(L,SU),'d':tr.get(L,SD)});route=f'top6={ids.tolist()}'
        x=r+y;expected.append(x.clone());print(f'layer {L:02d}/{c["L"]-1}: maxabs={x.abs().max().item():.6g} {route}',flush=True)
    z=rms(x,tr.get(GLOBAL,FINAL_NORM),c['eps']);lm=tr.get(GLOBAL,LM_HEAD);chunks=[]
    for a in range(0,c['V'],args.logit_chunk):
        b=min(c['V'],a+args.logit_chunk);chunks.append(F.linear(z,lm[a:b].float()).cpu())
    logits=torch.cat(chunks);argmax=int(torch.argmax(logits));print('logits argmax',argmax,'max',float(logits[argmax]),flush=True)
    blob=bytearray(FHDR.size);ol=len(blob);blob.extend(torch.stack(expected).float().contiguous().numpy().tobytes());og=len(blob);blob.extend(logits.float().contiguous().numpy().tobytes())
    ints=[1,args.token_id,c['V'],c['H'],c['N'],c['DN'],c['DR'],c['DV'],c['R'],c['DI'],c['I'],c['SI'],c['E'],c['K'],c['groups'],c['topkg'],c['L'],c['first'],c['freq'],1 if c['norm'] else 0]
    blob[:FHDR.size]=FHDR.pack(MAGIC,*ints,c['eps'],c['theta'],c['scale'],ol,og,len(blob));args.out_fixture.parent.mkdir(parents=True,exist_ok=True);args.out_fixture.write_bytes(blob);print('fixture',args.out_fixture,'bytes',len(blob))
if __name__=='__main__':main()
