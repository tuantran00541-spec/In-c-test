#!/usr/bin/env python3
import argparse, json, pathlib, struct, subprocess, tempfile
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

MAGIC=b"KVLV4OR1"
HDR=struct.Struct('<8s18I3f6Q')

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
    q=F.linear(x,w['q'].float()).view(S,N,DN+DR); qn,qp=q[...,:DN],q[...,DN:]
    comp=F.linear(x,w['kva'].float()); latent,kp=comp[:,:R],comp[:,R:]
    latent=rms(latent,w['kvan'],c['eps']); kv=F.linear(latent,w['kvb'].float()).view(S,N,DN+DV)
    kn,v=kv[...,:DN],kv[...,DN:]
    q=torch.cat((qn,rope(qp,c['theta'])),dim=-1); kr=rope(kp,c['theta']).unsqueeze(1).expand(-1,N,-1); k=torch.cat((kn,kr),dim=-1)
    scores=torch.einsum('thd,shd->hts',q,k)*(1.0/((DN+DR)**0.5)); mask=torch.triu(torch.full((S,S),float('-inf')),diagonal=1)
    probs=torch.softmax((scores+mask.unsqueeze(0)).float(),dim=-1); heads=torch.einsum('hts,shd->thd',probs,v).reshape(S,N*DV)
    return F.linear(heads,w['o'].float())

def dense_mlp(x,w): return F.linear(F.silu(F.linear(x,w['g'].float()))*F.linear(x,w['u'].float()),w['d'].float())

def moe(x,c,w,experts):
    scores=torch.sigmoid(F.linear(x,w['router'].float())); choice=scores+w['bias'].float(); _,ids=torch.topk(choice,k=c['K'],dim=-1,sorted=False)
    ww=scores.gather(1,ids); ww=ww/(ww.sum(-1,keepdim=True)+1e-20) if c['norm'] else ww; ww=ww*c['scale']
    outs=[]
    for t in range(x.shape[0]):
        y=torch.zeros(c['H'],dtype=torch.float32)
        for eid,a in zip(ids[t].tolist(),ww[t].tolist()): y+=dense_mlp(x[t],experts[eid])*a
        y+=dense_mlp(x[t],w['shared']); outs.append(y)
    return torch.stack(outs),ids,ww

def common_layer(x,c,w):
    n1=rms(x,w['inorm'],c['eps']); a=mla(n1,c,w); r1=x+a; n2=rms(r1,w['pnorm'],c['eps']); return r1,n2

def write_fixture(path,c,x,after0,ids,ww,final):
    blob=bytearray(HDR.size)
    def add(t,kind='f'):
        off=len(blob)
        if kind=='i': blob.extend(t.to(torch.int32).contiguous().numpy().tobytes())
        else: blob.extend(t.float().contiguous().numpy().tobytes())
        return off
    ox=add(x); o0=add(after0); oi=add(ids,'i'); ow=add(ww); of=add(final)
    header=HDR.pack(MAGIC,1,c['S'],c['H'],c['N'],c['DN'],c['DR'],c['DV'],c['R'],c['DI'],c['I'],c['SI'],c['E'],c['K'],1,1,0,2,1,
                    c['eps'],c['theta'],c['scale'],ox,o0,oi,ow,of,len(blob))
    blob[:HDR.size]=header; path.write_bytes(blob)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--build-dir',type=pathlib.Path,required=True); args=ap.parse_args()
    torch.manual_seed(270828)
    c=dict(S=4,H=16,N=2,DN=4,DR=4,DV=4,R=6,DI=24,I=8,SI=16,E=64,K=6,eps=1e-5,theta=800000.0,scale=2.446,norm=True)
    def b(shape,s=.08): return (torch.randn(*shape)*s).to(torch.bfloat16)
    def attnweights(): return {'inorm':b((c['H'],),.2)+torch.tensor(1.,dtype=torch.bfloat16),'pnorm':b((c['H'],),.2)+torch.tensor(1.,dtype=torch.bfloat16),
      'q':b((c['N']*(c['DN']+c['DR']),c['H'])),'kva':b((c['R']+c['DR'],c['H'])),'kvan':b((c['R'],),.2)+torch.tensor(1.,dtype=torch.bfloat16),
      'kvb':b((c['N']*(c['DN']+c['DV']),c['R'])),'o':b((c['H'],c['N']*c['DV']))}
    w0=attnweights(); w0.update({'g':b((c['DI'],c['H'])),'u':b((c['DI'],c['H'])),'d':b((c['H'],c['DI']))})
    w1=attnweights(); w1.update({'router':b((c['E'],c['H'])),'bias':b((c['E'],),.02),
                                'shared':{'g':b((c['SI'],c['H'])),'u':b((c['SI'],c['H'])),'d':b((c['H'],c['SI']))}})
    experts={e:{'g':b((c['I'],c['H'])),'u':b((c['I'],c['H'])),'d':b((c['H'],c['I']))} for e in range(c['E'])}
    x=(torch.randn(c['S'],c['H'])*.2).to(torch.bfloat16).float()
    r,n=common_layer(x,c,w0); after0=r+dense_mlp(n,w0)
    r,n=common_layer(after0,c,w1); m,ids,ww=moe(n,c,w1,experts); final=r+m
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); model=td/'model';model.mkdir();packed=td/'packed';fixture=td/'v4.fixture'; names={}
        for L,w in [(0,w0),(1,w1)]:
            p=f'language_model.model.layers.{L}'; names[p+'.input_layernorm.weight']=w['inorm'];names[p+'.post_attention_layernorm.weight']=w['pnorm']
            names[p+'.self_attn.q_proj.weight']=w['q'];names[p+'.self_attn.kv_a_proj_with_mqa.weight']=w['kva'];names[p+'.self_attn.kv_a_layernorm.weight']=w['kvan'];names[p+'.self_attn.kv_b_proj.weight']=w['kvb'];names[p+'.self_attn.o_proj.weight']=w['o']
        p='language_model.model.layers.0.mlp'; names[p+'.gate_proj.weight']=w0['g'];names[p+'.up_proj.weight']=w0['u'];names[p+'.down_proj.weight']=w0['d']
        p='language_model.model.layers.1.mlp'; names[p+'.gate.weight']=w1['router'];names[p+'.gate.e_score_correction_bias']=w1['bias']
        for key,k in [('gate_proj.weight','g'),('up_proj.weight','u'),('down_proj.weight','d')]: names[p+'.shared_experts.'+key]=w1['shared'][k]
        for e,z in experts.items():
            ep=f'{p}.experts.{e}'; names[ep+'.gate_proj.weight']=z['g'];names[ep+'.up_proj.weight']=z['u'];names[ep+'.down_proj.weight']=z['d']
        shard='model-00001-of-00001.safetensors';save_file(names,str(model/shard));wm={k:shard for k in names}
        cfg={'text_config':{'hidden_size':c['H'],'intermediate_size':c['DI'],'moe_intermediate_size':c['I'],'num_hidden_layers':27,'num_attention_heads':c['N'],'n_shared_experts':2,'n_routed_experts':c['E'],'routed_scaling_factor':c['scale'],'kv_lora_rank':c['R'],'q_lora_rank':None,'qk_rope_head_dim':c['DR'],'v_head_dim':c['DV'],'qk_nope_head_dim':c['DN'],'n_group':1,'topk_group':1,'num_experts_per_tok':c['K'],'moe_layer_freq':1,'first_k_dense_replace':1,'norm_topk_prob':True,'rms_norm_eps':c['eps'],'rope_theta':c['theta']}}
        (model/'config.json').write_text(json.dumps(cfg));(model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':wm}))
        root=pathlib.Path(__file__).parents[1]
        subprocess.run(['python',str(root/'tools/pack_trunk.py'),str(model),str(packed),'--layers','0,1'],check=True)
        subprocess.run(['python',str(root/'tools/pack_experts.py'),str(model),str(packed),'--layer','1'],check=True)
        write_fixture(fixture,c,x,after0,ids,ww,final)
        exe=args.build_dir/('kvl_stack_probe.exe' if (args.build_dir/'kvl_stack_probe.exe').exists() else 'kvl_stack_probe')
        r=subprocess.run([str(exe),str(packed/'trunk.bin'),str(packed/'trunk.idx'),str(packed/'experts.bin'),str(packed/'experts.idx'),str(fixture),'65536'],text=True,capture_output=True)
        print(r.stdout,end='');print(r.stderr,end='')
        if r.returncode: raise SystemExit(r.returncode)

if __name__=='__main__': main()
