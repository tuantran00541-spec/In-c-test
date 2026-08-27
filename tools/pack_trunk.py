#!/usr/bin/env python3
"""Pack Kimi-VL non-routed tensors into aligned trunk.bin/trunk.idx.

The format is intentionally simple: every tensor is an independently aligned direct-I/O
record. V4 streams one decoder layer's resident tensors, computes the layer, then frees
them. Later versions can coalesce records into layer-contiguous ring slots without
changing model math.
"""
import argparse, json, pathlib, struct
from safetensors import safe_open
import torch

ALIGN=4096
MAGIC=b"KVLTRNK1"
VERSION=1
GLOBAL=0xffffffff
DT_BF16=1
DT_F32=2
HDR=struct.Struct('<8s4I2Q')
REC=struct.Struct('<8I3Q')

KINDS={
 'embed':1,'final_norm':2,'lm_head':3,
 'input_norm':10,'post_norm':11,'q':12,'kva':13,'kvan':14,'kvb':15,'o':16,
 'dense_gate':20,'dense_up':21,'dense_down':22,
 'router':30,'router_bias':31,'shared_gate':32,'shared_up':33,'shared_down':34,
}

def align_up(n,a=ALIGN): return (n+a-1)//a*a

def tensor_spec(layer, dense):
    p=f'language_model.model.layers.{layer}'
    out=[
      ('input_norm',p+'.input_layernorm.weight'),
      ('post_norm',p+'.post_attention_layernorm.weight'),
      ('q',p+'.self_attn.q_proj.weight'),
      ('kva',p+'.self_attn.kv_a_proj_with_mqa.weight'),
      ('kvan',p+'.self_attn.kv_a_layernorm.weight'),
      ('kvb',p+'.self_attn.kv_b_proj.weight'),
      ('o',p+'.self_attn.o_proj.weight')]
    if dense:
        out += [('dense_gate',p+'.mlp.gate_proj.weight'),('dense_up',p+'.mlp.up_proj.weight'),('dense_down',p+'.mlp.down_proj.weight')]
    else:
        out += [('router',p+'.mlp.gate.weight'),('router_bias',p+'.mlp.gate.e_score_correction_bias'),
                ('shared_gate',p+'.mlp.shared_experts.gate_proj.weight'),('shared_up',p+'.mlp.shared_experts.up_proj.weight'),('shared_down',p+'.mlp.shared_experts.down_proj.weight')]
    return out

def load_existing(idx_path):
    if not idx_path.exists(): return []
    raw=idx_path.read_bytes()
    if len(raw)<HDR.size: raise SystemExit('bad existing trunk.idx')
    magic,version,align,nrec,reserved,roff,data_bytes=HDR.unpack_from(raw,0)
    if magic!=MAGIC or version!=VERSION or align!=ALIGN or roff!=HDR.size: raise SystemExit('incompatible existing trunk.idx')
    recs=[]
    off=roff
    for _ in range(nrec):
        recs.append(REC.unpack_from(raw,off)); off+=REC.size
    return recs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('model_dir',type=pathlib.Path)
    ap.add_argument('out_dir',type=pathlib.Path)
    ap.add_argument('--layers',default='0,1',help='comma-separated decoder layer ids')
    ap.add_argument('--include-globals',action='store_true')
    ap.add_argument('--append',action='store_true')
    args=ap.parse_args()
    layers=[int(x) for x in args.layers.split(',') if x.strip()]
    cfg=json.loads((args.model_dir/'config.json').read_text())['text_config']
    wm=json.loads((args.model_dir/'model.safetensors.index.json').read_text())['weight_map']
    first_dense=int(cfg['first_k_dense_replace']); freq=int(cfg['moe_layer_freq'])
    args.out_dir.mkdir(parents=True,exist_ok=True)
    binp=args.out_dir/'trunk.bin'; idxp=args.out_dir/'trunk.idx'
    existing=load_existing(idxp) if args.append else []
    existing_keys={(r[0],r[1]) for r in existing}
    mode='r+b' if args.append and binp.exists() else 'wb'
    records=list(existing)
    def get(name):
        shard=wm.get(name)
        if shard is None: raise KeyError(name)
        path=args.model_dir/shard
        if not path.exists(): raise FileNotFoundError(f'{path} required for {name}')
        with safe_open(path,framework='pt',device='cpu') as f: return f.get_tensor(name)
    specs=[]
    if args.include_globals:
        specs += [(GLOBAL,'embed','language_model.model.embed_tokens.weight'),
                  (GLOBAL,'final_norm','language_model.model.norm.weight'),
                  (GLOBAL,'lm_head','language_model.lm_head.weight')]
    for L in layers:
        is_moe=(int(cfg['n_routed_experts'])>0 and L>=first_dense and L%freq==0)
        for key,name in tensor_spec(L,not is_moe): specs.append((L,key,name))
    with open(binp,mode) as bf:
        if mode=='r+b': bf.seek(0,2)
        for layer,key,name in specs:
            kind=KINDS[key]
            if (layer,kind) in existing_keys: continue
            t=get(name).contiguous().cpu()
            if str(t.dtype)!='torch.bfloat16': t=t.to(dtype=torch.bfloat16)
            data=t.view(torch.uint16).numpy().tobytes()
            at=align_up(bf.tell())
            if at>bf.tell(): bf.write(b'\0'*(at-bf.tell()))
            payload=len(data); readb=align_up(payload)
            bf.write(data); bf.write(b'\0'*(readb-payload))
            dims=list(t.shape)[:4]+[0]*4; dims=dims[:4]
            records.append((layer,kind,DT_BF16,len(t.shape),*map(int,dims),at,readb,payload))
            print(f'packed L={layer if layer!=GLOBAL else "global"} {key:12s} {tuple(t.shape)} {payload/1048576:.2f} MiB')
        data_bytes=bf.tell()
    blob=bytearray(HDR.pack(MAGIC,VERSION,ALIGN,len(records),0,HDR.size,data_bytes))
    for r in records: blob.extend(REC.pack(*r))
    idxp.write_bytes(blob)
    print(f'trunk records={len(records)} data={data_bytes/1073741824:.3f} GiB index={len(blob)} bytes')

if __name__=='__main__': main()
