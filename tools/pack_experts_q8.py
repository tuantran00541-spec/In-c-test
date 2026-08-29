#!/usr/bin/env python3
"""Experimental routed-expert packer using symmetric per-row int8 weights.

Each matrix blob is stored as:
  float32 scales[out_rows]
  int8    weights[out_rows, in_cols]
The containing expert record stays 4096-byte aligned for direct I/O. Router, shared expert,
attention, embeddings, LM head and vision weights remain unchanged BF16.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, struct
import numpy as np

ALIGN=4096
MAGIC=b"KVLXPRT1"
VERSION=1
DTYPE_Q8_ROW=3
HDR=struct.Struct("<8sIIIIIIQQ")
REC=struct.Struct("<IIQQQQQQQQQ")
PAT=re.compile(r"language_model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def align_up(x,a=ALIGN): return (x+a-1)//a*a


def read_st_header(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        h=json.loads(f.read(n))
    return 8+n,h


def bf16_bytes_to_f32(raw: bytes, shape) -> np.ndarray:
    u=np.frombuffer(raw,dtype='<u2').astype(np.uint32)
    bits=u << np.uint32(16)
    return bits.view(np.float32).reshape(shape)


def load_tensor(model_dir: pathlib.Path, headers, name: str, shard: str) -> np.ndarray:
    base,h=headers[shard]; meta=h[name]
    if meta['dtype']!='BF16': raise SystemExit(f"{name}: expected BF16, got {meta['dtype']}")
    a,b=meta['data_offsets']
    with open(model_dir/shard,'rb') as f:
        f.seek(base+a); raw=f.read(b-a)
    if len(raw)!=(b-a): raise IOError(f"short read {name}")
    return bf16_bytes_to_f32(raw,meta['shape'])


def quantize_rows(w: np.ndarray) -> bytes:
    if w.ndim!=2: raise ValueError(w.shape)
    maxabs=np.max(np.abs(w),axis=1)
    scales=np.where(maxabs>0,maxabs/127.0,1.0).astype('<f4')
    q=np.rint(w/scales[:,None]).clip(-127,127).astype(np.int8)
    return scales.tobytes(order='C')+q.tobytes(order='C')


def load_existing(idx_path):
    if not idx_path.exists(): return None, []
    raw=idx_path.read_bytes()
    if len(raw)<HDR.size: raise SystemExit('bad existing experts.idx')
    h=HDR.unpack_from(raw,0)
    if h[0]!=MAGIC or h[1]!=VERSION or h[2]!=ALIGN or h[6]!=DTYPE_Q8_ROW or h[7]!=HDR.size:
        raise SystemExit('incompatible existing Q8 experts.idx')
    recs=[]; off=h[7]
    for _ in range(h[5]): recs.append(REC.unpack_from(raw,off)); off+=REC.size
    return h,recs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('model_dir',type=pathlib.Path)
    ap.add_argument('out_dir',type=pathlib.Path)
    ap.add_argument('--layer',type=int,action='append')
    ap.add_argument('--append',action='store_true')
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)

    idx=json.loads((args.model_dir/'model.safetensors.index.json').read_text())
    wm=idx['weight_map']; wanted=set(args.layer) if args.layer is not None else None
    entries={}
    for name,shard in wm.items():
        m=PAT.match(name)
        if not m: continue
        L,E=int(m.group(1)),int(m.group(2)); part=m.group(3)
        if wanted is not None and L not in wanted: continue
        entries.setdefault((L,E),{})[part]=(name,shard)
    if not entries: raise SystemExit('No routed experts matched')
    missing=[k for k,v in entries.items() if set(v)!={'gate_proj','up_proj','down_proj'}]
    if missing: raise SystemExit(f'Incomplete experts, first: {missing[:3]}')

    cfg=json.loads((args.model_dir/'config.json').read_text()); tc=cfg.get('text_config',cfg)
    n_layers=int(tc['num_hidden_layers']); n_experts=int(tc['n_routed_experts'])
    binp=args.out_dir/'experts.bin'; idxp=args.out_dir/'experts.idx'
    old_h,recs=load_existing(idxp) if args.append else (None,[])
    if old_h is not None and (old_h[3]!=n_layers or old_h[4]!=n_experts):
        raise SystemExit('existing Q8 expert store dimensions mismatch')
    existing={(r[0],r[1]) for r in recs}; entries={k:v for k,v in entries.items() if k not in existing}
    if not entries:
        print('no new Q8 expert records to append'); return

    headers={}
    for shard in sorted(set(s for v in entries.values() for _,s in v.values())):
        p=args.model_dir/shard
        if not p.exists(): raise FileNotFoundError(p)
        headers[shard]=read_st_header(p)

    mode='r+b' if args.append and binp.exists() else 'wb'
    with open(binp,mode) as out:
        if mode=='r+b': out.seek(0,os.SEEK_END)
        for (L,E),parts in sorted(entries.items()):
            start=align_up(out.tell()); out.write(b'\0'*(start-out.tell()))
            offsets={}; sizes={}
            for part in ('gate_proj','up_proj','down_proj'):
                name,shard=parts[part]
                w=load_tensor(args.model_dir,headers,name,shard)
                blob=quantize_rows(w)
                offsets[part]=out.tell()-start; sizes[part]=len(blob); out.write(blob)
                del w,blob
            payload=out.tell()-start; end=align_up(out.tell()); out.write(b'\0'*(end-out.tell()))
            recs.append((L,E,start,end-start,payload,
                         offsets['gate_proj'],sizes['gate_proj'],offsets['up_proj'],sizes['up_proj'],
                         offsets['down_proj'],sizes['down_proj']))
    with open(idxp,'wb') as f:
        f.write(HDR.pack(MAGIC,VERSION,ALIGN,n_layers,n_experts,len(recs),DTYPE_Q8_ROW,HDR.size,os.path.getsize(binp)))
        for r in recs: f.write(REC.pack(*r))
    print(f'Q8 expert records={len(recs)} data={os.path.getsize(binp)/1024**3:.3f} GiB index={os.path.getsize(idxp)} bytes')


if __name__=='__main__': main()
