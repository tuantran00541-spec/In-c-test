#!/usr/bin/env python3
"""Pack Kimi-VL routed experts from local sharded safetensors into aligned experts.bin.

V5 keeps the checkpoint dtype (BF16) and adds --append so a full model can be converted
with a bounded source-shard working set. Each expert remains one contiguous direct-I/O
record containing gate/up/down in that order.
"""
import argparse, json, os, pathlib, re, struct

ALIGN=4096
MAGIC=b"KVLXPRT1"
VERSION=1
DTYPE_BF16=1
HDR=struct.Struct("<8sIIIIIIQQ")
REC=struct.Struct("<IIQQQQQQQQQ")
PAT=re.compile(r"language_model\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

def align_up(x,a=ALIGN): return (x+a-1)//a*a

def read_st_header(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        h=json.loads(f.read(n))
    return 8+n,h

def load_existing(idx_path):
    if not idx_path.exists(): return None, []
    raw=idx_path.read_bytes()
    if len(raw)<HDR.size: raise SystemExit("bad existing experts.idx")
    h=HDR.unpack_from(raw,0)
    if h[0]!=MAGIC or h[1]!=VERSION or h[2]!=ALIGN or h[6]!=DTYPE_BF16 or h[7]!=HDR.size:
        raise SystemExit("incompatible existing experts.idx")
    recs=[]; off=h[7]
    for _ in range(h[5]): recs.append(REC.unpack_from(raw,off)); off+=REC.size
    return h,recs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("model_dir",type=pathlib.Path)
    ap.add_argument("out_dir",type=pathlib.Path)
    ap.add_argument("--layer", type=int, action="append", help="pack only this decoder layer (repeatable)")
    ap.add_argument("--append", action="store_true", help="append records to an existing expert store")
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    idx=json.loads((args.model_dir/"model.safetensors.index.json").read_text())
    weight_map=idx["weight_map"]
    entries={}
    wanted=set(args.layer) if args.layer is not None else None
    for name,shard in weight_map.items():
        m=PAT.match(name)
        if not m: continue
        L,E=int(m.group(1)),int(m.group(2)); part=m.group(3)
        if wanted is not None and L not in wanted: continue
        entries.setdefault((L,E),{})[part]=(name,shard)
    if not entries: raise SystemExit("No routed expert tensor names matched expected Kimi-VL naming")
    missing=[k for k,v in entries.items() if set(v)!={"gate_proj","up_proj","down_proj"}]
    if missing: raise SystemExit(f"Incomplete experts in index, first: {missing[:3]}")

    binp=args.out_dir/"experts.bin"; idxp=args.out_dir/"experts.idx"
    old_h, recs = load_existing(idxp) if args.append else (None, [])
    existing={(r[0],r[1]) for r in recs}
    entries={k:v for k,v in entries.items() if k not in existing}
    if not entries:
        print("no new expert records to append")
        return

    headers={}
    for shard in sorted(set(s for v in entries.values() for _,s in v.values())):
        p=args.model_dir/shard
        if not p.exists(): raise FileNotFoundError(f"required source shard missing: {p}")
        headers[shard]=read_st_header(p)

    try:
        cfg=json.loads((args.model_dir/"config.json").read_text())
        tc=cfg.get("text_config",cfg)
        n_layers=int(tc["num_hidden_layers"]); n_experts=int(tc["n_routed_experts"])
    except Exception:
        n_layers=max(k[0] for k in entries)+1; n_experts=max(k[1] for k in entries)+1
    if old_h is not None and (old_h[3]!=n_layers or old_h[4]!=n_experts):
        raise SystemExit("existing expert store logical dimensions do not match model")

    mode="r+b" if args.append and binp.exists() else "wb"
    with open(binp,mode) as out:
        if mode=="r+b": out.seek(0,os.SEEK_END)
        for (L,E),parts in sorted(entries.items()):
            start=align_up(out.tell()); out.write(b"\0"*(start-out.tell()))
            offsets={}; sizes={}
            for part in ("gate_proj","up_proj","down_proj"):
                name,shard=parts[part]; base,h=headers[shard]; meta=h[name]
                if meta["dtype"] != "BF16": raise SystemExit(f"{name}: expected BF16, got {meta['dtype']}")
                a,b=meta["data_offsets"]; n=b-a
                offsets[part]=out.tell()-start; sizes[part]=n
                with open(args.model_dir/shard,"rb") as f:
                    f.seek(base+a); remain=n
                    while remain:
                        chunk=f.read(min(remain,16<<20))
                        if not chunk: raise IOError(f"short read {name}")
                        out.write(chunk); remain-=len(chunk)
            payload=out.tell()-start; end=align_up(out.tell()); out.write(b"\0"*(end-out.tell()))
            recs.append((L,E,start,end-start,payload,
                         offsets["gate_proj"],sizes["gate_proj"],offsets["up_proj"],sizes["up_proj"],
                         offsets["down_proj"],sizes["down_proj"]))
    with open(idxp,"wb") as f:
        f.write(HDR.pack(MAGIC,VERSION,ALIGN,n_layers,n_experts,len(recs),DTYPE_BF16,HDR.size,os.path.getsize(binp)))
        for r in recs: f.write(REC.pack(*r))
    print(f"expert records={len(recs)} data={os.path.getsize(binp)/1024**3:.3f} GiB index={os.path.getsize(idxp)} bytes")

if __name__=="__main__": main()
