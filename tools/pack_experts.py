#!/usr/bin/env python3
"""Pack Kimi-VL routed experts from local sharded safetensors into aligned experts.bin.

V0 deliberately preserves the checkpoint dtype (normally BF16). This is for proving
streaming correctness before introducing MXFP4 quantization. The output format is read
by src/expert_store.c.
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("model_dir",type=pathlib.Path)
    ap.add_argument("out_dir",type=pathlib.Path)
    ap.add_argument("--layer", type=int, action="append", help="pack only this decoder layer (repeatable)")
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    idx=json.loads((args.model_dir/"model.safetensors.index.json").read_text())
    weight_map=idx["weight_map"]
    entries={}
    for name,shard in weight_map.items():
        m=PAT.match(name)
        if not m: continue
        L,E=int(m.group(1)),int(m.group(2)); part=m.group(3)
        if args.layer is not None and L not in set(args.layer):
            continue
        entries.setdefault((L,E),{})[part]=(name,shard)
    if not entries:
        raise SystemExit("No routed expert tensor names matched expected Kimi-VL naming")
    missing=[k for k,v in entries.items() if set(v)!={"gate_proj","up_proj","down_proj"}]
    if missing: raise SystemExit(f"Incomplete experts, first: {missing[:3]}")

    headers={}
    for shard in sorted(set(s for v in entries.values() for _,s in v.values())):
        headers[shard]=read_st_header(args.model_dir/shard)

    binp=args.out_dir/"experts.bin"; idxp=args.out_dir/"experts.idx"
    recs=[]
    with open(binp,"wb") as out:
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
    # Preserve the model's logical routing dimensions even for a layer-only pack, so
    # flattened (layer, expert) cache keys retain the same meaning as a full pack.
    try:
        cfg=json.loads((args.model_dir/"config.json").read_text())
        tc=cfg.get("text_config",cfg)
        n_layers=int(tc["num_hidden_layers"]); n_experts=int(tc["n_routed_experts"])
    except Exception:
        n_layers=max(k[0] for k in entries)+1; n_experts=max(k[1] for k in entries)+1
    with open(idxp,"wb") as f:
        f.write(HDR.pack(MAGIC,VERSION,ALIGN,n_layers,n_experts,len(recs),DTYPE_BF16,HDR.size,os.path.getsize(binp)))
        for r in recs: f.write(REC.pack(*r))
    print(f"packed {len(recs)} experts -> {binp} ({os.path.getsize(binp)/1024**3:.3f} GiB)")
    print(f"index -> {idxp} ({os.path.getsize(idxp)} bytes)")

if __name__=="__main__": main()
