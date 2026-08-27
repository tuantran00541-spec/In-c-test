#!/usr/bin/env python3
"""Resolve/download only checkpoint shards for selected Kimi-VL decoder layers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from huggingface_hub import hf_hub_download

DEFAULT_REPO='moonshotai/Kimi-VL-A3B-Instruct'

def fetch(repo,filename,out,revision):
    return Path(hf_hub_download(repo_id=repo,filename=filename,revision=revision,local_dir=str(out)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out_dir',type=Path);ap.add_argument('--repo',default=DEFAULT_REPO);ap.add_argument('--revision',default='main');ap.add_argument('--layers',default='0,1');ap.add_argument('--metadata-only',action='store_true');args=ap.parse_args()
    layers=[int(x) for x in args.layers.split(',') if x.strip()];args.out_dir.mkdir(parents=True,exist_ok=True)
    fetch(args.repo,'config.json',args.out_dir,args.revision);idxp=fetch(args.repo,'model.safetensors.index.json',args.out_dir,args.revision)
    wm=json.loads(idxp.read_text())['weight_map']; names=[]
    for L in layers:
        p=f'language_model.model.layers.{L}.'; hit=sorted(k for k in wm if k.startswith(p))
        if not hit: raise SystemExit(f'no tensors for layer {L}')
        print(f'layer {L}: tensors={len(hit)} shards={sorted({wm[k] for k in hit})}')
        names += hit
    shards=sorted({wm[k] for k in names});print('union shards:',len(shards),shards)
    plan={'repo_id':args.repo,'revision':args.revision,'layers':layers,'tensor_count':len(names),'shards':shards}
    (args.out_dir/'real_v4_download_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.metadata_only:return
    for i,s in enumerate(shards,1): print(f'[{i}/{len(shards)}] downloading {s}');fetch(args.repo,s,args.out_dir,args.revision)

if __name__=='__main__':main()
