#!/usr/bin/env python3
"""Bounded-working-set converter for the complete Kimi-VL text model.

Downloads checkpoint shards in filename order, keeps only shards still needed to complete a
layer that crosses a shard boundary, appends runtime trunk/expert records, then deletes
consumed source shards. The released Kimi-VL-A3B checkpoint needs at most two adjacent
~5 GB source shards simultaneously.
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
from huggingface_hub import hf_hub_download

REPO='moonshotai/Kimi-VL-A3B-Instruct'
GLOBALS={'embed':'language_model.model.embed_tokens.weight','final_norm':'language_model.model.norm.weight','lm_head':'language_model.lm_head.weight'}

def dl(repo,rev,root,name): return pathlib.Path(hf_hub_download(repo_id=repo,revision=rev,filename=name,local_dir=str(root)))
def run(cmd): print('+',' '.join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('work_dir',type=pathlib.Path);ap.add_argument('out_dir',type=pathlib.Path);ap.add_argument('--repo',default=REPO);ap.add_argument('--revision',default='main');ap.add_argument('--keep-source-shards',action='store_true');ap.add_argument('--max-layer',type=int,default=None);args=ap.parse_args()
    args.work_dir.mkdir(parents=True,exist_ok=True);args.out_dir.mkdir(parents=True,exist_ok=True)
    dl(args.repo,args.revision,args.work_dir,'config.json');idxp=dl(args.repo,args.revision,args.work_dir,'model.safetensors.index.json')
    cfg=json.loads((args.work_dir/'config.json').read_text())['text_config'];wm=json.loads(idxp.read_text())['weight_map'];n_layers=int(cfg['num_hidden_layers']);max_layer=n_layers-1 if args.max_layer is None else min(args.max_layer,n_layers-1)
    layer_req={}
    for L in range(max_layer+1):
        p=f'language_model.model.layers.{L}.';names=[k for k in wm if k.startswith(p)]
        if not names:raise SystemExit(f'no tensors for layer {L}')
        layer_req[L]={wm[n] for n in names}
    global_req={g:wm[n] for g,n in GLOBALS.items()};all_shards=sorted(set().union(*layer_req.values(),set(global_req.values())))
    print('source shard order:',all_shards)
    for L in range(max_layer+1):print(f'  L{L:02d}: {sorted(layer_req[L])}')
    print('  globals:',global_req)
    here=pathlib.Path(__file__).resolve().parent;done_layers=set();done_globals=set();loaded=set()
    def ta():return ['--append'] if (args.out_dir/'trunk.idx').exists() else []
    def ea():return ['--append'] if (args.out_dir/'experts.idx').exists() else []
    for si,shard in enumerate(all_shards,1):
        print(f'\n=== source shard {si}/{len(all_shards)}: {shard} ===',flush=True);dl(args.repo,args.revision,args.work_dir,shard);loaded.add(shard)
        for g,gs in global_req.items():
            if g not in done_globals and gs in loaded:
                run([sys.executable,here/'pack_trunk.py',args.work_dir,args.out_dir,'--layers','', '--global',g,*ta()]);done_globals.add(g)
        progress=True
        while progress:
            progress=False
            for L in range(max_layer+1):
                if L in done_layers or not layer_req[L].issubset(loaded):continue
                run([sys.executable,here/'pack_trunk.py',args.work_dir,args.out_dir,'--layers',str(L),*ta()])
                if L>=int(cfg['first_k_dense_replace']):run([sys.executable,here/'pack_experts.py',args.work_dir,args.out_dir,'--layer',str(L),*ea()])
                done_layers.add(L);progress=True;print(f'completed layer {L}; {len(done_layers)}/{max_layer+1} layers packed',flush=True)
        if not args.keep_source_shards:
            still=set()
            for L,req in layer_req.items():
                if L not in done_layers:still|=req
            for g,gs in global_req.items():
                if g not in done_globals:still.add(gs)
            for s in sorted(list(loaded)):
                if s not in still:
                    p=args.work_dir/s
                    if p.exists():print('deleting consumed source shard',p,flush=True);p.unlink()
                    loaded.remove(s)
        print('source shards still resident:',sorted(loaded),flush=True)
    missL=sorted(set(range(max_layer+1))-done_layers);missG=sorted(set(GLOBALS)-done_globals)
    if missL or missG:raise SystemExit(f'incomplete conversion: layers={missL} globals={missG}')
    print('\nfull text runtime pack ready')
    for n in ('trunk.bin','trunk.idx','experts.bin','experts.idx'):
        p=args.out_dir/n
        if p.exists():print(f'  {n}: {p.stat().st_size/1024**3:.3f} GiB' if p.suffix=='.bin' else f'  {n}: {p.stat().st_size} bytes')
if __name__=='__main__':main()
