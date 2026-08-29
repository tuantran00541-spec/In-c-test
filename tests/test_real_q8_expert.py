#!/usr/bin/env python3
"""Compare one released Kimi-VL routed expert against the lab row-wise Q8 C kernel."""
from __future__ import annotations
import argparse, json, math, pathlib, struct, subprocess, tempfile
import numpy as np
import torch
from safetensors import safe_open

HDR=struct.Struct('<8sIIIIIIQQ')
REC=struct.Struct('<IIQQQQQQQQQ')


def read_record(idx_path: pathlib.Path, layer: int, expert: int):
    raw=idx_path.read_bytes(); h=HDR.unpack_from(raw,0)
    off=h[7]
    for _ in range(h[5]):
        r=REC.unpack_from(raw,off); off += REC.size
        if r[0]==layer and r[1]==expert: return h,r
    raise KeyError((layer,expert))


def run_q8(exe: pathlib.Path, blob: bytes, x: np.ndarray, in_dim: int, out_dim: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix='kvl-real-q8-') as td:
        td=pathlib.Path(td); wp=td/'w.q8'; xp=td/'x.f32'; yp=td/'y.f32'
        wp.write_bytes(blob); xp.write_bytes(np.asarray(x,dtype='<f4').tobytes())
        subprocess.run([str(exe),str(wp),str(xp),str(in_dim),str(out_dim),str(yp)],check=True)
        return np.fromfile(yp,dtype='<f4')


def rel_rmse(a: np.ndarray,b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a-b)**2))/max(np.sqrt(np.mean(a**2)),1e-12))


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('model_dir',type=pathlib.Path)
    ap.add_argument('q8_dir',type=pathlib.Path)
    ap.add_argument('--build-dir',type=pathlib.Path,default=pathlib.Path('build'))
    ap.add_argument('--layer',type=int,default=1)
    ap.add_argument('--expert',type=int,default=0)
    args=ap.parse_args()
    exe=args.build_dir/('kvl_q8_probe.exe' if __import__('os').name=='nt' else 'kvl_q8_probe')
    if not exe.exists():
        p=args.build_dir/'Release'/'kvl_q8_probe.exe'
        if p.exists(): exe=p
    if not exe.exists(): raise SystemExit(f'missing {exe}')

    cfg=json.loads((args.model_dir/'config.json').read_text())['text_config']
    H=int(cfg['hidden_size']); I=int(cfg['moe_intermediate_size'])
    wm=json.loads((args.model_dir/'model.safetensors.index.json').read_text())['weight_map']
    base=f'language_model.model.layers.{args.layer}.mlp.experts.{args.expert}'
    names={p:f'{base}.{p}_proj.weight' for p in ('gate','up','down')}
    shards={wm[n] for n in names.values()}
    if len(shards)!=1: raise SystemExit(f'expert spans shards unexpectedly: {shards}')
    shard=args.model_dir/next(iter(shards))
    with safe_open(shard,framework='pt',device='cpu') as f:
        gate=f.get_tensor(names['gate']).float(); up=f.get_tensor(names['up']).float(); down=f.get_tensor(names['down']).float()
    assert tuple(gate.shape)==(I,H) and tuple(up.shape)==(I,H) and tuple(down.shape)==(H,I)

    ih,rec=read_record(args.q8_dir/'experts.idx',args.layer,args.expert)
    assert ih[6]==3,ih[6]
    raw=(args.q8_dir/'experts.bin').read_bytes()[rec[2]:rec[2]+rec[3]]
    blobs={
        'gate':raw[rec[5]:rec[5]+rec[6]],
        'up':raw[rec[7]:rec[7]+rec[8]],
        'down':raw[rec[9]:rec[9]+rec[10]],
    }
    expected_gu=I*4+I*H; expected_dn=H*4+H*I
    assert len(blobs['gate'])==expected_gu and len(blobs['up'])==expected_gu and len(blobs['down'])==expected_dn

    rng=np.random.default_rng(20260829)
    cases=[]
    for sigma in (0.08,0.25,0.7): cases.append(rng.normal(0,sigma,size=H).astype(np.float32))
    # Structured vector catches rows that depend on a few large coordinates.
    x=np.zeros(H,dtype=np.float32); x[0]=2.0; x[17]=-1.5; x[511]=0.75; x[-1]=-0.5; cases.append(x)

    worst_gate=worst_up=worst_out=0.0
    with torch.no_grad():
        for ci,x in enumerate(cases):
            xt=torch.from_numpy(x)
            bg=(gate@xt).numpy(); bu=(up@xt).numpy()
            bact=(torch.nn.functional.silu(torch.from_numpy(bg))*torch.from_numpy(bu)).numpy()
            bout=(down@torch.from_numpy(bact)).numpy()

            qg=run_q8(exe,blobs['gate'],x,H,I); qu=run_q8(exe,blobs['up'],x,H,I)
            qact=(qg/(1.0+np.exp(-qg))*qu).astype(np.float32)
            qout=run_q8(exe,blobs['down'],qact,I,H)
            eg,eu,eo=rel_rmse(bg,qg),rel_rmse(bu,qu),rel_rmse(bout,qout)
            worst_gate=max(worst_gate,eg); worst_up=max(worst_up,eu); worst_out=max(worst_out,eo)
            print(f'CASE={ci} gate_rel_rmse={eg:.6g} up_rel_rmse={eu:.6g} mlp_rel_rmse={eo:.6g}')

    bf16_record=3*I*H*2
    q8_payload=rec[4]
    print(f'REAL_H={H} REAL_I={I}')
    print(f'REAL_Q8_RECORD_READ_BYTES={rec[3]} REAL_Q8_PAYLOAD_BYTES={q8_payload}')
    print(f'REAL_BF16_EXPERT_PAYLOAD_BYTES={bf16_record}')
    print(f'REAL_SIZE_RATIO={q8_payload/bf16_record:.6f}')
    print(f'WORST_GATE_REL_RMSE={worst_gate:.9g}')
    print(f'WORST_UP_REL_RMSE={worst_up:.9g}')
    print(f'WORST_MLP_REL_RMSE={worst_out:.9g}')
    # This is a lab gate, not a release criterion. Keep it conservative until token tests.
    assert q8_payload < bf16_record*0.52
    assert worst_gate < 0.02 and worst_up < 0.02
    assert worst_out < 0.04
    print('PASS: released Kimi routed expert survives row-wise Q8 lab gate')
    return 0

if __name__=='__main__': raise SystemExit(main())
