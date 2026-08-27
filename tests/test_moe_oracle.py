#!/usr/bin/env python3
"""V2 numerical test: Torch formula oracle -> BF16 safetensors -> V1 pack/cache -> C MoE."""
import argparse, json, pathlib, struct, subprocess, tempfile
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

MAGIC=b"KVLV2OR1"
HDR=struct.Struct("<8s10If10Q")

def bf16_bytes(t: torch.Tensor) -> bytes:
    t=t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu()
    return t.numpy().tobytes()

def add(blob: bytearray, data: bytes):
    off=len(blob); blob += data; return off

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--build-dir', type=pathlib.Path, default=pathlib.Path('build'))
    args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1]
    exe=args.build_dir/'kvl_moe_probe'
    if not exe.exists():
        raise SystemExit(f"missing {exe}; build first")

    torch.manual_seed(260827)
    # Actual Kimi-VL routing cardinality (64 experts, top-6) with reduced matrix widths.
    H,I,SI,E,K,L=32,20,40,64,6,1
    SCALE=2.446
    x=torch.randn(H,dtype=torch.float32)*0.35
    rw=torch.randn(E,H,dtype=torch.float32)*0.22
    # Spread correction biases so top-k membership is safely away from numerical ties.
    bias=(torch.linspace(-0.28,0.31,E)+0.035*torch.randn(E)).to(torch.float32)

    experts=[]
    tensors={}
    for e in range(E):
        gate=(torch.randn(I,H)*0.16).to(torch.bfloat16)
        up=(torch.randn(I,H)*0.15).to(torch.bfloat16)
        down=(torch.randn(H,I)*0.14).to(torch.bfloat16)
        experts.append((gate,up,down))
        base=f"language_model.model.layers.{L}.mlp.experts.{e}"
        tensors[f"{base}.gate_proj.weight"]=gate
        tensors[f"{base}.up_proj.weight"]=up
        tensors[f"{base}.down_proj.weight"]=down

    sg=(torch.randn(SI,H)*0.12).to(torch.bfloat16)
    su=(torch.randn(SI,H)*0.11).to(torch.bfloat16)
    sd=(torch.randn(H,SI)*0.10).to(torch.bfloat16)

    # Match official MoEGate for Kimi-VL's n_group=1/noaux_tc config.
    logits=F.linear(x.float(),rw.float())
    scores=torch.sigmoid(logits)
    choice=scores+bias
    _,ids=torch.topk(choice,k=K,dim=-1,sorted=False)
    weights=scores.gather(0,ids)
    weights=weights/(weights.sum()+1e-20)*SCALE

    routed=torch.zeros(H,dtype=torch.float32)
    for eid,w in zip(ids.tolist(),weights.tolist()):
        gate,up,down=experts[eid]
        g=F.linear(x,gate.float())
        u=F.linear(x,up.float())
        y=F.linear(F.silu(g)*u,down.float())
        routed += y * w
    shared=F.linear(F.silu(F.linear(x,sg.float()))*F.linear(x,su.float()),sd.float())
    expected=(routed+shared).contiguous()

    with tempfile.TemporaryDirectory(prefix='kvl-v2-') as td:
        td=pathlib.Path(td); model=td/'model'; packed=td/'packed'; model.mkdir(); packed.mkdir()
        shard='model-00001-of-00001.safetensors'
        save_file(tensors, model/shard)
        weight_map={k:shard for k in tensors}
        (model/'model.safetensors.index.json').write_text(json.dumps({'metadata':{},'weight_map':weight_map}))
        subprocess.run(['python',str(root/'tools/pack_experts.py'),str(model),str(packed)],check=True)

        blob=bytearray(HDR.size)
        off_x=add(blob,x.numpy().tobytes())
        off_rw=add(blob,rw.numpy().tobytes())
        off_bias=add(blob,bias.numpy().tobytes())
        off_sg=add(blob,bf16_bytes(sg)); off_su=add(blob,bf16_bytes(su)); off_sd=add(blob,bf16_bytes(sd))
        off_ids=add(blob,ids.to(torch.int32).cpu().numpy().tobytes())
        off_w=add(blob,weights.float().cpu().numpy().tobytes())
        off_out=add(blob,expected.float().cpu().numpy().tobytes())
        header=HDR.pack(MAGIC,1,H,I,SI,E,K,1,1,L,1,SCALE,
                        off_x,off_rw,off_bias,off_sg,off_su,off_sd,off_ids,off_w,off_out,len(blob))
        blob[:HDR.size]=header
        fixture=td/'fixture.bin'; fixture.write_bytes(blob)

        # Exactly top-k slots: getmany must reserve six distinct INFLIGHT slots, with no slack.
        # Each synthetic expert occupies one aligned 4096-byte record.
        budget=K*4096
        p=subprocess.run([str(exe),str(packed/'experts.bin'),str(packed/'experts.idx'),str(fixture),str(budget)],
                         text=True,capture_output=True)
        print(p.stdout,end=''); print(p.stderr,end='')
        if p.returncode: raise SystemExit(p.returncode)
        print('PASS: Torch router + BF16 routed/shared MoE matches streamed C path')

if __name__=='__main__': main()
