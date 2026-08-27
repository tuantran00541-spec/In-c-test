#!/usr/bin/env python3
import json, pathlib, struct, subprocess, tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]

def bf16_bytes(seed, n):
    # arbitrary 16-bit payload; packer must preserve it byte-for-byte
    return b''.join(struct.pack('<H', (seed+i*977) & 0xffff) for i in range(n))

def write_safe(path, tensors):
    header={}; data=bytearray()
    for name,(shape,payload) in tensors.items():
        a=len(data); data += payload; b=len(data)
        header[name]={"dtype":"BF16","shape":shape,"data_offsets":[a,b]}
    raw=json.dumps(header,separators=(',',':')).encode()
    # safetensors requires JSON header padded with spaces to 8-byte alignment
    raw += b' ' * ((8-len(raw)%8)%8)
    with open(path,'wb') as f:
        f.write(struct.pack('<Q',len(raw))); f.write(raw); f.write(data)

def main():
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); model=td/'model'; out=td/'out'; model.mkdir()
        names={}; tensors={}
        # Two experts, tiny shapes. Names match Kimi-VL state_dict layout.
        for e in range(2):
            for j,part in enumerate(('gate_proj','up_proj','down_proj')):
                name=f'language_model.model.layers.1.mlp.experts.{e}.{part}.weight'
                payload=bf16_bytes(1000*e+100*j, 24+j)
                tensors[name]=([3,8],payload)
                names[name]='model-00001-of-00001.safetensors'
        write_safe(model/'model-00001-of-00001.safetensors',tensors)
        (model/'model.safetensors.index.json').write_text(json.dumps({"metadata":{"total_size":sum(len(p) for _,p in tensors.values())},"weight_map":names}))
        subprocess.run(['python',str(ROOT/'tools/pack_experts.py'),str(model),str(out)],check=True)

        HDR=struct.Struct('<8sIIIIIIQQ'); REC=struct.Struct('<IIQQQQQQQQQ')
        with open(out/'experts.idx','rb') as f:
            h=HDR.unpack(f.read(HDR.size)); recs=[REC.unpack(f.read(REC.size)) for _ in range(h[5])]
        assert h[0]==b'KVLXPRT1' and h[5]==2
        packed=(out/'experts.bin').read_bytes()
        for rec in recs:
            L,E,off,read_n,payload_n,go,gn,uo,un,do,dn=rec
            assert L==1
            for part,rel,n in [('gate_proj',go,gn),('up_proj',uo,un),('down_proj',do,dn)]:
                name=f'language_model.model.layers.1.mlp.experts.{E}.{part}.weight'
                expected=tensors[name][1]
                got=packed[off+rel:off+rel+n]
                assert got==expected, (E,part)
            assert off%4096==0 and read_n%4096==0
        print('PASS: packer preserves all expert tensor bytes and 4096-byte alignment')

if __name__=='__main__': main()
