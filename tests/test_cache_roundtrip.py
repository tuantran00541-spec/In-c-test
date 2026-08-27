#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import struct
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h


def bf16_bytes(seed, n):
    return b''.join(struct.pack('<H', (seed + i * 977) & 0xffff) for i in range(n))


def write_safe(path, tensors):
    header = {}
    data = bytearray()
    for name, (shape, payload) in tensors.items():
        a = len(data); data += payload; b = len(data)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [a, b]}
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * ((8 - len(raw) % 8) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(raw))); f.write(raw); f.write(data)


def make_fixture(model: pathlib.Path, n_experts=12):
    names = {}; tensors = {}
    for e in range(n_experts):
        for j, part in enumerate(('gate_proj', 'up_proj', 'down_proj')):
            name = f'language_model.model.layers.1.mlp.experts.{e}.{part}.weight'
            # Non-identical sizes exercise internal offsets while each record still rounds to 4096.
            payload = bf16_bytes(10000 * e + 100 * j, 120 + 7 * j + e)
            tensors[name] = ([1, len(payload)//2], payload)
            names[name] = 'model-00001-of-00001.safetensors'
    write_safe(model/'model-00001-of-00001.safetensors', tensors)
    (model/'model.safetensors.index.json').write_text(json.dumps({
        "metadata": {"total_size": sum(len(p) for _, p in tensors.values())},
        "weight_map": names,
    }))
    return tensors


def packed_payloads(out: pathlib.Path):
    HDR = struct.Struct('<8sIIIIIIQQ'); REC = struct.Struct('<IIQQQQQQQQQ')
    with open(out/'experts.idx', 'rb') as f:
        h = HDR.unpack(f.read(HDR.size))
        recs = [REC.unpack(f.read(REC.size)) for _ in range(h[5])]
    data = (out/'experts.bin').read_bytes()
    ret = {}
    for r in recs:
        L,E,off,read_n,payload_n,*_ = r
        ret[(L,E)] = data[off:off+payload_n]
        assert off % 4096 == 0 and read_n % 4096 == 0
    return ret


def expected_checksum(payloads, layer, ids, repeats):
    checksum = 0
    n = len(ids)
    for rep in range(repeats):
        for i,e in enumerate(ids):
            checksum ^= (fnv1a64(payloads[(layer,e)]) + (i + 1 + rep*n)) & MASK64
    return checksum & MASK64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-dir', type=pathlib.Path, default=ROOT/'build')
    args = ap.parse_args()
    exe = args.build_dir / ('kvl_cache_probe.exe' if __import__('os').name == 'nt' else 'kvl_cache_probe')
    if not exe.exists():
        raise SystemExit(f'missing {exe}; build the project first')

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td); model = td/'model'; out = td/'out'; model.mkdir()
        make_fixture(model)
        subprocess.run(['python', str(ROOT/'tools/pack_experts.py'), str(model), str(out)], check=True)
        payloads = packed_payloads(out)

        ids = [0,1,2,3,4,5]
        repeats = 25
        # Each synthetic record occupies exactly one 4096-byte cache slot. Three slots means
        # top-6 cannot all be resident and forces reserve/publish/evict behavior repeatedly.
        budget = 3 * 4096
        cmd = [str(exe), str(out/'experts.bin'), str(out/'experts.idx'), str(budget), '1', str(repeats)] + [str(x) for x in ids]
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(proc.stdout.strip())
        print(proc.stderr.strip())
        m = re.search(r'checksum=([0-9a-fA-F]+) slots=(\d+) arena_bytes=(\d+) budget_bytes=(\d+)', proc.stdout)
        assert m, proc.stdout
        got = int(m.group(1), 16)
        assert got == expected_checksum(payloads, 1, ids, repeats), (hex(got), hex(expected_checksum(payloads,1,ids,repeats)))
        assert int(m.group(2)) == 3
        assert int(m.group(3)) <= int(m.group(4)) == budget
        assert 'failures=0' in proc.stderr
        # With only 3 slots for a repeated top-6 sequence, misses and evictions are mandatory.
        mm = re.search(r'hit=(\d+) miss=(\d+).*evict=(\d+)', proc.stderr)
        assert mm and int(mm.group(2)) > 0 and int(mm.group(3)) > 0, proc.stderr
        print('PASS: top-k cache survives repeated over-capacity prefetch/get cycles with exact payload checksums')


if __name__ == '__main__':
    main()
