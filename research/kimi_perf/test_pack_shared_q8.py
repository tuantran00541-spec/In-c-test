#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import struct
import subprocess
import sys
import tempfile
import numpy as np

ALIGN = 4096
THDR = struct.Struct("<8s4I2Q")
TREC = struct.Struct("<8I3Q")
EHDR = struct.Struct("<8sIIIIIIQQ")
EREC = struct.Struct("<IIQQQQQQQQQ")
BF16 = 1
SG, SU, SD = 32, 33, 34


def align_up(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def f32_to_bf16_bytes(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32, order="C")
    u = x.view(np.uint32)
    # Match ordinary truncating FP32->BF16 byte construction used by repo fixtures.
    return (u >> np.uint32(16)).astype("<u2").tobytes(order="C")


def bf16_roundtrip(x: np.ndarray) -> np.ndarray:
    raw = np.frombuffer(f32_to_bf16_bytes(x), dtype="<u2").astype(np.uint32)
    return (raw << np.uint32(16)).view(np.float32).reshape(x.shape)


def make_trunk(root: pathlib.Path, layers=(1, 2), h=32, inter=48):
    rng = np.random.default_rng(260901)
    recs = []
    originals = {}
    binp = root / "trunk.bin"
    with binp.open("wb") as f:
        for layer in layers:
            mats = {
                SG: bf16_roundtrip(rng.normal(0.0, 0.12, size=(inter, h)).astype(np.float32)),
                SU: bf16_roundtrip(rng.normal(0.0, 0.11, size=(inter, h)).astype(np.float32)),
                SD: bf16_roundtrip(rng.normal(0.0, 0.10, size=(h, inter)).astype(np.float32)),
            }
            originals[layer] = mats
            for kind in (SG, SU, SD):
                w = mats[kind]
                raw = f32_to_bf16_bytes(w)
                at = align_up(f.tell())
                if at > f.tell():
                    f.write(b"\0" * (at - f.tell()))
                payload = len(raw)
                readb = align_up(payload)
                f.write(raw)
                f.write(b"\0" * (readb - payload))
                recs.append((
                    layer, kind, BF16, 2, w.shape[0], w.shape[1], 0, 0,
                    at, readb, payload,
                ))
        data_bytes = f.tell()
    blob = bytearray(THDR.pack(b"KVLTRNK1", 1, ALIGN, len(recs), 0, THDR.size, data_bytes))
    for r in recs:
        blob.extend(TREC.pack(*r))
    (root / "trunk.idx").write_bytes(blob)
    return originals


def parse_q8_matrix(blob: bytes, off: int, nbytes: int, out: int, inc: int) -> np.ndarray:
    scale_bytes = out * 4
    assert nbytes == scale_bytes + out * inc
    scales = np.frombuffer(blob, dtype="<f4", count=out, offset=off).copy()
    q = np.frombuffer(blob, dtype=np.int8, count=out * inc,
                      offset=off + scale_bytes).astype(np.float32).reshape(out, inc)
    return q * scales[:, None]


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    packer = here / "pack_shared_q8_from_trunk.py"
    with tempfile.TemporaryDirectory(prefix="kvl-shared-q8-") as td:
        td = pathlib.Path(td)
        packed = td / "packed"
        side = td / "side"
        packed.mkdir()
        originals = make_trunk(packed)
        p = subprocess.run([sys.executable, str(packer), str(packed), str(side)],
                           text=True, capture_output=True)
        print(p.stdout, end="")
        print(p.stderr, end="", file=sys.stderr)
        if p.returncode:
            raise SystemExit(p.returncode)

        idx = (side / "shared_q8.idx").read_bytes()
        h = EHDR.unpack_from(idx, 0)
        assert h[0] == b"KVLXPRT1" and h[1] == 1 and h[2] == ALIGN
        assert h[5] == len(originals) and h[6] == 3 and h[7] == EHDR.size
        data = (side / "shared_q8.bin").read_bytes()
        assert len(data) == h[8]
        recs = {}
        off = h[7]
        for _ in range(h[5]):
            r = EREC.unpack_from(idx, off)
            off += EREC.size
            recs[r[0]] = r
            assert r[1] == 0
            assert r[2] % ALIGN == 0 and r[3] % ALIGN == 0
            assert r[4] <= r[3]

        rng = np.random.default_rng(90261)
        worst_rel = 0.0
        for layer, mats in originals.items():
            r = recs[layer]
            base = r[2]
            gate = parse_q8_matrix(data, base + r[5], r[6], *mats[SG].shape)
            up = parse_q8_matrix(data, base + r[7], r[8], *mats[SU].shape)
            down = parse_q8_matrix(data, base + r[9], r[10], *mats[SD].shape)
            for _ in range(16):
                x = rng.normal(0.0, 0.4, size=(mats[SG].shape[1],)).astype(np.float32)
                ref = mats[SD] @ (silu(mats[SG] @ x) * (mats[SU] @ x))
                got = down @ (silu(gate @ x) * (up @ x))
                rms = float(np.sqrt(np.mean((got - ref) ** 2)))
                sig = float(np.sqrt(np.mean(ref ** 2))) + 1e-30
                worst_rel = max(worst_rel, rms / sig)

        print(f"shared_q8_worst_rel_rms={worst_rel:.6f}")
        if worst_rel >= 0.035:
            raise SystemExit("shared Q8 synthetic MLP error exceeded 3.5%")
        print("SHARED_Q8_SIDECAR_PACK_PASS")


if __name__ == "__main__":
    main()
