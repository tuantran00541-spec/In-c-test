#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('knife', ROOT/'tools'/'qwen38_basic_knife.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# Q6 RTN should preserve exactly representable values and keep shape.
w=torch.tensor([[0.0,1.0,-1.0,0.5,-0.5,0.0,1.0,-1.0]],dtype=torch.float32)
q=m.qdq_rows(w,6,4)
assert q.shape==w.shape
assert torch.isfinite(q).all()

# Structured 896-channel cut is exact and group aligned.
keep=m.INTERMEDIATE-896
assert keep==16512 and keep%128==0
full=3*m.HIDDEN*m.INTERMEDIATE
kept=3*m.HIDDEN*keep
assert full==267386880
assert full-kept==13762560
p=m.projected_qbytes(m.HIDDEN,keep,6,128)
assert p['total_bytes']==194181120
assert p['total_bytes'] < kept*2 < full*2
print('QWEN38_BASIC_KNIFE_UNIT_PASS')
