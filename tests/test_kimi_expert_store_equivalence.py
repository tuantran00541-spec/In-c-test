#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_kimi_expert_store_equivalence",
    ROOT / "tools" / "run_kimi_expert_store_equivalence.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)

with tempfile.TemporaryDirectory() as td_raw:
    td = pathlib.Path(td_raw)
    mock = td / "mock_runtime.py"
    mock.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "is_sparse = any(pathlib.Path(x).name == 'sparse.idx' for x in sys.argv)\n"
        "if is_sparse and os.environ.get('KVL_MOE_MASK'):\n"
        "    raise SystemExit('sparse auto-mask run unexpectedly received KVL_MOE_MASK')\n"
        "if not is_sparse and not os.environ.get('KVL_MOE_MASK'):\n"
        "    raise SystemExit('full run missing explicit KVL_MOE_MASK')\n"
        "print('TOKEN 101')\n"
        "print('TOKEN 202')\n"
        "pathlib.Path(os.environ['KVL_MOE_TRACE']).write_bytes(b'route-same\\n')\n"
        "pathlib.Path(os.environ['KVL_LOGITS_DUMP']).write_bytes(b'logits-same\\x00\\x01')\n",
        encoding="utf-8",
    )
    mock.chmod(0o755)

    files = {}
    for name in (
        "trunk.bin", "trunk.idx", "full.bin", "full.idx",
        "sparse.bin", "sparse.idx", "prompt.ids", "mask.txt",
    ):
        p = td / name
        p.write_bytes((name + "\n").encode())
        files[name] = p
    (td / "sparse.mask").write_text("# KVL_MOE_MASK_V1\n1 1\n", encoding="utf-8")

    work = td / "work"
    cmd = [
        sys.executable, str(ROOT / "tools" / "run_kimi_expert_store_equivalence.py"),
        "--binary", str(mock),
        "--trunk-bin", str(files["trunk.bin"]),
        "--trunk-idx", str(files["trunk.idx"]),
        "--full-experts-bin", str(files["full.bin"]),
        "--full-experts-idx", str(files["full.idx"]),
        "--sparse-experts-bin", str(files["sparse.bin"]),
        "--sparse-experts-idx", str(files["sparse.idx"]),
        "--prompt-ids", str(files["prompt.ids"]),
        "--mask", str(files["mask.txt"]),
        "--sparse-auto-mask",
        "--work-dir", str(work),
        "--cache-mib", "1", "--max-new", "2", "--temperature", "0", "--seed", "1",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "KIMI_EXPERT_STORE_RUNTIME_EQ_PASS" in proc.stdout
    assert "sparse_mask=sparse-sidecar-auto" in proc.stdout
    report = work / "expert-store-equivalence.json"
    assert report.is_file()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["sparse_auto_mask"] is True
    assert data["full"]["mask_mode"] == "explicit"
    assert data["sparse"]["mask_mode"] == "sparse-sidecar-auto"
    assert data["byte_exact"] is True

    # Unit-level mismatch localization: differing payloads must expose byte offset 3.
    a = td / "a.bin"
    b = td / "b.bin"
    a.write_bytes(b"abcXdef")
    b.write_bytes(b"abcYdef")
    assert mod.first_mismatch(a, b) == 3
    b.write_bytes(a.read_bytes())
    assert mod.first_mismatch(a, b) is None

print("KIMI_EXPERT_STORE_EQUIVALENCE_TEST_PASS")
