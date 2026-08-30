#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "tests" else Path(__file__).resolve().parent
TOOLS = ROOT / "tools" if (ROOT / "tools").is_dir() else ROOT
sys.path.insert(0, str(TOOLS))
TOOL = TOOLS / "run_kimi_pruning_sentinel_screen.py"
spec = importlib.util.spec_from_file_location("sentinel", TOOL)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def write_mask(path: Path, rows):
    path.write_text("# KVL_MOE_MASK_V1\n" + "".join(f"{l} {e}\n" for l, e in rows), encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root / "a.mask"; b = root / "b.mask"; bad = root / "bad.mask"
        write_mask(a, [(1, 1), (2, 2)])
        write_mask(b, [(1, 1), (2, 2), (3, 3)])
        write_mask(bad, [(1, 1), (4, 4), (5, 5), (6, 6)])
        rows = mod.validate_candidates([f"a={a}", f"b={b}"])
        assert [r["disabled_count"] for r in rows] == [2, 3]
        try:
            mod.validate_candidates([f"a={a}", f"bad={bad}"])
        except ValueError as e:
            assert "nested" in str(e)
        else:
            raise AssertionError("non-nested masks must fail")
        selected = mod.select_strongest([
            {"name":"a", "disabled_count":54, "token_exact":True},
            {"name":"b", "disabled_count":56, "token_exact":False},
            {"name":"c", "disabled_count":58, "token_exact":True},
        ])
        assert selected["name"] == "c"
        assert mod.select_strongest([{"disabled_count":54, "token_exact":False}]) is None
    print("KIMI_SENTINEL_TEST_PASS")


if __name__ == "__main__":
    main()
