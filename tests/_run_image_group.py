"""Manual driver for tests/test_image_group_mask.py (no pytest available)."""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub pytest so the test module imports without the real package.
import types
_pytest = types.ModuleType("pytest")
class _Mark:
    def parametrize(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco
_pytest.mark = _Mark()
sys.modules["pytest"] = _pytest

import test_image_group_mask as M

PASS = 0
FAIL = 0
FAILS = []


def run(name, fn, *args):
    global PASS, FAIL
    label = f"{name}{args if args else ''}"
    try:
        fn(*args)
        PASS += 1
        print(f"  PASS  {label}")
    except Exception as e:
        FAIL += 1
        FAILS.append((label, traceback.format_exc()))
        print(f"  FAIL  {label}: {e}")


print("=== fwd image-group OR-mask ===")
for shape in M.SHAPES:
    run("test_fwd_image_group_or_mask", M.test_fwd_image_group_or_mask, *shape)

print("\n=== bwd image-group OR-mask ===")
for shape in M.BWD_SHAPES:
    run("test_bwd_image_group_or_mask", M.test_bwd_image_group_or_mask, *shape)

print("\n=== regression: no-image path unchanged ===")
run("test_no_image_path_unchanged", M.test_no_image_path_unchanged)

print(f"\n--- {PASS} passed / {FAIL} failed ---")
if FAILS:
    for name, tb in FAILS:
        print(f"\n>>> {name}\n{tb}")
sys.exit(1 if FAIL else 0)
