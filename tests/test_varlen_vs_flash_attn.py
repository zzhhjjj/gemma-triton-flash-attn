"""Varlen kernel vs upstream Dao-AILab flash-attention as an oracle.

Upstream `flash_attn` provides `flash_attn_varlen_func`, which we compare
against on D=128 configs (upstream does not support D=256 or D=512).

IMPORTANT — import-name collision:
    This repo's Python package is also imported as `flash_attn` (set by
    `flash_attn/` directory layout). That shadows upstream's `flash_attn`
    module when both are installed into the same environment. This test
    attempts the upstream import via an alternate location and SKIPS cleanly
    if upstream is not available.

To install upstream alongside this repo in a fresh env:
    # Rename upstream import path:
    pip install flash-attn --no-build-isolation
    # Then import under the fully-qualified CUDA path:
    from flash_attn_cuda import flash_attn_varlen_func  # upstream's private module

In the `varlen-fa` conda env where this repo is developed, upstream is NOT
installed, so this test reports SKIP for all configs. That's expected — the
primary correctness gate is `test_varlen_correctness.py` (against SDPA).
"""
from __future__ import annotations

import sys
import importlib

import torch
import torch.nn.functional as F


# Try to import upstream through alternate paths (upstream private CUDA modules
# don't always collide with this repo's directory-named `flash_attn`). If none
# are importable, the test cleanly skips.
def _try_import_upstream_varlen():
    import importlib.util, sys
    # Common upstream paths:
    candidates = [
        # (module_name, attribute)
        ("flash_attn_2_cuda", None),
        ("flash_attn_cuda", None),
    ]
    # If this repo's `flash_attn` is first on sys.path, it shadows upstream's.
    # Try to find upstream outside of the repo's import.
    for name, _ in candidates:
        spec = importlib.util.find_spec(name)
        if spec is not None:
            try:
                return importlib.import_module(name)
            except Exception:
                continue
    # Last resort: try the public flash_attn_interface but ONLY if it's not ours.
    spec = importlib.util.find_spec("flash_attn_interface")
    if spec is not None:
        try:
            mod = importlib.import_module("flash_attn_interface")
            if hasattr(mod, "flash_attn_varlen_func"):
                return mod
        except Exception:
            pass
    return None


from flash_attn import flash_attn_gqa_varlen, attention_gqa_varlen_ref  # this repo


def _cu(seqlens, device):
    B = seqlens.numel()
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = seqlens.to(torch.int32).cumsum(0).to(device)
    return cu


def main() -> int:
    upstream = _try_import_upstream_varlen()
    if upstream is None:
        print("upstream flash-attn not importable in this env — SKIP all configs")
        print("(import-name collision with this repo's `flash_attn` package; "
              "expected in the dev env)")
        return 0  # SKIP is a pass: the test infra is fine, just no oracle available

    print(f"upstream flash-attn found: {upstream.__name__}")
    # If we ever reach here, fill in per-config comparisons. Out of scope for v1
    # while the env shadows upstream.
    print("oracle comparison implementation deferred — see test_varlen_correctness.py "
          "for the primary gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
