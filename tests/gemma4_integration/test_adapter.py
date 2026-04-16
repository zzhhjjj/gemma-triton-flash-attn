"""Unit test for the HF attention adapter — no model download.

Directly invokes `triton_gqa_attention` (the function transformers would call)
across every combination we care about:

  - **Attention pattern**: full causal AND sliding window (SWA)
  - **GQA ratio**:         1:1 (MHA), 2:1, 4:1, 8:1
  - **HEAD_DIM**:          256 (Gemma4 sliding) and 512 (Gemma4 full)
  - **Sequence length**:   short (N≤slide) and long (N>slide)

Compares output against a manual SDPA reference that encodes the same mask.
This is what guarantees that swapping `config._attn_implementation` to
`"triton_gqa"` preserves semantics for both **full** and **sliding** layers
in a model like Gemma-4-E2B (29 sliding + 6 full).

Run:
    source /opt/tiger/flash_gemma/bin/activate
    python tests/gemma4_integration/test_adapter.py
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from gemma_triton_flash_attn import register_triton_attention, triton_gqa_attention


# =====================================================================
# Reference: SDPA with explicit causal + optional sliding window mask
# =====================================================================

def sdpa_reference(q, k, v, *, is_causal, slide_size):
    """Ground-truth attention computed via eager matmul + softmax.

    q: (B, H_Q,  N, D)
    k: (B, H_KV, N, D)
    v: (B, H_KV, N, D)

    Supports GQA by repeat_interleave; supports causal and sliding-window masks.
    Returns (B, N, H_Q, D) to match transformers' post-attention shape.
    """
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    if H_Q != H_KV:
        ratio = H_Q // H_KV
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)

    scale = D ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale                # (B, H_Q, N, N)

    idx = torch.arange(N, device=q.device)
    if is_causal:
        mask = idx[:, None] >= idx[None, :]                    # lower triangular
    else:
        mask = torch.ones(N, N, dtype=torch.bool, device=q.device)
    if slide_size > 0:
        window = (idx[:, None] - idx[None, :]) < slide_size    # q - k < S
        mask = mask & window

    scores = scores.masked_fill(~mask, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = probs @ v                                            # (B, H_Q, N, D)
    return out.transpose(1, 2).contiguous()                    # (B, N, H_Q, D)


# =====================================================================
# Adapter driver: simulate transformers' call site
# =====================================================================

@dataclass
class Case:
    label: str
    B: int
    H_Q: int
    H_KV: int
    N: int
    D: int
    slide_size: int            # 0 means full causal; >0 means SWA
    is_causal: bool = True


def run_case(case: Case, *, dtype=torch.bfloat16) -> tuple[float, float]:
    """Run both adapter and reference, return (cosine_sim, max_rel_err)."""
    torch.manual_seed(0)
    q = torch.randn(case.B, case.H_Q, case.N, case.D, dtype=dtype, device="cuda")
    k = torch.randn(case.B, case.H_KV, case.N, case.D, dtype=dtype, device="cuda")
    v = torch.randn(case.B, case.H_KV, case.N, case.D, dtype=dtype, device="cuda")

    # Fake `module` — the adapter reads `.head_dim` and `.is_causal` off it.
    module = SimpleNamespace(head_dim=case.D, is_causal=case.is_causal)

    # transformers passes sliding_window=None for full layers, positive int for SWA.
    sliding_window = case.slide_size if case.slide_size > 0 else None

    tri_out, _ = triton_gqa_attention(
        module, q, k, v, attention_mask=None,
        dropout=0.0, scaling=None, softcap=None,
        sliding_window=sliding_window,
    )
    ref_out = sdpa_reference(q, k, v, is_causal=case.is_causal, slide_size=case.slide_size)

    tri_f = tri_out.float().flatten()
    ref_f = ref_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(tri_f, ref_f, dim=0).item()
    rel = ((tri_f - ref_f).norm() / ref_f.norm()).item()
    return cos, rel


def main():
    register_triton_attention()

    # Design matrix. Sizes small enough to run in seconds, large enough for
    # Triton's block sizes to kick in.
    N_short, N_long, slide = 512, 2048, 512
    cases: list[Case] = []

    # --- GQA ratio sweep at D=256 (Gemma4 sliding config) ---
    for H_Q, H_KV in [(8, 8), (8, 4), (8, 2), (8, 1), (16, 2)]:
        # Full causal
        cases.append(Case(f"causal D=256 {H_Q}:{H_KV} N={N_short}",
                          B=1, H_Q=H_Q, H_KV=H_KV, N=N_short, D=256, slide_size=0))
        cases.append(Case(f"causal D=256 {H_Q}:{H_KV} N={N_long}",
                          B=1, H_Q=H_Q, H_KV=H_KV, N=N_long, D=256, slide_size=0))
        # SWA: N <= slide (window covers everything — should match causal)
        cases.append(Case(f"SWA    D=256 {H_Q}:{H_KV} N={N_short} slide={slide}",
                          B=1, H_Q=H_Q, H_KV=H_KV, N=N_short, D=256, slide_size=slide))
        # SWA: N > slide (actual window truncation)
        cases.append(Case(f"SWA    D=256 {H_Q}:{H_KV} N={N_long} slide={slide}",
                          B=1, H_Q=H_Q, H_KV=H_KV, N=N_long, D=256, slide_size=slide))

    # --- HEAD_DIM=512 (Gemma4 full config): GQA 8:1, full causal only ---
    for N in (N_short, N_long):
        cases.append(Case(f"causal D=512 32:4   N={N}",
                          B=1, H_Q=32, H_KV=4, N=N, D=512, slide_size=0))

    # --- batch size > 1 ---
    cases.append(Case("causal D=256 8:1   N=1024 B=2",
                      B=2, H_Q=8, H_KV=1, N=1024, D=256, slide_size=0))
    cases.append(Case("SWA    D=256 8:1   N=2048 B=2 slide=512",
                      B=2, H_Q=8, H_KV=1, N=2048, D=256, slide_size=512))

    # Run all cases.
    print(f"{'Case':<52} {'cos sim':>10} {'rel err':>10}  status")
    print("-" * 88)
    n_fail = 0
    for case in cases:
        try:
            cos, rel = run_case(case)
        except Exception as e:
            print(f"{case.label:<52} ERROR: {e!r}")
            n_fail += 1
            continue
        ok = cos > 0.999 and rel < 0.05
        status = "OK" if ok else "FAIL"
        print(f"{case.label:<52} {cos:>10.6f} {rel:>10.2e}  {status}")
        if not ok:
            n_fail += 1

    print("-" * 88)
    print(f"{len(cases) - n_fail}/{len(cases)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
