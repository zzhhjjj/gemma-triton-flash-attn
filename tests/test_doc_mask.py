"""Packed-document isolation (doc masking) correctness tests.

Reference: SDPA in float32 with an explicit block-diagonal causal (+ optional
sliding window) bool mask built from the same packed position_ids.

Checks (per repo convention: fp32 cos sim, threshold 0.9999):
  1. fwd output + dq/dk/dv grads vs reference, for
     {D=512 full causal, D=256 slide=1024} x {GQA 2:1, 8:1} x {B=1, B=2}
  2. degenerate single-document input == plain causal path (no doc mask)
  3. doc_bounds_from_position_ids unit check against a python loop

Run: python tests/test_doc_mask.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_attn import doc_bounds_from_position_ids, flash_attn_gqa_train  # noqa: E402


def make_packed_position_ids(N, doc_lens, device):
    pos = torch.empty(N, dtype=torch.int64, device=device)
    i = 0
    for L in doc_lens:
        L = min(L, N - i)
        pos[i:i + L] = torch.arange(L, device=device)
        i += L
        if i >= N:
            break
    assert i == N, "doc_lens must cover N"
    return pos


def ref_attention(q, k, v, pos, slide_size):
    """float32 SDPA with explicit block-diag causal (+window) mask."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    if H_Q != H_KV:
        r = H_Q // H_KV
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
    idx = torch.arange(N, device=q.device)
    doc_lo = idx.unsqueeze(0) - pos  # (B, N)
    allowed = (idx[None, None, :] <= idx[None, :, None])  # causal (1, N, N)
    allowed = allowed & (idx[None, None, :] >= doc_lo[:, :, None])  # same doc
    if slide_size > 0:
        allowed = allowed & (idx[None, :, None] - idx[None, None, :] < slide_size)
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=allowed.unsqueeze(1))


def cos(a, b):
    return F.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()


def run_case(B, H_Q, H_KV, N, D, slide, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    doc_lens_all = []
    pos_rows = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(B):
        lens, tot = [], 0
        while tot < N:
            L = int(torch.randint(60, 1600, (1,), generator=g))
            L = min(L, N - tot)
            lens.append(L)
            tot += L
        doc_lens_all.append(lens)
        pos_rows.append(make_packed_position_ids(N, lens, device))
    pos = torch.stack(pos_rows)  # (B, N)

    q = torch.randn(B, H_Q, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(B, H_Q, N, D, device=device, dtype=torch.bfloat16)

    doc_lo, doc_hi = doc_bounds_from_position_ids(pos)
    out = flash_attn_gqa_train(q, k, v, causal=True, slide_size=slide,
                               doc_lo=doc_lo, doc_hi_excl=doc_hi)
    out.backward(do)
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = None

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    ref = ref_attention(q2, k2, v2, pos, slide)
    ref.backward(do.float())

    sims = {
        "out": cos(out, ref),
        "dq": cos(dq, q2.grad),
        "dk": cos(dk, k2.grad),
        "dv": cos(dv, v2.grad),
    }
    ok = all(s >= 0.9999 for s in sims.values()) and not out.isnan().any()
    ndocs = sum(len(x) for x in doc_lens_all)
    print(f"B={B} H_Q={H_Q} H_KV={H_KV} N={N} D={D} slide={slide} docs={ndocs}: "
          + " ".join(f"{k_}={v_:.6f}" for k_, v_ in sims.items())
          + ("  PASS" if ok else "  FAIL"))
    return ok


def test_single_doc_equals_plain_causal():
    """pos = arange (one doc) must reproduce the no-doc-mask path."""
    torch.manual_seed(1)
    B, H_Q, H_KV, N, D = 1, 16, 8, 2048, 512
    q = torch.randn(B, H_Q, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_KV, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_KV, N, D, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(N, device="cuda").unsqueeze(0)
    doc_lo, doc_hi = doc_bounds_from_position_ids(pos)
    a = flash_attn_gqa_train(q, k, v, causal=True, doc_lo=doc_lo, doc_hi_excl=doc_hi)
    b = flash_attn_gqa_train(q, k, v, causal=True)
    sim = cos(a, b)
    ok = sim >= 0.999999
    print(f"single-doc == plain causal: cos={sim:.8f} {'PASS' if ok else 'FAIL'}")
    return ok


def test_bounds_helper():
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]], device="cuda")
    lo, hi = doc_bounds_from_position_ids(pos)
    exp_lo = [0, 0, 0, 3, 3, 5, 5, 5, 5]
    exp_hi = [3, 3, 3, 5, 5, 9, 9, 9, 9]
    ok = lo[0].tolist() == exp_lo and hi[0].tolist() == exp_hi
    print(f"doc_bounds helper: {'PASS' if ok else 'FAIL'} lo={lo[0].tolist()} hi={hi[0].tolist()}")
    return ok


if __name__ == "__main__":
    results = [test_bounds_helper(), test_single_doc_equals_plain_causal()]
    # Gemma-4-31B-ish config (H_Q=16, H_KV=8): full layers D=512, sliding D=256
    for B in (1, 2):
        results.append(run_case(B, 16, 8, 4096, 512, 0))
        results.append(run_case(B, 16, 8, 4096, 256, 1024))
    # Gemma-4-E2B config (GQA 8:1)
    results.append(run_case(1, 8, 1, 4096, 512, 0))
    results.append(run_case(1, 8, 1, 4096, 256, 1024))
    print("=" * 60)
    print("ALL PASS" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)
