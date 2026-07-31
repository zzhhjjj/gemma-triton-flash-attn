"""Edge cases for the varlen kernel.

Covers:
  1. B=1 single sample → must match batched kernel.
  2. Single-token samples (seqlen=1).
  3. Heavily skewed distribution [1, 1, 1, N]  (OOR program early-return).
  4. window_size > max_seqlen (degenerates to pure causal).
  5. Non-contiguous input (sliced from a larger allocation).

Runs on the `varlen-fa` conda env (triton 3.2, H200). Keeps to HEAD_DIM=128
to avoid the pre-existing shmem OOM at D=256/512 on this env.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
import triton

from flash_attn import (
    flash_attn_gqa_varlen,
    flash_attn_gqa_train,
    attention_gqa_varlen_ref,
    pack_batched_to_varlen,
    unpack_varlen_to_batched,
)


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _cu(seqlens, device):
    B = seqlens.numel()
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = seqlens.to(torch.int32).cumsum(0).to(device)
    return cu


def test_single_sample_matches_batched() -> tuple[bool, str]:
    """B=1 single sample: varlen must match the batched kernel exactly."""
    torch.manual_seed(0)
    H_Q, H_KV, N, D = 8, 2, 1024, 128
    qb = torch.randn(1, H_Q, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    kb = torch.randn(1, H_KV, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    vb = torch.randn(1, H_KV, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    do = torch.randn(1, H_Q, N, D, dtype=torch.float16, device="cuda")

    out_b = flash_attn_gqa_train(qb, kb, vb, causal=True, slide_size=0)
    out_b.backward(do)
    dqb, dkb, dvb = qb.grad, kb.grad, vb.grad

    seqlens = torch.tensor([N], dtype=torch.int64)
    qp_data, kp_data, vp_data, cu = pack_batched_to_varlen(
        qb.detach(), kb.detach(), vb.detach(), seqlens)
    do_packed, _, _, _ = pack_batched_to_varlen(do, do[:, :H_KV], do[:, :H_KV], seqlens)

    qp = qp_data.requires_grad_(True)
    kp = kp_data.requires_grad_(True)
    vp = vp_data.requires_grad_(True)
    out_v = flash_attn_gqa_varlen(qp, kp, vp, cu, cu, N, N, causal=True, window_size=0)
    out_v.backward(do_packed)

    out_v_bhnd = unpack_varlen_to_batched(out_v, cu, N, H_Q)
    dq_v = unpack_varlen_to_batched(qp.grad, cu, N, H_Q)
    dk_v = unpack_varlen_to_batched(kp.grad, cu, N, H_KV)
    dv_v = unpack_varlen_to_batched(vp.grad, cu, N, H_KV)

    out_cos = _cos(out_v_bhnd, out_b)
    dq_cos = _cos(dq_v, dqb)
    dk_cos = _cos(dk_v, dkb)
    dv_cos = _cos(dv_v, dvb)
    ok = out_cos > 0.99999 and dq_cos > 0.99999 and dk_cos > 0.99999 and dv_cos > 0.99999
    msg = f"out_cos={out_cos:.9f} dq={dq_cos:.6f} dk={dk_cos:.6f} dv={dv_cos:.6f}"
    return ok, msg


def test_single_token_sample() -> tuple[bool, str]:
    """cu_seqlens = [0, 1]: a sample of length 1 must produce finite output."""
    torch.manual_seed(1)
    H_Q, H_KV, D = 4, 2, 128
    q = torch.randn(1, H_Q, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(1, H_KV, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(1, H_KV, D, dtype=torch.float16, device="cuda", requires_grad=True)
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    out = flash_attn_gqa_varlen(q, k, v, cu, cu, 1, 1, causal=True, window_size=0)
    assert out.shape == q.shape
    assert out.isfinite().all().item(), "seqlen=1 produced non-finite output"
    # Ground truth: attention over a single token = v (softmax-weighted by
    # itself, causal mask trivially passes).
    # With GQA expansion, v is repeated to match Q's heads.
    ref = v.repeat_interleave(H_Q // H_KV, dim=1)  # (1, H_Q, D)
    diff = (out.float() - ref.float()).abs().max().item()
    out.sum().backward()
    assert q.grad is not None and q.grad.isfinite().all().item()
    ok = diff < 1e-3
    return ok, f"max_abs={diff:.2e} out_finite={out.isfinite().all().item()}"


def test_skewed_distribution_with_length_ones() -> tuple[bool, str]:
    """Skewed [1, 1, 1, 128]: tiny samples must early-return cleanly for all
    q_block > 0 of the dominant sample's run."""
    torch.manual_seed(2)
    H_Q, H_KV, D = 8, 2, 128
    seqlens = torch.tensor([1, 1, 1, 128], dtype=torch.int64)
    total = int(seqlens.sum().item())
    max_len = int(seqlens.max().item())
    q = torch.randn(total, H_Q, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(total, H_KV, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(total, H_KV, D, dtype=torch.float16, device="cuda", requires_grad=True)
    cu = _cu(seqlens, "cuda")

    out = flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                                causal=True, window_size=0)
    ref = attention_gqa_varlen_ref(q.detach(), k.detach(), v.detach(), cu, cu,
                                   max_len, max_len, causal=True, window_size=0)
    diff = (out.float() - ref.float()).abs().max().item()
    cos = _cos(out, ref)

    do = torch.randn_like(out)
    out.backward(do)
    # Length-1 samples: dQ for that row must equal dO * (1/softmax_denom) * 1,
    # but any finite value is acceptable.
    ok_shape = (q.grad.shape == q.shape)
    ok_finite = q.grad.isfinite().all().item() and k.grad.isfinite().all().item() and v.grad.isfinite().all().item()
    ok = cos > 0.999 and diff < 1e-2 and ok_shape and ok_finite
    return ok, f"max_abs={diff:.2e} cos={cos:.6f} finite_grads={ok_finite}"


def test_window_size_exceeds_seqlen() -> tuple[bool, str]:
    """window_size > max_seqlen should degenerate to pure causal (kernel path
    normalizes slide_size to 0 internally)."""
    torch.manual_seed(3)
    H_Q, H_KV, D = 8, 2, 128
    seqlens = torch.tensor([64, 128], dtype=torch.int64)
    total = int(seqlens.sum().item())
    max_len = int(seqlens.max().item())
    q = torch.randn(total, H_Q, D, dtype=torch.float16, device="cuda")
    k = torch.randn(total, H_KV, D, dtype=torch.float16, device="cuda")
    v = torch.randn(total, H_KV, D, dtype=torch.float16, device="cuda")
    cu = _cu(seqlens, "cuda")

    out_causal = flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                                       causal=True, window_size=0)
    # Window larger than every sample: should match plain causal.
    out_big_win = flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                                        causal=True, window_size=1024)
    diff = (out_causal.float() - out_big_win.float()).abs().max().item()
    cos = _cos(out_causal, out_big_win)
    ok = cos > 0.99999 and diff < 1e-4
    return ok, f"max_abs={diff:.2e} cos={cos:.9f}"


def test_noncontiguous_input() -> tuple[bool, str]:
    """Non-contiguous input sliced from a larger allocation: must still work
    because the kernel carries per-tensor strides."""
    torch.manual_seed(4)
    H_Q, H_KV, D = 8, 2, 128
    seqlens = torch.tensor([64, 96, 48], dtype=torch.int64)
    total = int(seqlens.sum().item())
    max_len = int(seqlens.max().item())

    # Allocate larger buffers, then slice along the token axis.
    pad = 32
    big_q = torch.randn(total + pad, H_Q, D, dtype=torch.float16, device="cuda")
    big_k = torch.randn(total + pad, H_KV, D, dtype=torch.float16, device="cuda")
    big_v = torch.randn(total + pad, H_KV, D, dtype=torch.float16, device="cuda")
    q = big_q[pad:pad + total]
    k = big_k[pad:pad + total]
    v = big_v[pad:pad + total]
    # stride(0) is unchanged (still H_Q*D for q, H_KV*D for k/v); stride(-1)==1.
    cu = _cu(seqlens, "cuda")

    out = flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                                causal=True, window_size=0)
    ref = attention_gqa_varlen_ref(q, k, v, cu, cu, max_len, max_len,
                                   causal=True, window_size=0)
    diff = (out.float() - ref.float()).abs().max().item()
    cos = _cos(out, ref)
    ok = cos > 0.999 and diff < 5e-2
    return ok, (f"max_abs={diff:.2e} cos={cos:.6f} "
                f"(q.stride={q.stride()}, contig={q.is_contiguous()})")


def main() -> int:
    tests = [
        ("single_sample_matches_batched", test_single_sample_matches_batched),
        ("single_token_sample",           test_single_token_sample),
        ("skewed_distribution",           test_skewed_distribution_with_length_ones),
        ("window_exceeds_seqlen",         test_window_size_exceeds_seqlen),
        ("noncontiguous_input",           test_noncontiguous_input),
    ]
    print(f"{'Test':<35}{'Status':>8}  Details")
    print("-" * 92)
    all_ok = True
    for name, fn in tests:
        try:
            ok, msg = fn()
        except triton.runtime.errors.OutOfResources as e:
            ok, msg = False, f"SHMEM OOM on this env: {e}"
        except AssertionError as e:
            ok, msg = False, f"ASSERT: {e}"
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"{name:<35}{status:>8}  {msg}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
