"""Fused dQ+dK+dV backward kernel: correctness + benchmark vs split design.

Baseline: existing split design (delta + dQ + dKV + reduce).
New:      fused kernel (dQ + atomic dK/dV in one pass).

Usage:
    source /opt/tiger/flash_gemma/bin/activate
    python tests/test_fused_backward.py
"""
from __future__ import annotations

import math
import sys
from functools import partial

import torch
import triton

from gemma_triton_flash_attn.attention import (
    _flash_attn_gqa_bwd_dq_kernel,
    _flash_attn_gqa_bwd_dkv_kernel,
    _flash_attn_gqa_bwd_fused_kernel,
    _delta_kernel,
    _flash_attn_gqa_kernel,
    attention_gqa_ref,
    attention_swa_ref,
)
from gemma_triton_flash_attn.utils import benchmark_fn


def run_forward(q, k, v, *, causal, slide_size, BQ=None, BKV=None, num_warps=8):
    """Fwd + return (output, lse)."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    if slide_size > 0 and slide_size >= N:
        slide_size = 0
    # Match wrapper's D-specific defaults
    if BQ is None:
        BQ = 64 if D >= 512 else 128
    if BKV is None:
        BKV = 32 if D >= 512 else 64
    BQ = min(BQ, triton.next_power_of_2(N))
    BKV = min(BKV, triton.next_power_of_2(N))
    output = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(N, BQ), B * H_Q)
    _flash_attn_gqa_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ, BLOCK_KV=BKV, BLOCK_D=D,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=True,
        num_warps=num_warps, num_stages=2,
    )
    return output, lse, slide_size  # return normalized slide_size


def run_delta(do, o):
    """Wrapped delta kernel."""
    B, H, N, D = do.shape
    delta = torch.empty(B, H, N, dtype=torch.float32, device=do.device)
    _delta_kernel[(N, B * H)](
        do, o, delta,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_HEADS=H, SEQ_LEN=N, HEAD_DIM=D, num_warps=4, num_stages=2,
    )
    return delta


def run_backward_split(q, k, v, do, o, lse, *, causal, slide_size):
    """Original split design: delta + dQ kernel + dKV kernel + reduce."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    GQA_RATIO = H_Q // H_KV
    scale = 1.0 / math.sqrt(D)

    delta = run_delta(do, o)
    dq = torch.empty_like(q)
    dk_expanded = torch.empty(B, H_Q, N, D, dtype=q.dtype, device=q.device)
    dv_expanded = torch.empty(B, H_Q, N, D, dtype=q.dtype, device=q.device)

    # dQ kernel config (D-specific)
    if D >= 512:
        BQ_dq, BKV_dq, nw_dq = 32, 64, 8
    else:
        BQ_dq, BKV_dq, nw_dq = 64, 64, 4
    BQ_dq = min(BQ_dq, triton.next_power_of_2(N))
    BKV_dq = min(BKV_dq, triton.next_power_of_2(N))

    _flash_attn_gqa_bwd_dq_kernel[(triton.cdiv(N, BQ_dq), B * H_Q)](
        q, k, v, do, dq, lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D, scale=scale,
        BLOCK_Q=BQ_dq, BLOCK_KV=BKV_dq,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
        num_warps=nw_dq, num_stages=2,
    )

    # dKV kernel config
    if D >= 512:
        BQ_dkv, BKV_dkv, nw_dkv = 16, 32, 8
    else:
        BQ_dkv, BKV_dkv, nw_dkv = 64, 32, 4
    BQ_dkv = min(BQ_dkv, triton.next_power_of_2(N))
    BKV_dkv = min(BKV_dkv, triton.next_power_of_2(N))

    _flash_attn_gqa_bwd_dkv_kernel[(triton.cdiv(N, BKV_dkv), B * H_Q)](
        q, k, v, do, dk_expanded, dv_expanded, lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk_expanded.stride(0), dk_expanded.stride(1), dk_expanded.stride(2), dk_expanded.stride(3),
        dv_expanded.stride(0), dv_expanded.stride(1), dv_expanded.stride(2), dv_expanded.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D, scale=scale,
        BLOCK_Q=BQ_dkv, BLOCK_KV=BKV_dkv,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
        ATOMIC_REDUCE=False,
        num_warps=nw_dkv, num_stages=2,
    )

    if GQA_RATIO == 2:
        dk_r = dk_expanded.view(B, H_KV, 2, N, D)
        dv_r = dv_expanded.view(B, H_KV, 2, N, D)
        dk = dk_r[:, :, 0] + dk_r[:, :, 1]
        dv = dv_r[:, :, 0] + dv_r[:, :, 1]
    else:
        dk = dk_expanded.view(B, H_KV, GQA_RATIO, N, D).sum(dim=2)
        dv = dv_expanded.view(B, H_KV, GQA_RATIO, N, D).sum(dim=2)

    return dq, dk, dv


def run_backward_fused(q, k, v, do, o, lse, *, causal, slide_size,
                       BQ=None, BKV=None, num_warps=None):
    """Fused design: single kernel does dQ + fp32 atomic dK/dV, cast at end."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    scale = 1.0 / math.sqrt(D)

    # Block config — shmem-constrained. Fused kernel holds Q, dO, K, V, dq_acc
    # simultaneously plus atomic scratch tiles. Smaller blocks than dQ-only.
    if BQ is None:
        BQ = 16 if D >= 512 else 32
    if BKV is None:
        BKV = 32
    if num_warps is None:
        num_warps = 8 if D >= 512 else 4
    BQ = min(BQ, triton.next_power_of_2(N))
    BKV = min(BKV, triton.next_power_of_2(N))

    delta = run_delta(do, o)
    dq = torch.empty_like(q)
    # fp32 scratch — atomic_add on fp16 loses precision with GQA_RATIO-way contention
    dk_f32 = torch.zeros(B, H_KV, N, D, dtype=torch.float32, device=q.device)
    dv_f32 = torch.zeros(B, H_KV, N, D, dtype=torch.float32, device=q.device)

    _flash_attn_gqa_bwd_fused_kernel[(triton.cdiv(N, BQ), B * H_Q)](
        q, k, v, do, dq, dk_f32, dv_f32, lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk_f32.stride(0), dk_f32.stride(1), dk_f32.stride(2), dk_f32.stride(3),
        dv_f32.stride(0), dv_f32.stride(1), dv_f32.stride(2), dv_f32.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D, scale=scale,
        BLOCK_Q=BQ, BLOCK_KV=BKV,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
        num_warps=num_warps, num_stages=2,
    )
    dk = dk_f32.to(k.dtype)
    dv = dv_f32.to(v.dtype)
    return dq, dk, dv


def make_tensors(B, H_Q, H_KV, N, D, dtype=torch.float16):
    torch.manual_seed(0)
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda")
    do = torch.randn(B, H_Q, N, D, dtype=dtype, device="cuda")
    return q, k, v, do


def test_correctness():
    print("=== Correctness: fused vs split backward ===")
    all_ok = True
    configs = [
        # (H_Q, H_KV, N, D, causal, slide)
        (8, 1, 512, 256, True, 0),
        (8, 1, 1024, 256, True, 0),
        (8, 1, 2048, 256, True, 512),  # SWA
        (32, 4, 1024, 512, True, 0),
        (32, 4, 2048, 512, True, 0),
    ]
    for H_Q, H_KV, N, D, causal, slide in configs:
        q, k, v, do = make_tensors(1, H_Q, H_KV, N, D)
        o, lse, slide_n = run_forward(q, k, v, causal=causal, slide_size=slide)

        dq_split, dk_split, dv_split = run_backward_split(
            q, k, v, do, o, lse, causal=causal, slide_size=slide_n)
        dq_fused, dk_fused, dv_fused = run_backward_fused(
            q, k, v, do, o, lse, causal=causal, slide_size=slide_n)

        dq_diff = (dq_split - dq_fused).abs().max().item()
        dk_diff = (dk_split - dk_fused).abs().max().item()
        dv_diff = (dv_split - dv_fused).abs().max().item()
        ok = dq_diff < 1e-2 and dk_diff < 1e-2 and dv_diff < 1e-2
        tag = f"H_Q={H_Q} H_KV={H_KV} N={N:>5} D={D} causal={causal} slide={slide}"
        status = "OK" if ok else "FAIL"
        print(f"  {tag:<58} dQ={dq_diff:.2e} dK={dk_diff:.2e} dV={dv_diff:.2e}  {status}")
        if not ok:
            all_ok = False
    return all_ok


def bench_sweep():
    print("\n=== Throughput: fused vs split backward (D=256, SWA slide=1024) ===")
    print(f"{'N':>6} {'split (ms)':>12} {'fused (ms)':>12} {'Speedup':>8}")
    print("-" * 46)
    for N in [1024, 2048, 4096, 8192, 16384]:
        q, k, v, do = make_tensors(1, 32, 16, N, 256)
        o, lse, slide_n = run_forward(q, k, v, causal=True, slide_size=1024)

        def fn_split():
            run_backward_split(q, k, v, do, o, lse, causal=True, slide_size=slide_n)
        def fn_fused():
            run_backward_fused(q, k, v, do, o, lse, causal=True, slide_size=slide_n)

        # Warm up (trigger compile)
        fn_split(); fn_fused()
        torch.cuda.synchronize()

        t_split = benchmark_fn(fn_split, warmup=5, rep=15 if N <= 4096 else 10)
        t_fused = benchmark_fn(fn_fused, warmup=5, rep=15 if N <= 4096 else 10)
        sp = t_split / t_fused
        print(f"{N:>6} {t_split:>12.3f} {t_fused:>12.3f} {sp:>7.2f}x")
        del q, k, v, do, o, lse
        torch.cuda.empty_cache()

    print("\n=== Throughput: fused vs split backward (D=512, full causal) ===")
    print(f"{'N':>6} {'split (ms)':>12} {'fused (ms)':>12} {'Speedup':>8}")
    print("-" * 46)
    for N in [1024, 2048, 4096, 8192]:
        q, k, v, do = make_tensors(1, 32, 4, N, 512)
        o, lse, slide_n = run_forward(q, k, v, causal=True, slide_size=0, BQ=64, BKV=32)

        def fn_split():
            run_backward_split(q, k, v, do, o, lse, causal=True, slide_size=0)
        def fn_fused():
            run_backward_fused(q, k, v, do, o, lse, causal=True, slide_size=0)

        try:
            fn_split(); fn_fused(); torch.cuda.synchronize()
            t_split = benchmark_fn(fn_split, warmup=3, rep=10 if N <= 4096 else 5)
            t_fused = benchmark_fn(fn_fused, warmup=3, rep=10 if N <= 4096 else 5)
            sp = t_split / t_fused
            print(f"{N:>6} {t_split:>12.3f} {t_fused:>12.3f} {sp:>7.2f}x")
        except Exception as e:
            print(f"{N:>6} ERROR {type(e).__name__}: {str(e)[:50]}")
        del q, k, v, do, o, lse
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ok = test_correctness()
    if not ok:
        print("\nFAIL — skipping benchmark")
        sys.exit(1)
    bench_sweep()
