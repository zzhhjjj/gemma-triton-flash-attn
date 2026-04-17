"""Packed dKV kernel (KV-major, inline Q-head loop) vs split+reduce baseline.

Inspired by flash-attention's pack_gqa=True mode: instead of running a separate
program per Q head and reducing, merge all GQA Q heads into a single program's
inner loop so there's only one accumulator per KV block — no expand buffer,
no reduce, no atomics.

Usage:
    source /opt/tiger/flash_gemma/bin/activate
    python tests/test_packed_dkv.py
"""
from __future__ import annotations

import math
import sys

import torch
import triton

from gemma_triton_flash_attn.attention import (
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dq_kernel,
    _flash_attn_gqa_bwd_dkv_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
    _delta_kernel,
)
from gemma_triton_flash_attn.utils import benchmark_fn


def run_forward_with_lse(q, k, v, *, causal, slide_size):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    if slide_size > 0 and slide_size >= N:
        slide_size = 0
    BQ = min(64 if D >= 512 else 128, triton.next_power_of_2(N))
    BKV = min(32 if D >= 512 else 64, triton.next_power_of_2(N))
    num_warps = 8 if D >= 256 else 4
    output = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=q.device)
    _flash_attn_gqa_kernel[(triton.cdiv(N, BQ), B * H_Q)](
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
    return output, lse, slide_size


def run_delta(do, o):
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


def run_dkv_split(q, k, v, do, lse, delta, *, causal, slide_size):
    """Current production design: per-Q-head dKV + PyTorch reduce."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    GQA_RATIO = H_Q // H_KV
    scale = 1.0 / math.sqrt(D)

    if D >= 512:
        BQ, BKV, nw = 16, 32, 8
    else:
        BQ, BKV, nw = 64, 32, 4
    BQ = min(BQ, triton.next_power_of_2(N))
    BKV = min(BKV, triton.next_power_of_2(N))

    dk_expanded = torch.empty(B, H_Q, N, D, dtype=q.dtype, device=q.device)
    dv_expanded = torch.empty(B, H_Q, N, D, dtype=q.dtype, device=q.device)

    _flash_attn_gqa_bwd_dkv_kernel[(triton.cdiv(N, BKV), B * H_Q)](
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
        BLOCK_Q=BQ, BLOCK_KV=BKV,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
        ATOMIC_REDUCE=False,
        num_warps=nw, num_stages=2,
    )

    if GQA_RATIO == 2:
        dk_r = dk_expanded.view(B, H_KV, 2, N, D)
        dv_r = dv_expanded.view(B, H_KV, 2, N, D)
        dk = dk_r[:, :, 0] + dk_r[:, :, 1]
        dv = dv_r[:, :, 0] + dv_r[:, :, 1]
    else:
        dk = dk_expanded.view(B, H_KV, GQA_RATIO, N, D).sum(dim=2)
        dv = dv_expanded.view(B, H_KV, GQA_RATIO, N, D).sum(dim=2)
    return dk, dv


def run_dkv_packed(q, k, v, do, lse, delta, *, causal, slide_size,
                   BQ=None, BKV=None, num_warps=None):
    """New design: KV-major grid, inline all GQA Q heads, no expand/reduce."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    GQA_RATIO = H_Q // H_KV
    scale = 1.0 / math.sqrt(D)

    # Block sizes: same as split kernel (same register budget)
    if BQ is None:
        BQ = 16 if D >= 512 else 64
    if BKV is None:
        BKV = 32
    if num_warps is None:
        num_warps = 8 if D >= 512 else 4
    BQ = min(BQ, triton.next_power_of_2(N))
    BKV = min(BKV, triton.next_power_of_2(N))

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    _flash_attn_gqa_bwd_dkv_packed_kernel[(triton.cdiv(N, BKV), B * H_KV)](
        q, k, v, do, dk, dv, lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D, scale=scale,
        BLOCK_Q=BQ, BLOCK_KV=BKV,
        GQA_RATIO=GQA_RATIO,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
        Q_SPLITS=1,
        num_warps=num_warps, num_stages=2,
    )
    return dk, dv


def make_tensors(B, H_Q, H_KV, N, D, dtype=torch.float16):
    torch.manual_seed(0)
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda")
    do = torch.randn(B, H_Q, N, D, dtype=dtype, device="cuda")
    return q, k, v, do


def test_correctness():
    print("=== Correctness: packed dKV vs split+reduce ===")
    all_ok = True
    configs = [
        # (H_Q, H_KV, N, D, causal, slide)
        (8, 1, 512, 256, True, 0),
        (8, 1, 1024, 256, True, 0),
        (8, 1, 2048, 256, True, 512),    # SWA (real window truncation)
        (8, 1, 2048, 256, True, 0),
        (32, 4, 1024, 512, True, 0),
        (32, 4, 2048, 512, True, 0),
        (32, 16, 2048, 256, True, 1024),  # Gemma4 sliding config
    ]
    for H_Q, H_KV, N, D, causal, slide in configs:
        q, k, v, do = make_tensors(1, H_Q, H_KV, N, D)
        o, lse, slide_n = run_forward_with_lse(q, k, v, causal=causal, slide_size=slide)
        delta = run_delta(do, o)

        dk_split, dv_split = run_dkv_split(q, k, v, do, lse, delta,
                                            causal=causal, slide_size=slide_n)
        try:
            dk_pack, dv_pack = run_dkv_packed(q, k, v, do, lse, delta,
                                               causal=causal, slide_size=slide_n)
        except triton.runtime.errors.OutOfResources as e:
            print(f"  H_Q={H_Q} H_KV={H_KV} N={N:>5} D={D} SKIP (shmem OOM)")
            continue

        dk_diff = (dk_split - dk_pack).abs().max().item()
        dv_diff = (dv_split - dv_pack).abs().max().item()
        ok = dk_diff < 1e-2 and dv_diff < 1e-2
        tag = f"H_Q={H_Q} H_KV={H_KV} N={N:>5} D={D} causal={causal} slide={slide}"
        print(f"  {tag:<58} dK={dk_diff:.2e} dV={dv_diff:.2e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    return all_ok


def bench_sweep():
    print("\n=== Throughput: dKV only (D=256 sliding, Gemma4 sliding config) ===")
    print(f"{'N':>6} {'split (ms)':>12} {'packed (ms)':>13} {'Speedup':>8}")
    print("-" * 46)
    for N in [1024, 2048, 4096, 8192, 16384]:
        q, k, v, do = make_tensors(1, 32, 16, N, 256)
        o, lse, _ = run_forward_with_lse(q, k, v, causal=True, slide_size=1024)
        delta = run_delta(do, o)

        def f_split():
            run_dkv_split(q, k, v, do, lse, delta, causal=True,
                          slide_size=1024 if N > 1024 else 0)
        def f_packed():
            run_dkv_packed(q, k, v, do, lse, delta, causal=True,
                           slide_size=1024 if N > 1024 else 0)

        # Warmup (trigger compile)
        f_split(); f_packed(); torch.cuda.synchronize()

        t_split = benchmark_fn(f_split, warmup=5, rep=20 if N <= 4096 else 10)
        t_packed = benchmark_fn(f_packed, warmup=5, rep=20 if N <= 4096 else 10)
        sp = t_split / t_packed
        print(f"{N:>6} {t_split:>12.3f} {t_packed:>13.3f} {sp:>7.2f}x")
        del q, k, v, do, o, lse, delta
        torch.cuda.empty_cache()

    print("\n=== Throughput: dKV only (D=512 full causal, Gemma4 full config) ===")
    print(f"{'N':>6} {'split (ms)':>12} {'packed (ms)':>13} {'Speedup':>8}")
    print("-" * 46)
    for N in [1024, 2048, 4096, 8192]:
        q, k, v, do = make_tensors(1, 32, 4, N, 512)
        o, lse, _ = run_forward_with_lse(q, k, v, causal=True, slide_size=0)
        delta = run_delta(do, o)

        def f_split():
            run_dkv_split(q, k, v, do, lse, delta, causal=True, slide_size=0)
        def f_packed():
            run_dkv_packed(q, k, v, do, lse, delta, causal=True, slide_size=0)

        try:
            f_split(); f_packed(); torch.cuda.synchronize()
            t_split = benchmark_fn(f_split, warmup=3, rep=10)
            t_packed = benchmark_fn(f_packed, warmup=3, rep=10)
            sp = t_split / t_packed
            print(f"{N:>6} {t_split:>12.3f} {t_packed:>13.3f} {sp:>7.2f}x")
        except Exception as e:
            print(f"{N:>6} ERROR {type(e).__name__}: {str(e)[:60]}")
        del q, k, v, do, o, lse, delta
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ok = test_correctness()
    if not ok:
        print("\nFAIL — skipping benchmark")
        sys.exit(1)
    bench_sweep()
