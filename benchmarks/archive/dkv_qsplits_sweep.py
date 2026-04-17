"""Sweep Q_SPLITS for dKV packed kernel to validate the grid-vs-SM heuristic.

Target: Config B Gemma-4-E2B (H_Q=8, H_KV=1, D=256) which has low dKV grid.
For each (N, Q_SPLITS), measure the standalone dKV kernel time and the
full fwd+bwd time (monkey-patching the heuristic if needed).
"""
import math
import os
import sys
import json

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flash_attn.attention import (
    _delta_kernel,
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dq_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
)

H100_SMS = 132


def time_cuda(fn, warmup=20, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times)//2]


def run(B, H_Q, H_KV, D, N, slide_size):
    device = "cuda"
    dtype = torch.float16
    GQA_RATIO = H_Q // H_KV
    scale = 1.0 / math.sqrt(D)
    swa_slide = slide_size if slide_size < N else 0

    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

    # fwd
    BLOCK_Q_F = min(128, triton.next_power_of_2(N))
    BLOCK_KV_F = min(64, triton.next_power_of_2(N))
    grid_f = (triton.cdiv(N, BLOCK_Q_F), B * H_Q)
    o = torch.empty_like(q); lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    _flash_attn_gqa_kernel[grid_f](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=scale, BLOCK_Q=BLOCK_Q_F, BLOCK_KV=BLOCK_KV_F, BLOCK_D=D,
        IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=True,
        num_warps=8, num_stages=2,
    )
    do = torch.randn_like(o)
    delta = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    _delta_kernel[(N, B*H_Q)](
        do, o, delta,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_HEADS=H_Q, SEQ_LEN=N, HEAD_DIM=D, num_warps=4, num_stages=2,
    )

    BKV_dkv, BQ_dkv, w_dkv, s_dkv = 32, 64, 4, 2
    raw_grid = triton.cdiv(N, BKV_dkv) * B * H_KV
    results = []
    for QS in [1, 2, 4, 8]:
        if QS > 1:
            dk = torch.zeros_like(k); dv = torch.zeros_like(v)
        else:
            dk = torch.empty_like(k); dv = torch.empty_like(v)
        grid_dkv = (triton.cdiv(N, BKV_dkv), B * H_KV, QS)

        def run_dkv():
            if QS > 1:
                dk.zero_(); dv.zero_()  # reset for each atomic run
            _flash_attn_gqa_bwd_dkv_packed_kernel[grid_dkv](
                q, k, v, do, dk, dv, lse, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
                HEAD_DIM=D, scale=scale,
                BLOCK_Q=BQ_dkv, BLOCK_KV=BKV_dkv,
                GQA_RATIO=GQA_RATIO,
                IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
                Q_SPLITS=QS,
                num_warps=w_dkv, num_stages=s_dkv,
            )
        t = time_cuda(run_dkv, warmup=30, rep=150)
        results.append({
            "QS": QS, "t_ms": t,
            "grid_mul": raw_grid * QS,
            "sm_util_%": 100 * raw_grid * QS / H100_SMS,
        })
    return results, raw_grid


def main():
    torch.manual_seed(0)
    for tag, B, H_Q, H_KV, D, slide in [
        ("Config B (Gemma-4-E2B 8:1)", 1, 8, 1, 256, 512),
        ("Config A (Gemma4 syn 2:1)", 1, 32, 16, 256, 1024),
    ]:
        print(f"\n{'='*86}")
        print(f" {tag}")
        print('='*86)
        print(f"{'N':>5} | {'raw_grid':>9} | {'QS=1':>8} | {'QS=2':>8} | {'QS=4':>8} | {'QS=8':>8} | best_QS | {'best(ms)':>8}")
        print('-'*86)
        for N in [1024, 2048, 4096, 8192]:
            res, raw_grid = run(1, H_Q, H_KV, D, N, slide)
            d = {r["QS"]: r["t_ms"] for r in res}
            best = min(d.items(), key=lambda x: x[1])
            print(f"{N:>5} | {raw_grid:>9} | {d[1]:>8.3f} | {d[2]:>8.3f} | {d[4]:>8.3f} | {d[8]:>8.3f} | QS={best[0]:>2} | {best[1]:>8.3f}")


if __name__ == "__main__":
    main()
