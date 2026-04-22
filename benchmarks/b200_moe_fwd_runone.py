"""Run ONE fwd config and print ms — isolated subprocess for the sweep driver.

The wgmma misaligned-address fault on B200 D=512 corrupts CUDA driver state
(even empty_cache fails), so we run each config in its own process. The
parent collects results from stdout.

Usage:
    python b200_moe_fwd_runone.py <H_Q> <H_KV> <N> <D> <slide> <BQ> <BKV> <warps> <stages>
"""
from __future__ import annotations

import math
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.attention import _flash_attn_gqa_kernel


def main():
    H_Q, H_KV, N, D, slide, BQ, BKV, warps, stages = map(int, sys.argv[1:10])
    if BQ > triton.next_power_of_2(N) or BKV > triton.next_power_of_2(N):
        print("RESULT skip oversized")
        return

    torch.manual_seed(0)
    q = torch.randn(1, H_Q, N, D, dtype=torch.float16, device="cuda")
    k = torch.randn(1, H_KV, N, D, dtype=torch.float16, device="cuda")
    v = torch.randn(1, H_KV, N, D, dtype=torch.float16, device="cuda")
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BQ), 1 * H_Q)

    def fn():
        _flash_attn_gqa_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
            scale=1.0 / math.sqrt(D),
            BLOCK_Q=BQ, BLOCK_KV=BKV, BLOCK_D=D,
            IS_CAUSAL=True, SLIDE_SIZE=slide,
            LSE_ptr=o, stride_lseb=0, stride_lseh=0, stride_lsen=0,
            STORE_LSE=False,
            GroupIds_ptr=None, GroupLo_ptr=None, GroupHi_ptr=None,
            stride_gb=0, stride_gn=0, HAS_GROUP_IDS=False,
            num_warps=warps, num_stages=stages,
        )

    # Warmup (compile + first launches). If we fault here the process dies
    # with a CUDA error and the parent records it as a fault.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 20.0
    print(f"RESULT ok {ms:.4f}")


if __name__ == "__main__":
    main()
