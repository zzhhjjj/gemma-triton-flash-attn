"""Run ONE dKV (packed) config and print ms — isolated subprocess.

Usage:
    python b200_moe_bwd_dkv_runone.py <H_Q> <H_KV> <N> <D> <slide> <BQ> <BKV> <warps> <stages> <q_splits>
"""
from __future__ import annotations

import math
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.attention import _flash_attn_gqa_bwd_dkv_packed_kernel


def main():
    H_Q, H_KV, N, D, slide, BQ, BKV, warps, stages, q_splits = map(int, sys.argv[1:11])
    if BQ > triton.next_power_of_2(N) or BKV > triton.next_power_of_2(N):
        print("RESULT skip oversized")
        return
    if H_Q % H_KV != 0:
        print("RESULT skip badGQA")
        return
    GQA_RATIO = H_Q // H_KV

    torch.manual_seed(0)
    q = torch.randn(1, H_Q, N, D, dtype=torch.float16, device="cuda")
    k = torch.randn(1, H_KV, N, D, dtype=torch.float16, device="cuda")
    v = torch.randn(1, H_KV, N, D, dtype=torch.float16, device="cuda")
    do = torch.randn(1, H_Q, N, D, dtype=torch.float16, device="cuda")
    lse = torch.zeros(1, H_Q, N, dtype=torch.float32, device="cuda")
    delta = torch.zeros(1, H_Q, N, dtype=torch.float32, device="cuda")

    if q_splits > 1:
        dkv = torch.zeros((2, 1, H_KV, N, D), dtype=k.dtype, device="cuda")
        dk, dv = dkv[0], dkv[1]
    else:
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

    grid = (triton.cdiv(N, BKV), 1 * H_KV, q_splits)

    def fn():
        if q_splits > 1:
            dkv.zero_()
        _flash_attn_gqa_bwd_dkv_packed_kernel[grid](
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
            HEAD_DIM=D, scale=1.0 / math.sqrt(D),
            BLOCK_Q=BQ, BLOCK_KV=BKV,
            GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=True, SLIDE_SIZE=slide,
            Q_SPLITS=q_splits,
            GroupIds_ptr=None, GroupLo_ptr=None, GroupHi_ptr=None,
            stride_gb=0, stride_gn=0, HAS_GROUP_IDS=False,
            num_warps=warps, num_stages=stages,
        )

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
