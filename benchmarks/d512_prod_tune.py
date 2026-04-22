"""Tune block config for the actual production D=512 shape.

Existing baseline defaults (D=512): BQ=64 BKV=32 warps=8 stages=2 — these
were tuned for H_Q=32 H_KV=4 (a hypothetical shape, see mfu_sweep.py).
Real production shape is H_Q=8 H_KV=1 (E2B), GQA 8:1, fewer programs in
the grid → maybe a different block balance wins.

Sweep BQ × BKV × warps × stages, report ms + SMEM + spills.
"""
import math
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.attention import _flash_attn_gqa_kernel


def run(q, k, v, BQ, BKV, warps, stages, causal=True):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BQ), B * H_Q)
    _flash_attn_gqa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ, BLOCK_KV=BKV, BLOCK_D=D,
        IS_CAUSAL=causal, SLIDE_SIZE=0,
        LSE_ptr=o, stride_lseb=0, stride_lseh=0, stride_lsen=0,
        STORE_LSE=False,
        num_warps=warps, num_stages=stages,
    )


def time_fn(fn, n_warmup=10, n_rep=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_rep


def get_ck():
    dc = _flash_attn_gqa_kernel.device_caches
    bc, *_ = dc[0]
    return list(bc.values())[-1]


def sweep(B, H_Q, H_KV, N, D, label):
    print(f"\n=== {label}: B={B} H_Q={H_Q} H_KV={H_KV} N={N} D={D} causal bf16 ===")
    q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")

    flops = 2.0 * B * H_Q * N * N * D  # causal
    print(f"{'BQ':>3} {'BKV':>3} {'w':>2} {'s':>2} | {'ms':>7} | {'TF':>5} | {'MFU':>5} | {'shmem':>7} | {'regs':>4} | {'spl':>4}")
    print("-" * 80)

    configs = [
        # (BQ, BKV, warps, stages)
        (64, 32, 8, 2),  # baseline
        (32, 32, 4, 2),
        (32, 32, 4, 3),
        (32, 32, 8, 3),
        (32, 64, 4, 2),
        (32, 64, 8, 2),
        (16, 64, 4, 3),
        (16, 64, 8, 3),
        (16, 128, 4, 2),
        (16, 128, 8, 2),
        (16, 32, 4, 4),
        (16, 32, 4, 3),
        (128, 16, 8, 2),
        (128, 16, 8, 3),
        (128, 32, 8, 2),
    ]
    rows = []
    for BQ, BKV, w, st in configs:
        try:
            ms = time_fn(lambda: run(q, k, v, BQ, BKV, w, st))
            ck = get_ck()
            tf = flops / (ms / 1000) / 1e12
            mfu = 100 * tf / 989.0
            rows.append((ms, tf, mfu, BQ, BKV, w, st, ck.metadata.shared/1024, ck.n_regs, ck.n_spills))
            print(f"{BQ:>3} {BKV:>3} {w:>2} {st:>2} | {ms:>6.3f} | {tf:>4.0f} | {mfu:>4.1f}% | "
                  f"{ck.metadata.shared/1024:>5.1f}KB | {ck.n_regs:>4} | {ck.n_spills:>4}")
        except Exception as e:
            print(f"{BQ:>3} {BKV:>3} {w:>2} {st:>2} | FAIL: {str(e)[:60]}")
    rows.sort()
    if rows:
        ms, tf, mfu, BQ, BKV, w, st, *_ = rows[0]
        print(f"\nBEST: BQ={BQ} BKV={BKV} w={w} s={st}  →  {ms:.3f} ms  ({mfu:.1f}% MFU)")


def main():
    torch.manual_seed(0)
    # Real E2B/E4B full-attention layer
    sweep(1, 8, 1, 8192, 512, "E2B full D=512")
    # MoE 26B-A4B full-attention layer (different GQA ratio)
    sweep(1, 16, 8, 8192, 512, "MoE full D=512")


if __name__ == "__main__":
    main()
