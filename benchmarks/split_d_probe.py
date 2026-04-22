"""Forward-attn micro-bench at HEAD_DIM=512 — before writing a Split-D variant.

Hypothesis (from dump_kernel_regs.py):
  fwd uses 192KB SMEM at (BQ=64, BKV=32, stages=2), regs=255, spills=4.
  A 3rd stage needs another 64KB → 256KB, exceeds the 228KB budget.

This probe answers: before we invest in Split-D, is the simpler alternative —
shrink BLOCK_KV so num_stages=3 fits — a better config?

Tested configs (all BQ=64, warps=8, causal=True):
  current:  BKV=32, stages=2   ← baseline
  alt-a:    BKV=16, stages=3   ← half tile, one extra stage (156KB est.)
  alt-b:    BKV=16, stages=4   ← maximally pipelined small tile
  alt-c:    BKV=32, stages=3   ← expect OOS/compile error; confirm limit
"""
import math
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import _flash_attn_gqa_kernel


def bench(B, H_Q, H_KV, N, D, BQ, BKV, warps, stages, causal=True, iters=50, warmup=10):
    q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    o = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device="cuda")

    grid = (triton.cdiv(N, BQ), B * H_Q)

    def run():
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
            LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
            stride_lsen=lse.stride(2), STORE_LSE=True,
            num_warps=warps, num_stages=stages,
        )

    # Warmup + compile probe
    try:
        for _ in range(warmup):
            run()
    except Exception as e:
        return None, None, str(e)[:120]

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    # Fetch compiled kernel SMEM + spill
    dc = _flash_attn_gqa_kernel.device_caches
    binder_cache, *_ = dc[0]
    ck = list(binder_cache.values())[-1]  # last compiled variant
    shmem_kb = ck.metadata.shared / 1024
    spills = ck.n_spills
    regs = ck.n_regs
    return ms, dict(shmem_kb=shmem_kb, spills=spills, regs=regs), None


def main():
    B, H_Q, H_KV, D = 1, 32, 4, 512
    torch.manual_seed(0)

    configs = [
        ("baseline  BKV=32 stages=2", 64, 32, 8, 2),
        ("alt-a     BKV=16 stages=3", 64, 16, 8, 3),
        ("alt-b     BKV=16 stages=4", 64, 16, 8, 4),
        ("alt-c     BKV=16 stages=2", 64, 16, 8, 2),
        ("alt-d     BKV=32 stages=3", 64, 32, 8, 3),  # expected to fail
        ("alt-e     BKV=64 stages=2", 64, 64, 8, 2),  # expected to fail
        ("alt-f     BQ=32 BKV=32 s=3", 32, 32, 4, 3),
        ("alt-g     BQ=32 BKV=64 s=2", 32, 64, 4, 2),
    ]

    for N in [4096, 8192]:
        print(f"\n=== N={N}, D={D}, H_Q={H_Q}, H_KV={H_KV}, causal, bf16 ===")
        print(f"{'config':<32} | {'ms':>7} | {'shmem':>7} | {'regs':>4} | {'spills':>6}")
        print("-" * 80)
        for name, BQ, BKV, w, stages in configs:
            ms, meta, err = bench(B, H_Q, H_KV, N, D, BQ, BKV, w, stages)
            if err:
                print(f"{name:<32} |    FAIL | {err}")
            else:
                print(f"{name:<32} | {ms:>6.3f} | {meta['shmem_kb']:>5.1f}KB | "
                      f"{meta['regs']:>4} | {meta['spills']:>5}")


if __name__ == "__main__":
    main()
