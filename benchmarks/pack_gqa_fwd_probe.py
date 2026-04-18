"""Probe the existing `_flash_attn_gqa_grouped_kernel` on MoE full shape.

Questions to answer:
  1. Does it compile at D=512, GROUP_SIZE=8, various BQ / BKV / num_warps?
  2. Does it produce numerically correct output vs SDPA reference?
  3. How does it compare to the current split kernel (`_flash_attn_gqa_kernel`)
     on the MoE full config (H_Q=16, H_KV=2, D=512, slide=0)?

Not a tuning sweep yet — just a feasibility probe + initial data point.
"""
import math
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import (
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_grouped_kernel,
    attention_gqa_ref,
)


def run_split(q, k, v, causal, slide, BQ, BKV, num_warps, num_stages):
    """Current production split kernel."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    out = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(N, BQ), B * H_Q)
    _flash_attn_gqa_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ, BLOCK_KV=BKV, BLOCK_D=D,
        IS_CAUSAL=causal, SLIDE_SIZE=slide,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=False,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def run_grouped(q, k, v, causal, slide, GROUP, BQ, BKV, num_warps, num_stages):
    """Pack-GQA fwd kernel."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    assert H_Q % GROUP == 0
    assert GROUP * H_KV == H_Q or GROUP * (H_Q // H_KV) // (H_Q // H_KV) == GROUP  # sanity
    out = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=q.device)
    n_groups = H_Q // GROUP
    grid = (triton.cdiv(N, BQ), B * n_groups)
    _flash_attn_gqa_grouped_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ, BLOCK_KV=BKV,
        IS_CAUSAL=causal, SLIDE_SIZE=slide,
        GROUP_SIZE=GROUP,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=False,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def time_cuda(fn, warmup, rep):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def probe_one(B, H_Q, H_KV, N, D, slide, configs):
    print(f"\n{'=' * 100}")
    print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, slide={slide}, GQA={H_Q//H_KV}:1")
    print('=' * 100)

    torch.manual_seed(0)
    q = torch.randn(B, H_Q, N, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda")

    # Reference (SDPA)
    ref = attention_gqa_ref(q, k, v, causal=True)

    # Baseline: split kernel (current production config for D=512)
    BQ_s, BKV_s, w_s, st_s = 64, 32, 8, 2
    out_split = run_split(q, k, v, True, slide, BQ_s, BKV_s, w_s, st_s)
    t_split = time_cuda(lambda: run_split(q, k, v, True, slide, BQ_s, BKV_s, w_s, st_s),
                        warmup=10, rep=40)
    cs_split = cos_sim(out_split, ref)
    print(f"  SPLIT  (BQ={BQ_s}, BKV={BKV_s}, w={w_s}, s={st_s})  "
          f"t={t_split:.3f}ms  cos_sim={cs_split:.6f}")

    print(f"\n  {'GROUP':>5} | {'BQ':>3} | {'BKV':>4} | {'w':>2} | {'s':>1} | "
          f"{'t(ms)':>7} | {'cos_sim':>9} | {'vs split':>8}")
    print('  ' + '-' * 70)
    for (GROUP, BQ, BKV, nw, ns) in configs:
        if H_Q % GROUP != 0: continue
        try:
            out_g = run_grouped(q, k, v, True, slide, GROUP, BQ, BKV, nw, ns)
            cs = cos_sim(out_g, ref)
            t_g = time_cuda(lambda: run_grouped(q, k, v, True, slide, GROUP, BQ, BKV, nw, ns),
                            warmup=10, rep=40)
            sp = t_split / t_g
            print(f"  {GROUP:>5} | {BQ:>3} | {BKV:>4} | {nw:>2} | {ns:>1} | "
                  f"{t_g:>7.3f} | {cs:>9.6f} | {sp:>7.2f}x")
        except Exception as exc:
            print(f"  {GROUP:>5} | {BQ:>3} | {BKV:>4} | {nw:>2} | {ns:>1} | FAIL: {type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()


def main():
    # Configs to probe: start conservative on BQ (register pressure at D=512 * GROUP=8)
    # Add num_stages=1 variants to free shmem for bigger blocks.
    configs_d512 = [
        # GROUP, BQ, BKV, num_warps, num_stages
        # --- stages=2 sweep ---
        (8,  8, 32, 8, 2),
        (8, 16, 32, 8, 2),
        (4, 16, 32, 4, 2),
        (4, 32, 32, 8, 2),
        (2, 32, 32, 8, 2),
        (2, 64, 32, 8, 2),
        # --- stages=1 sweep (half the KV pipeline shmem → enable bigger Q tiles) ---
        (8, 16, 32, 4, 1),
        (8, 16, 32, 8, 1),
        (8, 32, 32, 8, 1),
        (8, 32, 64, 8, 1),
        (8, 64, 32, 8, 1),
        (4, 32, 32, 8, 1),
        (4, 32, 64, 8, 1),
        (4, 64, 32, 8, 1),
        (4, 64, 64, 8, 1),
        (4, 128, 32, 8, 1),
        (2, 64, 32, 8, 1),
        (2, 64, 64, 8, 1),
        (2, 128, 32, 8, 1),
        (2, 128, 64, 8, 1),
    ]

    # MoE full: H_Q=16, H_KV=2, GQA=8
    for N in (1024, 4096):
        probe_one(B=1, H_Q=16, H_KV=2, N=N, D=512, slide=0, configs=configs_d512)

    # E2B full (for comparison): H_Q=32, H_KV=4, GQA=8
    for N in (1024, 4096):
        probe_one(B=1, H_Q=32, H_KV=4, N=N, D=512, slide=0, configs=configs_d512)


if __name__ == "__main__":
    main()
