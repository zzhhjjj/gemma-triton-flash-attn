"""MFU measurement across realistic Gemma shapes — Triton vs SDPA.

H100 SXM5 bf16 peak (no sparsity) = 989 TFLOPS.

FLOPs per fwd attention call (bf16 MMA counts each fma as 2 flops):
  - Full causal:  2 × B × H_Q × N²  × D × 2  (× 0.5 causal) = 2 B H_Q N² D
    [QK^T: 2 B H_Q N² D, P@V: 2 B H_Q N² D, halve for causal]
  - SWA causal:   2 × 2 × B × H_Q × N × W × D × (1 - W/(2N))  approx for N >> W
    [each query attends to W keys; minus boundary triangle]

Reports both TFLOPS and MFU = TFLOPS / 989.
"""
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.attention import attention_flash_gqa

H100_BF16_PEAK_TFLOPS = 989.0


def causal_flops(B, H_Q, N, D):
    return 2.0 * B * H_Q * N * N * D  # halved for causal already (2 matmul × 2 ops × 0.5)


def swa_flops(B, H_Q, N, D, W):
    if W >= N:
        return causal_flops(B, H_Q, N, D)
    # Each query attends to up to W keys (clipped at start by causal triangle).
    # Total key-contributions: W * N - W*(W-1)/2 (the leading triangle below W).
    contrib = W * N - W * (W - 1) / 2
    return 4.0 * B * H_Q * contrib * D  # 2 matmul × 2 flops/op


def run_sdpa(q, k, v, causal, slide_size):
    if slide_size > 0 and slide_size < q.shape[-2]:
        N = q.shape[-2]
        idx = torch.arange(N, device=q.device)
        causal_mask = idx[None, :] <= idx[:, None]
        win_mask = (idx[:, None] - idx[None, :]) < slide_size
        attend = causal_mask & win_mask
        bias = torch.zeros(N, N, dtype=q.dtype, device=q.device)
        bias.masked_fill_(~attend, float('-inf'))
        # expand KV to match Q heads
        H_Q, H_KV = q.shape[1], k.shape[1]
        if H_Q != H_KV:
            r = H_Q // H_KV
            ke = k.repeat_interleave(r, dim=1)
            ve = v.repeat_interleave(r, dim=1)
        else:
            ke, ve = k, v
        return F.scaled_dot_product_attention(q, ke, ve, attn_mask=bias, is_causal=False)
    H_Q, H_KV = q.shape[1], k.shape[1]
    if H_Q != H_KV:
        r = H_Q // H_KV
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def time_fn(fn, n_warmup=5, n_rep=20):
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


def bench_shape(label, B, H_Q, H_KV, N, D, causal, slide, sdpa_ok=True):
    q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")

    if slide > 0 and slide < N:
        flops = swa_flops(B, H_Q, N, D, slide)
        causal_f = True
    else:
        flops = causal_flops(B, H_Q, N, D)
        causal_f = causal

    # Triton
    try:
        ms_t = time_fn(lambda: attention_flash_gqa(q, k, v, causal=causal_f, slide_size=slide))
        tflops_t = flops / (ms_t / 1000) / 1e12
        mfu_t = 100 * tflops_t / H100_BF16_PEAK_TFLOPS
    except Exception as e:
        ms_t, tflops_t, mfu_t = float("nan"), 0.0, 0.0

    # SDPA
    if sdpa_ok:
        try:
            ms_s = time_fn(lambda: run_sdpa(q, k, v, causal_f, slide))
            tflops_s = flops / (ms_s / 1000) / 1e12
            mfu_s = 100 * tflops_s / H100_BF16_PEAK_TFLOPS
        except Exception as e:
            ms_s, tflops_s, mfu_s = float("nan"), 0.0, 0.0
    else:
        ms_s, tflops_s, mfu_s = float("nan"), 0.0, 0.0

    speedup = ms_s / ms_t if (ms_s == ms_s and ms_t == ms_t and ms_t > 0) else float("nan")
    print(f"{label:<42} | {N:>5} | {ms_t:>7.3f} | {tflops_t:>6.1f} | {mfu_t:>5.2f}% | "
          f"{ms_s:>7.3f} | {tflops_s:>6.1f} | {mfu_s:>5.2f}% | {speedup:>5.2f}×")


def main():
    torch.manual_seed(0)
    print(f"H100 bf16 peak = {H100_BF16_PEAK_TFLOPS} TFLOPS\n")
    print(f"{'shape':<42} | {'N':>5} | {'Tri ms':>7} | {'Tri TF':>6} | {'Tri MFU':>6} | "
          f"{'SDPA ms':>7} | {'SDPA TF':>6} | {'SDPA M':>6} | {'speed':>5}")
    print("-" * 130)

    # === Gemma-4-E2B (H_Q=8, H_KV=1). 35 layers: 28 sliding D=256 slide=512,
    #     7 full D=512.  GQA ratio 8:1. ===
    print("# Gemma-4-E2B  H_Q=8 H_KV=1  (28 sliding D=256, 7 full D=512)")
    for N in [2048, 4096, 8192, 16384]:
        bench_shape("E2B sliding D=256 slide=512", 1, 8, 1, N, 256, True, 512)
    for N in [2048, 4096, 8192, 16384]:
        bench_shape("E2B full    D=512", 1, 8, 1, N, 512, True, 0)

    # === Gemma-4-E4B (H_Q=8, H_KV=2). 42 layers: 35 sliding D=256, 7 full D=512.
    #     GQA ratio 4:1. ===
    print("\n# Gemma-4-E4B  H_Q=8 H_KV=2  (35 sliding D=256, 7 full D=512)")
    for N in [2048, 4096, 8192, 16384]:
        bench_shape("E4B sliding D=256 slide=512", 1, 8, 2, N, 256, True, 512)
    for N in [2048, 4096, 8192, 16384]:
        bench_shape("E4B full    D=512", 1, 8, 2, N, 512, True, 0)

    # === Gemma-4 MoE 26B-A4B (H_Q=16, H_KV=8). 30 layers: 25 sliding D=256
    #     slide=1024, 5 full D=512.  GQA ratio 2:1. ===
    print("\n# Gemma-4-MoE 26B-A4B  H_Q=16 H_KV=8  (25 sliding D=256 slide=1024, 5 full D=512)")
    for N in [2048, 4096, 8192]:
        bench_shape("MoE full    D=512", 1, 16, 8, N, 512, True, 0)
    for N in [2048, 4096, 8192, 16384]:
        bench_shape("MoE sliding D=256 slide=1024", 1, 16, 8, N, 256, True, 1024)


if __name__ == "__main__":
    main()
