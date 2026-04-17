"""D=512 Full-causal backward breakdown.

Measures delta / dQ / dKV kernel time independently for Gemma4 full config
(H_Q=32, H_KV=4, D=512), plus total bwd and SDPA baseline for comparison.
"""
import math
import os
import sys
import json

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import (
    FlashAttnGQAFunction,
    _delta_kernel,
    _flash_attn_gqa_bwd_dq_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
    flash_attn_gqa_train,
    attention_gqa_ref,
)


def time_cuda(fn, warmup=10, rep=30):
    """Median elapsed ms over `rep` runs (CUDA events)."""
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
    return times[len(times) // 2]


def run_breakdown(B, H_Q, H_KV, N, D, dtype=torch.float16, causal=True, slide_size=0,
                  warmup=10, rep=30):
    device = "cuda"
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device, requires_grad=True)

    # Run forward once to produce tensors fed to bwd kernels
    out = FlashAttnGQAFunction.apply(q.detach(), k.detach(), v.detach(), causal, slide_size)
    lse_shape_probe = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    # Re-run forward to capture lse explicitly (FlashAttnGQAFunction hides it in ctx)
    # Easier: call the kernel directly using the same config.
    from flash_attn.attention import _flash_attn_gqa_kernel
    o_ref = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    BLOCK_Q_F = 64 if D >= 512 else 128
    BLOCK_KV_F = 32 if D >= 512 else 64
    BLOCK_D_F = D
    num_warps_F = 8 if D >= 256 else 4
    BLOCK_Q_F = min(BLOCK_Q_F, triton.next_power_of_2(N))
    BLOCK_KV_F = min(BLOCK_KV_F, triton.next_power_of_2(N))
    grid_f = (triton.cdiv(N, BLOCK_Q_F), B * H_Q)
    qn, kn, vn = q.detach(), k.detach(), v.detach()
    _flash_attn_gqa_kernel[grid_f](
        qn, kn, vn, o_ref,
        qn.stride(0), qn.stride(1), qn.stride(2), qn.stride(3),
        kn.stride(0), kn.stride(1), kn.stride(2), kn.stride(3),
        vn.stride(0), vn.stride(1), vn.stride(2), vn.stride(3),
        o_ref.stride(0), o_ref.stride(1), o_ref.stride(2), o_ref.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BLOCK_Q_F, BLOCK_KV=BLOCK_KV_F, BLOCK_D=BLOCK_D_F,
        IS_CAUSAL=causal,
        SLIDE_SIZE=slide_size,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=True,
        num_warps=num_warps_F, num_stages=2,
    )

    do = torch.randn_like(o_ref)
    scale = 1.0 / math.sqrt(D)

    # --- per-kernel timings ---

    # 1) delta kernel
    delta = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    def run_delta():
        _delta_kernel[(N, B * H_Q)](
            do, o_ref, delta,
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            o_ref.stride(0), o_ref.stride(1), o_ref.stride(2), o_ref.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_HEADS=H_Q, SEQ_LEN=N, HEAD_DIM=D,
            num_warps=4, num_stages=2,
        )
    t_delta = time_cuda(run_delta, warmup=warmup, rep=rep)

    # Need a valid delta for the other kernels
    run_delta()
    torch.cuda.synchronize()

    # 2) dQ kernel
    dq = torch.empty_like(q)
    if D >= 512:
        BQ_dq, BKV_dq, w_dq = 32, 64, 8
    else:
        BQ_dq, BKV_dq, w_dq = 64, 64, 4
    BQ_dq = min(BQ_dq, triton.next_power_of_2(N))
    BKV_dq = min(BKV_dq, triton.next_power_of_2(N))
    grid_dq = (triton.cdiv(N, BQ_dq), B * H_Q)

    def run_dq():
        _flash_attn_gqa_bwd_dq_kernel[grid_dq](
            qn, kn, vn, do, dq, lse, delta,
            qn.stride(0), qn.stride(1), qn.stride(2), qn.stride(3),
            kn.stride(0), kn.stride(1), kn.stride(2), kn.stride(3),
            vn.stride(0), vn.stride(1), vn.stride(2), vn.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BQ_dq, BLOCK_KV=BKV_dq,
            IS_CAUSAL=causal,
            SLIDE_SIZE=slide_size,
            num_warps=w_dq, num_stages=2,
        )
    t_dq = time_cuda(run_dq, warmup=warmup, rep=rep)

    # 3) dKV packed kernel (mirrors attention.py defaults)
    GQA_RATIO = H_Q // H_KV
    if D >= 512:
        BKV_dkv, BQ_dkv, w_dkv = 16, 64, 4
    else:
        BKV_dkv, BQ_dkv, w_dkv = 32, 64, 4
    BKV_dkv = min(BKV_dkv, triton.next_power_of_2(N))
    BQ_dkv = min(BQ_dkv, triton.next_power_of_2(N))
    grid_dkv = (triton.cdiv(N, BKV_dkv), B * H_KV)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    def run_dkv():
        _flash_attn_gqa_bwd_dkv_packed_kernel[grid_dkv](
            qn, kn, vn, do, dk, dv, lse, delta,
            qn.stride(0), qn.stride(1), qn.stride(2), qn.stride(3),
            kn.stride(0), kn.stride(1), kn.stride(2), kn.stride(3),
            vn.stride(0), vn.stride(1), vn.stride(2), vn.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BQ_dkv, BLOCK_KV=BKV_dkv,
            GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=causal,
            SLIDE_SIZE=slide_size,
            num_warps=w_dkv, num_stages=2,
        )
    t_dkv = time_cuda(run_dkv, warmup=warmup, rep=rep)

    # 4) total backward via autograd (Triton)
    def run_full_bwd_triton():
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        o = flash_attn_gqa_train(q2, k2, v2, causal=causal, slide_size=slide_size)
        o.backward(do)
    # Different path: measure only bwd by calling .backward on cached output
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    def run_bwd_only_triton():
        q2.grad = None; k2.grad = None; v2.grad = None
        o = flash_attn_gqa_train(q2, k2, v2, causal=causal, slide_size=slide_size)
        o.backward(do, retain_graph=False)
    t_triton_fwdbwd = time_cuda(run_bwd_only_triton, warmup=warmup, rep=rep)

    # 5) SDPA baseline (fwd+bwd, same causal, GQA ref expands KV)
    q3 = q.detach().clone().requires_grad_(True)
    k3 = k.detach().clone().requires_grad_(True)
    v3 = v.detach().clone().requires_grad_(True)
    def run_bwd_only_sdpa():
        q3.grad = None; k3.grad = None; v3.grad = None
        o = attention_gqa_ref(q3, k3, v3, causal=causal)
        o.backward(do, retain_graph=False)
    # SDPA bwd will OOM at very long N for H_Q=32 (expands KV 8x)
    try:
        t_sdpa_fwdbwd = time_cuda(run_bwd_only_sdpa, warmup=3, rep=min(rep, 10))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        t_sdpa_fwdbwd = float("nan")

    # fwd-only for both (to derive bwd-only)
    def run_fwd_triton():
        FlashAttnGQAFunction.apply(q.detach(), k.detach(), v.detach(), causal, slide_size)
    t_triton_fwd = time_cuda(run_fwd_triton, warmup=warmup, rep=rep)

    def run_fwd_sdpa():
        attention_gqa_ref(q.detach(), k.detach(), v.detach(), causal=causal)
    try:
        t_sdpa_fwd = time_cuda(run_fwd_sdpa, warmup=5, rep=min(rep, 20))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        t_sdpa_fwd = float("nan")

    bwd_only_triton = t_triton_fwdbwd - t_triton_fwd
    bwd_only_sdpa = (t_sdpa_fwdbwd - t_sdpa_fwd) if not (math.isnan(t_sdpa_fwdbwd) or math.isnan(t_sdpa_fwd)) else float("nan")

    return {
        "N": N, "B": B, "H_Q": H_Q, "H_KV": H_KV, "D": D, "causal": causal,
        "t_delta": t_delta,
        "t_dq": t_dq,
        "t_dkv": t_dkv,
        "t_triton_fwd": t_triton_fwd,
        "t_triton_fwdbwd": t_triton_fwdbwd,
        "t_triton_bwd_only": bwd_only_triton,
        "t_sdpa_fwd": t_sdpa_fwd,
        "t_sdpa_fwdbwd": t_sdpa_fwdbwd,
        "t_sdpa_bwd_only": bwd_only_sdpa,
        "kernel_sum": t_delta + t_dq + t_dkv,
    }


def main():
    B, H_Q, H_KV, D = 1, 32, 4, 512
    dtype = torch.float16
    causal = True
    Ns = [1024, 4096, 16384, 32768]

    print(f"Config: B={B}, H_Q={H_Q}, H_KV={H_KV}, D={D}, dtype=fp16, causal={causal}")
    print(f"{'N':>6} | {'delta':>7} | {'dQ':>7} | {'dKV':>7} | {'sum':>7} | "
          f"{'tri_bwd':>8} | {'tri_fwd+bwd':>11} | {'sdpa_bwd':>9} | {'sdpa_fwd+bwd':>12} | "
          f"{'bwd_spd':>8}")
    rows = []
    for N in Ns:
        # Reduce rep count at very long N to stay under a few minutes total.
        if N <= 4096:
            warmup, rep = 10, 30
        elif N <= 16384:
            warmup, rep = 5, 15
        else:
            warmup, rep = 3, 8
        r = run_breakdown(B, H_Q, H_KV, N, D, dtype=dtype, causal=causal,
                          warmup=warmup, rep=rep)
        rows.append(r)
        spd = (r["t_sdpa_bwd_only"] / r["t_triton_bwd_only"]) if not math.isnan(r["t_sdpa_bwd_only"]) else float("nan")
        print(f"{N:>6} | {r['t_delta']:>7.3f} | {r['t_dq']:>7.3f} | {r['t_dkv']:>7.3f} | "
              f"{r['kernel_sum']:>7.3f} | {r['t_triton_bwd_only']:>8.3f} | "
              f"{r['t_triton_fwdbwd']:>11.3f} | {r['t_sdpa_bwd_only']:>9.3f} | "
              f"{r['t_sdpa_fwdbwd']:>12.3f} | {spd:>7.2f}x")

    # Percent breakdown
    print("\nPercent breakdown of (delta + dQ + dKV) inside triton fwd+bwd:")
    print(f"{'N':>6} | {'delta%':>7} | {'dQ%':>7} | {'dKV%':>7} | {'overhead%':>10}")
    for r in rows:
        tot = r["t_triton_fwdbwd"]
        print(f"{r['N']:>6} | {100*r['t_delta']/tot:>6.1f}% | "
              f"{100*r['t_dq']/tot:>6.1f}% | {100*r['t_dkv']/tot:>6.1f}% | "
              f"{100*(tot-r['kernel_sum']-r['t_triton_fwd'])/tot:>9.1f}%")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bwd_breakdown_D512.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
