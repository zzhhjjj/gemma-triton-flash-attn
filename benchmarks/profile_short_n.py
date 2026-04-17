"""Short-N SWA profiling (N=1K/2K/4K) to analyze the remaining gap vs SDPA.

We target the synthetic Gemma4 sliding config that appears in baseline.md:
  D=256, H_Q=32, H_KV=16, slide=1024, where Fwd+Bwd is 0.73-1.23x vs SDPA.
Also runs the Gemma-4-E2B real config (H_Q=8, H_KV=1, slide=512) for contrast.

Reports per-kernel time (fwd / delta / dQ / dKV), end-to-end fwd-only and
fwd+bwd, SDPA fwd-only and fwd+bwd, plus launch-overhead estimate:
    overhead = fwd+bwd_total - (fwd + delta + dQ + dKV)
"""

import math
import os
import sys
import json
import argparse

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import (
    _delta_kernel,
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dq_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
    flash_attn_gqa_train,
    attention_gqa_ref,
)


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
    return times[len(times) // 2]


def run(B, H_Q, H_KV, D, N, slide_size, dtype=torch.float16, warmup=20, rep=100):
    device = "cuda"
    GQA_RATIO = H_Q // H_KV
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

    # Forward defaults match FlashAttnGQAFunction (D=256 path)
    BLOCK_Q_F = min(128, triton.next_power_of_2(N))
    BLOCK_KV_F = min(64, triton.next_power_of_2(N))
    num_warps_F = 8
    grid_f = (triton.cdiv(N, BLOCK_Q_F), B * H_Q)
    o = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)

    # When slide_size >= N the wrapper rewrites to 0 (full-causal); mirror here.
    swa_slide = slide_size if slide_size < N else 0

    def run_fwd_kernel():
        _flash_attn_gqa_kernel[grid_f](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
            scale=scale,
            BLOCK_Q=BLOCK_Q_F, BLOCK_KV=BLOCK_KV_F, BLOCK_D=D,
            IS_CAUSAL=True,
            SLIDE_SIZE=swa_slide,
            LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
            stride_lsen=lse.stride(2), STORE_LSE=True,
            num_warps=num_warps_F, num_stages=2,
        )

    t_fwd_kernel = time_cuda(run_fwd_kernel, warmup, rep)

    # Produce output once for bwd kernels
    run_fwd_kernel()
    torch.cuda.synchronize()
    do = torch.randn_like(o)

    # Delta kernel
    delta = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    def run_delta():
        _delta_kernel[(N, B * H_Q)](
            do, o, delta,
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_HEADS=H_Q, SEQ_LEN=N, HEAD_DIM=D,
            num_warps=4, num_stages=2,
        )
    t_delta = time_cuda(run_delta, warmup, rep)
    run_delta(); torch.cuda.synchronize()

    # dQ kernel (D<512 path)
    dq = torch.empty_like(q)
    BQ_dq, BKV_dq, w_dq = 64, 64, 4
    BQ_dq = min(BQ_dq, triton.next_power_of_2(N))
    BKV_dq = min(BKV_dq, triton.next_power_of_2(N))
    grid_dq = (triton.cdiv(N, BQ_dq), B * H_Q)
    def run_dq():
        _flash_attn_gqa_bwd_dq_kernel[grid_dq](
            q, k, v, do, o, dq, lse, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BQ_dq, BLOCK_KV=BKV_dq,
            IS_CAUSAL=True,
            SLIDE_SIZE=swa_slide,
            STORE_DELTA=False,
            num_warps=w_dq, num_stages=2,
        )
    t_dq = time_cuda(run_dq, warmup, rep)

    # dKV packed (D<512, N<8K path = (BKV=32, BQ=64, w=4, s=2))
    BKV_dkv, BQ_dkv, w_dkv, s_dkv = 32, 64, 4, 2
    BKV_dkv = min(BKV_dkv, triton.next_power_of_2(N))
    BQ_dkv = min(BQ_dkv, triton.next_power_of_2(N))
    grid_dkv = (triton.cdiv(N, BKV_dkv), B * H_KV)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    def run_dkv():
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
            IS_CAUSAL=True,
            SLIDE_SIZE=swa_slide,
            Q_SPLITS=1,
            num_warps=w_dkv, num_stages=s_dkv,
        )
    t_dkv = time_cuda(run_dkv, warmup, rep)

    # Triton fwd-only via autograd wrapper (matches hot path)
    def run_triton_fwd_only():
        flash_attn_gqa_train(q, k, v, causal=True, slide_size=slide_size)
    t_triton_fwd = time_cuda(run_triton_fwd_only, warmup, rep)

    # Triton fwd+bwd via autograd
    qg = q.detach().clone().requires_grad_(True)
    kg = k.detach().clone().requires_grad_(True)
    vg = v.detach().clone().requires_grad_(True)
    def run_triton_fwdbwd():
        qg.grad = None; kg.grad = None; vg.grad = None
        out = flash_attn_gqa_train(qg, kg, vg, causal=True, slide_size=slide_size)
        out.backward(do, retain_graph=False)
    t_triton_fwdbwd = time_cuda(run_triton_fwdbwd, warmup, rep)

    # SDPA fwd (full-causal, same reference used in all baseline.md tables)
    def run_sdpa_fwd():
        attention_gqa_ref(q, k, v, causal=True)
    t_sdpa_fwd = time_cuda(run_sdpa_fwd, warmup, rep)

    # SDPA fwd+bwd
    qs = q.detach().clone().requires_grad_(True)
    ks = k.detach().clone().requires_grad_(True)
    vs = v.detach().clone().requires_grad_(True)
    def run_sdpa_fwdbwd():
        qs.grad = None; ks.grad = None; vs.grad = None
        out = attention_gqa_ref(qs, ks, vs, causal=True)
        out.backward(do, retain_graph=False)
    t_sdpa_fwdbwd = time_cuda(run_sdpa_fwdbwd, warmup, rep)

    # Derived
    kernel_sum_bwd = t_delta + t_dq + t_dkv
    kernel_sum_fwdbwd = t_fwd_kernel + kernel_sum_bwd
    overhead_ms = max(0.0, t_triton_fwdbwd - kernel_sum_fwdbwd)

    sdpa_bwd_only = t_sdpa_fwdbwd - t_sdpa_fwd
    triton_bwd_only = t_triton_fwdbwd - t_triton_fwd

    return {
        "config": {"B": B, "H_Q": H_Q, "H_KV": H_KV, "D": D, "N": N,
                   "slide_size": slide_size, "GQA_RATIO": GQA_RATIO,
                   "effective_slide": swa_slide},
        "kernels": {
            "fwd": t_fwd_kernel,
            "delta": t_delta,
            "dq": t_dq,
            "dkv": t_dkv,
            "sum_bwd_only": kernel_sum_bwd,
            "sum_fwdbwd": kernel_sum_fwdbwd,
        },
        "triton": {
            "fwd_only": t_triton_fwd,
            "fwdbwd": t_triton_fwdbwd,
            "bwd_only": triton_bwd_only,
            "overhead_ms": overhead_ms,
            "overhead_%": 100 * overhead_ms / t_triton_fwdbwd,
        },
        "sdpa": {
            "fwd_only": t_sdpa_fwd,
            "fwdbwd": t_sdpa_fwdbwd,
            "bwd_only": sdpa_bwd_only,
        },
        "speedup": {
            "fwd": t_sdpa_fwd / t_triton_fwd,
            "fwdbwd": t_sdpa_fwdbwd / t_triton_fwdbwd,
            "bwd_only": sdpa_bwd_only / triton_bwd_only if triton_bwd_only > 0 else float("nan"),
        },
    }


def print_summary(results, tag):
    print(f"\n{'='*96}")
    print(f" {tag}")
    print('='*96)
    hdr = (f"{'N':>5} | {'fwd':>6} | {'delta':>6} | {'dQ':>6} | {'dKV':>6} | "
           f"{'K_sum':>6} | {'TRI_fb':>7} | {'OH':>5} | {'OH%':>5} | "
           f"{'SDPA_f':>7} | {'SDPA_fb':>8} | {'s_fwd':>6} | {'s_fb':>6} | {'s_bwd':>6}")
    print(hdr)
    print('-'*len(hdr))
    for r in results:
        c = r["config"]
        k = r["kernels"]
        t = r["triton"]
        s = r["sdpa"]
        sp = r["speedup"]
        print(f"{c['N']:>5} | {k['fwd']:>6.3f} | {k['delta']:>6.3f} | {k['dq']:>6.3f} | "
              f"{k['dkv']:>6.3f} | {k['sum_fwdbwd']:>6.3f} | {t['fwdbwd']:>7.3f} | "
              f"{t['overhead_ms']:>5.3f} | {t['overhead_%']:>4.1f}% | "
              f"{s['fwd_only']:>7.3f} | {s['fwdbwd']:>8.3f} | "
              f"{sp['fwd']:>5.2f}x | {sp['fwdbwd']:>5.2f}x | {sp['bwd_only']:>5.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--out", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "profile_short_n.json"))
    args = p.parse_args()

    torch.manual_seed(0)

    all_results = {}

    # Config A: synthetic Gemma4 sliding (baseline.md 0.73x/0.88x table)
    tag_a = "A. Gemma4 synthetic sliding  (B=1, H_Q=32, H_KV=16, D=256, slide=1024)"
    res_a = []
    for N in [1024, 2048, 4096]:
        r = run(B=1, H_Q=32, H_KV=16, D=256, N=N, slide_size=1024,
                warmup=args.warmup, rep=args.rep)
        res_a.append(r)
    print_summary(res_a, tag_a)
    all_results["gemma4_synthetic_slide1024"] = res_a

    # Config B: Gemma-4-E2B real sliding (H_Q=8, H_KV=1, slide=512)
    tag_b = "B. Gemma-4-E2B real sliding  (B=1, H_Q=8, H_KV=1, D=256, slide=512)"
    res_b = []
    for N in [1024, 2048, 4096]:
        r = run(B=1, H_Q=8, H_KV=1, D=256, N=N, slide_size=512,
                warmup=args.warmup, rep=args.rep)
        res_b.append(r)
    print_summary(res_b, tag_b)
    all_results["gemma4_e2b_slide512"] = res_b

    # Save
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
