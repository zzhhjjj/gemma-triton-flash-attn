"""Attention-only (kernel-level) benchmark across full N range (1K-32K).

Covers three configs:
  A. Gemma4 synthetic sliding    (B=1, H_Q=32, H_KV=16, D=256, slide=1024)
  B. Gemma-4-E2B real sliding    (B=1, H_Q=8,  H_KV=1,  D=256, slide=512)
  F. Gemma4 full-causal          (B=1, H_Q=32, H_KV=4,  D=512, slide=0)

Reports for each: Forward-only and Fwd+Bwd timings, Triton vs SDPA.

Usage:
  Full sweep (Tier-2 release bench, ~15 min):
    python benchmarks/attn_only_all_n.py

  Quick check (Tier-1 pre-commit, ~2 min, skips long N):
    python benchmarks/attn_only_all_n.py --quick
"""
import argparse
import math
import os
import sys
import json

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import flash_attn_gqa_train, attention_gqa_ref


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
    return ts[len(ts)//2]


def bench_config(tag, B, H_Q, H_KV, D, slide, N_list):
    print(f"\n{'='*110}")
    print(f"  {tag}")
    print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, D={D}, slide_size={slide}, dtype=fp16")
    print('='*110)
    print(f"{'N':>6} | {'Tri fwd':>8} | {'SDPA fwd':>9} | {'fwd spd':>7} |"
          f" {'Tri F+B':>8} | {'SDPA F+B':>10} | {'F+B spd':>7} |"
          f" {'Tri bwd':>8} | {'SDPA bwd':>9} | {'bwd spd':>7}")
    print('-'*110)

    rows = []
    device = "cuda"; dtype = torch.float16
    for N in N_list:
        torch.manual_seed(0)
        # Adaptive rep counts: short N → more rep; long N → fewer
        if N <= 1024: warmup, rep = 20, 100
        elif N <= 4096: warmup, rep = 10, 40
        elif N <= 16384: warmup, rep = 5, 15
        else: warmup, rep = 3, 8

        q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

        # Triton fwd only
        def run_tri_fwd():
            flash_attn_gqa_train(q, k, v, causal=True, slide_size=slide)
        try:
            t_tri_fwd = time_cuda(run_tri_fwd, warmup, rep)
        except Exception: t_tri_fwd = float('nan')

        # SDPA fwd only (full-causal ref — no SWA backend)
        def run_sdpa_fwd():
            attention_gqa_ref(q, k, v, causal=True)
        try:
            t_sdpa_fwd = time_cuda(run_sdpa_fwd, warmup, rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); t_sdpa_fwd = float('nan')
        except Exception: t_sdpa_fwd = float('nan')

        # Triton fwd+bwd
        qg = q.detach().clone().requires_grad_(True)
        kg = k.detach().clone().requires_grad_(True)
        vg = v.detach().clone().requires_grad_(True)
        do = torch.randn_like(q)
        def run_tri_fb():
            qg.grad = None; kg.grad = None; vg.grad = None
            o = flash_attn_gqa_train(qg, kg, vg, causal=True, slide_size=slide)
            o.backward(do, retain_graph=False)
        try:
            t_tri_fb = time_cuda(run_tri_fb, warmup, rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); t_tri_fb = float('nan')
        except Exception: t_tri_fb = float('nan')

        # SDPA fwd+bwd
        qs = q.detach().clone().requires_grad_(True)
        ks = k.detach().clone().requires_grad_(True)
        vs = v.detach().clone().requires_grad_(True)
        def run_sdpa_fb():
            qs.grad = None; ks.grad = None; vs.grad = None
            o = attention_gqa_ref(qs, ks, vs, causal=True)
            o.backward(do, retain_graph=False)
        try:
            t_sdpa_fb = time_cuda(run_sdpa_fb, warmup, rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); t_sdpa_fb = float('nan')
        except Exception: t_sdpa_fb = float('nan')

        # Derive bwd-only by subtraction
        t_tri_bwd = t_tri_fb - t_tri_fwd if t_tri_fb==t_tri_fb and t_tri_fwd==t_tri_fwd else float('nan')
        t_sdpa_bwd = t_sdpa_fb - t_sdpa_fwd if t_sdpa_fb==t_sdpa_fb and t_sdpa_fwd==t_sdpa_fwd else float('nan')

        def s(a, b):
            if a==a and b==b and b > 0: return a/b
            return float('nan')

        sp_fwd = s(t_sdpa_fwd, t_tri_fwd)
        sp_fb = s(t_sdpa_fb, t_tri_fb)
        sp_bwd = s(t_sdpa_bwd, t_tri_bwd)

        def fmt(x, w):
            if x!=x: return f"{'OOM':>{w}}"
            return f"{x:>{w}.3f}"
        def fmt_sp(x, w):
            if x!=x: return f"{'—':>{w}}"
            return f"{x:>{w-1}.2f}x"

        print(f"{N:>6} | {fmt(t_tri_fwd,8)} | {fmt(t_sdpa_fwd,9)} | {fmt_sp(sp_fwd,7)} |"
              f" {fmt(t_tri_fb,8)} | {fmt(t_sdpa_fb,10)} | {fmt_sp(sp_fb,7)} |"
              f" {fmt(t_tri_bwd,8)} | {fmt(t_sdpa_bwd,9)} | {fmt_sp(sp_bwd,7)}")

        rows.append({"N": N, "tri_fwd": t_tri_fwd, "sdpa_fwd": t_sdpa_fwd,
                     "tri_fb": t_tri_fb, "sdpa_fb": t_sdpa_fb,
                     "tri_bwd": t_tri_bwd, "sdpa_bwd": t_sdpa_bwd,
                     "sp_fwd": sp_fwd, "sp_fb": sp_fb, "sp_bwd": sp_bwd})

        del q, k, v, qg, kg, vg, qs, ks, vs, do
        torch.cuda.empty_cache()

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Tier-1 pre-commit mode: only N=1K/4K/16K, ~2 min")
    args = p.parse_args()

    all_N = [1024, 4096, 16384] if args.quick else [1024, 2048, 4096, 8192, 16384, 32768]
    tag = "attn_only_all_n_quick" if args.quick else "attn_only_all_n"

    results = {}
    results["A_syn_slide1024"] = bench_config(
        "A. Gemma4 synthetic sliding  (slide=1024, D=256, GQA 2:1)",
        1, 32, 16, 256, 1024, all_N)
    results["B_e2b_slide512"] = bench_config(
        "B. Gemma-4-E2B real sliding  (slide=512,  D=256, GQA 8:1)",
        1, 8, 1, 256, 512, all_N)
    results["F_gemma4_full_D512"] = bench_config(
        "F. Gemma4 full-causal       (slide=0,    D=512, GQA 8:1)",
        1, 32, 4, 512, 0, all_N)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{tag}.json")
    with open(out, "w") as f:
        def sanitize(obj):
            if isinstance(obj, float) and (obj != obj or obj == float('inf')): return None
            if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [sanitize(x) for x in obj]
            return obj
        json.dump(sanitize(results), f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
