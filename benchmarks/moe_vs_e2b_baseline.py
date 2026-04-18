"""MoE vs E2B baseline: kernel-level attention benchmark on the exact attention
shapes of both Gemma-4 variants.

Purpose: identify bottlenecks specific to MoE shapes (smaller H_Q → fewer
programs; slide=1024 → different mask profile) before tuning.

Configs (per real HF config.json):
  E2B_slide:  H_Q=32, H_KV=16, D=256, slide= 512   (GQA 2:1, 29 layers of 35)
  E2B_full:   H_Q=32, H_KV= 4, D=512, slide=   0   (GQA 8:1,  6 layers of 35)
  MoE_slide:  H_Q=16, H_KV= 8, D=256, slide=1024   (GQA 2:1, 24 layers of 30)
  MoE_full:   H_Q=16, H_KV= 2, D=512, slide=   0   (GQA 8:1,  6 layers of 30)

Reports Triton fwd / fwd+bwd ms, TFLOPS, and MoE-vs-E2B ms ratio so the hot
config jumps out.
"""
import argparse
import json
import os
import sys

import torch

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
    return ts[len(ts) // 2]


def dense_causal_flops(B, H_Q, N, D):
    # 2·B·H·N²·D fwd (QK^T + PV), 7·B·H·N²·D fwd+bwd (fwd + dQ + dK + dV ≈ 5x fwd matmul)
    return 2 * B * H_Q * N * N * D, 7 * B * H_Q * N * N * D


def bench_config(tag, B, H_Q, H_KV, D, slide, N_list, dtype=torch.float16):
    print(f"\n{'=' * 118}")
    print(f"  {tag}")
    print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, D={D}, slide={slide}, dtype={dtype}")
    print('=' * 118)
    print(f"{'N':>6} | {'Tri fwd':>9} | {'TFLOPS':>8} | {'Sdpa fwd':>9} | {'fwd sp':>7} |"
          f" {'Tri F+B':>9} | {'TFLOPS':>8} | {'Sdpa F+B':>9} | {'F+B sp':>7}")
    print('-' * 118)

    rows = []
    device = "cuda"
    for N in N_list:
        torch.manual_seed(0)
        if N <= 1024: warmup, rep = 20, 100
        elif N <= 4096: warmup, rep = 10, 40
        elif N <= 16384: warmup, rep = 5, 15
        else: warmup, rep = 3, 8

        q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

        def run_tri_fwd():
            flash_attn_gqa_train(q, k, v, causal=True, slide_size=slide)
        try: t_tri_fwd = time_cuda(run_tri_fwd, warmup, rep)
        except Exception: t_tri_fwd = float('nan')

        def run_sdpa_fwd():
            attention_gqa_ref(q, k, v, causal=True)
        try: t_sdpa_fwd = time_cuda(run_sdpa_fwd, warmup, rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); t_sdpa_fwd = float('nan')
        except Exception: t_sdpa_fwd = float('nan')

        qg = q.detach().clone().requires_grad_(True)
        kg = k.detach().clone().requires_grad_(True)
        vg = v.detach().clone().requires_grad_(True)
        do = torch.randn_like(q)

        def run_tri_fb():
            qg.grad = None; kg.grad = None; vg.grad = None
            o = flash_attn_gqa_train(qg, kg, vg, causal=True, slide_size=slide)
            o.backward(do, retain_graph=False)
        try: t_tri_fb = time_cuda(run_tri_fb, warmup, rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); t_tri_fb = float('nan')
        except Exception: t_tri_fb = float('nan')

        qs = q.detach().clone().requires_grad_(True)
        ks = k.detach().clone().requires_grad_(True)
        vs = v.detach().clone().requires_grad_(True)

        def run_sdpa_fb():
            qs.grad = None; ks.grad = None; vs.grad = None
            o = attention_gqa_ref(qs, ks, vs, causal=True)
            o.backward(do, retain_graph=False)
        try: t_sdpa_fb = time_cuda(run_sdpa_fb, warmup, rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); t_sdpa_fb = float('nan')
        except Exception: t_sdpa_fb = float('nan')

        fwd_flops, fb_flops = dense_causal_flops(B, H_Q, N, D)
        tri_fwd_tflops = fwd_flops / (t_tri_fwd * 1e-3) / 1e12 if t_tri_fwd == t_tri_fwd else float('nan')
        tri_fb_tflops = fb_flops / (t_tri_fb * 1e-3) / 1e12 if t_tri_fb == t_tri_fb else float('nan')

        sp_fwd = (t_sdpa_fwd / t_tri_fwd) if t_tri_fwd == t_tri_fwd and t_sdpa_fwd == t_sdpa_fwd else float('nan')
        sp_fb = (t_sdpa_fb / t_tri_fb) if t_tri_fb == t_tri_fb and t_sdpa_fb == t_sdpa_fb else float('nan')

        def fmt(x, w, spec=".3f"):
            if x != x: return f"{'—':>{w}}"
            return f"{x:>{w}{spec}}"
        def fmt_sp(x, w):
            if x != x: return f"{'—':>{w}}"
            return f"{x:>{w-1}.2f}x"

        print(f"{N:>6} | {fmt(t_tri_fwd,9)} | {fmt(tri_fwd_tflops,8,'.1f')} | {fmt(t_sdpa_fwd,9)} | {fmt_sp(sp_fwd,7)} |"
              f" {fmt(t_tri_fb,9)} | {fmt(tri_fb_tflops,8,'.1f')} | {fmt(t_sdpa_fb,9)} | {fmt_sp(sp_fb,7)}")

        rows.append({
            "N": N,
            "tri_fwd_ms": t_tri_fwd, "sdpa_fwd_ms": t_sdpa_fwd,
            "tri_fb_ms": t_tri_fb, "sdpa_fb_ms": t_sdpa_fb,
            "tri_fwd_tflops": tri_fwd_tflops, "tri_fb_tflops": tri_fb_tflops,
            "sp_fwd": sp_fwd, "sp_fb": sp_fb,
        })

        del q, k, v, qg, kg, vg, qs, ks, vs, do
        torch.cuda.empty_cache()

    return rows


def compare(rows_e2b, rows_moe, tag):
    """Print MoE/E2B ms ratio + per-query-head throughput ratio."""
    print(f"\n{'-' * 72}")
    print(f"  MoE vs E2B: {tag}")
    print(f"  (ratio = MoE_ms / E2B_ms. Scaled ratio divides by H_Q ratio to")
    print(f"   isolate per-program efficiency: scaled > 1 ⇒ MoE slower per Q-head)")
    print('-' * 72)
    print(f"{'N':>6} | {'E2B fwd':>9} | {'MoE fwd':>9} | {'ratio':>7} | {'scaled':>7} |"
          f" {'E2B F+B':>9} | {'MoE F+B':>9} | {'ratio':>7} | {'scaled':>7}")
    print('-' * 72)
    # E2B has H_Q=32, MoE has H_Q=16 → MoE does half the work per layer at same N
    # So raw ms ratio of 0.5 means "exactly linear scaling, no inefficiency"
    # Scaled ratio normalizes to per-Q-head, where 1.0 = equal efficiency
    scale = 1.0 / 0.5  # H_Q_E2B / H_Q_MoE = 32/16 = 2.0
    for r_e, r_m in zip(rows_e2b, rows_moe):
        assert r_e["N"] == r_m["N"]
        def r(a, b):
            if a != a or b != b or b == 0: return float('nan')
            return a / b
        fwd_ratio = r(r_m["tri_fwd_ms"], r_e["tri_fwd_ms"])
        fb_ratio = r(r_m["tri_fb_ms"], r_e["tri_fb_ms"])
        def f(x, w, spec=".3f"):
            return f"{'—':>{w}}" if x != x else f"{x:>{w}{spec}}"
        print(f"{r_e['N']:>6} | {f(r_e['tri_fwd_ms'],9)} | {f(r_m['tri_fwd_ms'],9)} | "
              f"{f(fwd_ratio,7,'.2f')} | {f(fwd_ratio * scale,7,'.2f')} |"
              f" {f(r_e['tri_fb_ms'],9)} | {f(r_m['tri_fb_ms'],9)} | "
              f"{f(fb_ratio,7,'.2f')} | {f(fb_ratio * scale,7,'.2f')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Short N list, ~2 min total")
    args = p.parse_args()

    all_N = [1024, 4096, 16384] if args.quick else [1024, 2048, 4096, 8192, 16384, 32768]

    results = {}
    results["E2B_slide"] = bench_config(
        "E2B sliding  (H_Q=32, H_KV=16, D=256, slide=512,  GQA 2:1)",
        1, 32, 16, 256, 512, all_N)
    results["E2B_full"] = bench_config(
        "E2B full     (H_Q=32, H_KV= 4, D=512, slide=  0,  GQA 8:1)",
        1, 32, 4, 512, 0, all_N)
    results["MoE_slide"] = bench_config(
        "MoE sliding  (H_Q=16, H_KV= 8, D=256, slide=1024, GQA 2:1)",
        1, 16, 8, 256, 1024, all_N)
    results["MoE_full"] = bench_config(
        "MoE full     (H_Q=16, H_KV= 2, D=512, slide=  0,  GQA 8:1)",
        1, 16, 2, 512, 0, all_N)

    compare(results["E2B_slide"], results["MoE_slide"], "sliding (D=256, GQA 2:1)")
    compare(results["E2B_full"], results["MoE_full"], "full    (D=512, GQA 8:1)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "moe_vs_e2b_baseline_quick.json" if args.quick else "moe_vs_e2b_baseline.json")
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
