"""End-to-end SWA fwd+bwd benchmark vs SDPA full-causal.

Used to confirm whether dKV default change ships a net speedup on the
Gemma-4-E2B real sliding config (D=256, H_Q=8, H_KV=1, GQA 8:1).

Run before/after dKV default change and diff the tables.
"""
import math
import os
import sys
import json

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flash_attn.attention import flash_attn_gqa_train, attention_gqa_ref


def time_cuda(fn, warmup=5, rep=20):
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


def run(B, H_Q, H_KV, N, D, slide_size, dtype=torch.float16, warmup=5, rep=15):
    device = "cuda"
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

    o_ref = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    do = torch.randn_like(o_ref)

    # Triton fwd only (median)
    def run_tri_fwd():
        flash_attn_gqa_train(q, k, v, causal=True, slide_size=slide_size)
    t_tri_fwd = time_cuda(run_tri_fwd, warmup, rep)

    # Triton fwd+bwd
    qg = q.detach().clone().requires_grad_(True)
    kg = k.detach().clone().requires_grad_(True)
    vg = v.detach().clone().requires_grad_(True)
    def run_tri_fwdbwd():
        qg.grad = None; kg.grad = None; vg.grad = None
        out = flash_attn_gqa_train(qg, kg, vg, causal=True, slide_size=slide_size)
        out.backward(do, retain_graph=False)
    t_tri_fwdbwd = time_cuda(run_tri_fwdbwd, warmup, rep)

    # SDPA fwd only (full-causal reference — SDPA has no native SWA)
    def run_sdpa_fwd():
        attention_gqa_ref(q, k, v, causal=True)
    t_sdpa_fwd = time_cuda(run_sdpa_fwd, warmup=3, rep=min(rep, 10))

    qs = q.detach().clone().requires_grad_(True)
    ks = k.detach().clone().requires_grad_(True)
    vs = v.detach().clone().requires_grad_(True)
    def run_sdpa_fwdbwd():
        qs.grad = None; ks.grad = None; vs.grad = None
        out = attention_gqa_ref(qs, ks, vs, causal=True)
        out.backward(do, retain_graph=False)
    try:
        t_sdpa_fwdbwd = time_cuda(run_sdpa_fwdbwd, warmup=3, rep=min(rep, 8))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        t_sdpa_fwdbwd = float("nan")

    return {
        "N": N, "slide": slide_size,
        "tri_fwd_ms": t_tri_fwd, "sdpa_fwd_ms": t_sdpa_fwd,
        "tri_fwdbwd_ms": t_tri_fwdbwd, "sdpa_fwdbwd_ms": t_sdpa_fwdbwd,
        "fwd_speedup": t_sdpa_fwd / t_tri_fwd,
        "fwdbwd_speedup": (t_sdpa_fwdbwd / t_tri_fwdbwd) if not math.isnan(t_sdpa_fwdbwd) else float("nan"),
    }


def main():
    # Gemma-4-E2B real sliding config
    B, H_Q, H_KV, D = 1, 8, 1, 256

    print(f"B={B} H_Q={H_Q} H_KV={H_KV} D={D} fp16 causal=True (SDPA reference: full-causal)")
    print(f"{'N':>6} {'slide':>5} | {'tri_fwd':>8} {'sdpa_fwd':>9} {'fwd_sp':>7} | "
          f"{'tri_fwdbwd':>11} {'sdpa_fwdbwd':>12} {'fwdbwd_sp':>10}")
    print("-" * 90)

    results = []
    configs = []
    for slide in [512, 1024]:
        for N in [2048, 4096, 8192, 16384, 32768]:
            configs.append((N, slide))

    for N, slide in configs:
        if N <= 4096:
            warmup, rep = 10, 20
        elif N <= 16384:
            warmup, rep = 5, 15
        else:
            warmup, rep = 3, 10
        r = run(B, H_Q, H_KV, N, D, slide, warmup=warmup, rep=rep)
        print(f"{r['N']:>6} {r['slide']:>5} | {r['tri_fwd_ms']:>8.3f} {r['sdpa_fwd_ms']:>9.3f} "
              f"{r['fwd_speedup']:>6.2f}x | {r['tri_fwdbwd_ms']:>11.3f} "
              f"{r['sdpa_fwdbwd_ms'] if not math.isnan(r['sdpa_fwdbwd_ms']) else 'OOM':>12} "
              f"{r['fwdbwd_speedup']:>9.2f}x")
        results.append(r)

    label = os.environ.get("LABEL", "result")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"swa_e2e_bench_{label}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
