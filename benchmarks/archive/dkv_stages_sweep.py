"""Experiment 1: dKV num_stages 2 → 1 at default (BKV=16, BQ=64, w=4), D=512.

Hypothesis: current dKV spills 302 values (dump_kernel_regs.py). Pipeline
stages=2 needs 2× live buffers → register pressure. Dropping to stages=1
may free register budget and kill spills.

Risk: stages=1 gives up memory latency hiding. If HBM-bound dominates, we
lose; if spill-to-local-memory dominates, we win.

Reports per (N, stages): n_regs, n_spills, shmem_KB, kernel ms, vs s=2.
"""
import math
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flash_attn.attention import (
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
    _flash_attn_gqa_bwd_dq_kernel,
    _delta_kernel,
)


def get_all_compiled(jit_fn):
    """Return list of CompiledKernel objects in device_caches[0]."""
    binder_cache, _, _, _, _ = jit_fn.device_caches[0]
    return list(binder_cache.values())


def time_cuda(fn, warmup=10, rep=30):
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


def setup(B, H_Q, H_KV, N, D, causal=True):
    """Run fwd + delta + dQ to produce valid tensors for dKV test."""
    device = "cuda"
    dtype = torch.float16
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    o = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    BQ_F = min(64, triton.next_power_of_2(N))
    BKV_F = min(32, triton.next_power_of_2(N))
    grid_f = (triton.cdiv(N, BQ_F), B * H_Q)
    _flash_attn_gqa_kernel[grid_f](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ_F, BLOCK_KV=BKV_F, BLOCK_D=D,
        IS_CAUSAL=causal, SLIDE_SIZE=0,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=True,
        num_warps=8, num_stages=2,
    )
    do = torch.randn_like(o)
    delta = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    _delta_kernel[(N, B * H_Q)](
        do, o, delta,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_HEADS=H_Q, SEQ_LEN=N, HEAD_DIM=D,
        num_warps=4, num_stages=2,
    )
    torch.cuda.synchronize()
    return q, k, v, o, lse, do, delta


def bench_dkv(q, k, v, o, lse, do, delta, H_Q, H_KV, N, D,
              BQ, BKV, w, s, causal=True, warmup=8, rep=20):
    GQA_RATIO = H_Q // H_KV
    grid = (triton.cdiv(N, BKV), 1 * H_KV)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    def run():
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
            BLOCK_Q=BQ, BLOCK_KV=BKV, GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=causal, SLIDE_SIZE=0,
            Q_SPLITS=1,
            num_warps=w, num_stages=s,
        )
    t = time_cuda(run, warmup=warmup, rep=rep)
    # Compiled metadata: find the kernel variant with matching constexpr tuple.
    # Easiest: use the last entry (just compiled).
    cks = get_all_compiled(_flash_attn_gqa_bwd_dkv_packed_kernel)
    # Filter by num_warps and num_stages in metadata
    matched = None
    for ck in cks:
        m = ck.metadata
        nw = getattr(m, "num_warps", None) or (m.get("num_warps") if hasattr(m, "get") else None)
        ns = getattr(m, "num_stages", None) or (m.get("num_stages") if hasattr(m, "get") else None)
        if nw == w and ns == s:
            matched = ck
    if matched is None:
        matched = cks[-1]
    shmem = getattr(matched.metadata, "shared", None)
    if shmem is None and hasattr(matched.metadata, "get"):
        shmem = matched.metadata.get("shared", 0)
    return {
        "ms": t,
        "regs": matched.n_regs,
        "spills": matched.n_spills,
        "shmem_KB": shmem / 1024 if shmem else 0.0,
        "cfg": f"BQ={BQ}, BKV={BKV}, w={w}, s={s}",
        "dk": dk, "dv": dv,
    }


def main():
    B, H_Q, H_KV, D = 1, 32, 4, 512
    Ns = [4096, 8192]  # 8K is target; 4K as sanity
    configs = [
        # (BQ, BKV, warps, stages, label)
        # Experiment 3: GQA loop dynamic (no unroll). Re-test the space that
        # was infeasible before due to 8× code duplication.
        (64, 16, 4, 2, "BQ=64 w=4 s=2 (dyn-gqa)"),
        (64, 16, 4, 1, "BQ=64 w=4 s=1 (dyn-gqa)"),
        (64, 16, 4, 3, "BQ=64 w=4 s=3 (dyn-gqa)"),
        (64, 16, 8, 2, "BQ=64 w=8 s=2 (dyn-gqa)"),
        (32, 16, 4, 2, "BQ=32 w=4 s=2 (dyn-gqa)"),
        (32, 16, 4, 1, "BQ=32 w=4 s=1 (dyn-gqa)"),
        (32, 16, 8, 2, "BQ=32 w=8 s=2 (dyn-gqa)"),
    ]

    for N in Ns:
        print(f"\n============ N={N} ============")
        q, k, v, o, lse, do, delta = setup(B, H_Q, H_KV, N, D, causal=True)
        print(f"{'label':>28} | {'time_ms':>8} | {'regs':>4} | {'spills':>6} | {'shmem':>8} | vs baseline")
        print("-" * 100)
        baseline = None
        saved = {}
        for BQ, BKV, w, s, label in configs:
            try:
                r = bench_dkv(q, k, v, o, lse, do, delta,
                              H_Q, H_KV, N, D,
                              BQ=BQ, BKV=BKV, w=w, s=s,
                              causal=True, warmup=8, rep=20)
            except Exception as e:
                msg = type(e).__name__ + ": " + str(e).splitlines()[0][:80]
                print(f"{label:>28} | FAIL — {msg}")
                continue
            if baseline is None:
                baseline = r["ms"]
                delta_str = "— (baseline)"
            else:
                pct = 100 * (r["ms"] - baseline) / baseline
                delta_str = f"{pct:+.1f}% ({r['ms']/baseline:.3f}x)"
            print(f"{label:>28} | {r['ms']:>8.3f} | {r['regs']:>4} | {r['spills']:>6} | "
                  f"{r['shmem_KB']:>6.1f}KB | {delta_str}")
            saved[label] = r
        # Correctness: each feasible variant must match the first baseline
        r_base = next(iter(saved.values())) if saved else None
        base_label = next(iter(saved.keys())) if saved else None
        if r_base is not None:
            for label, r in saved.items():
                if label == base_label:
                    continue
                dk_diff = (r_base["dk"] - r["dk"]).abs().max().item()
                dv_diff = (r_base["dv"] - r["dv"]).abs().max().item()
                print(f"{'correctness':>28} | {label}: dK max = {dk_diff:.2e}, dV max = {dv_diff:.2e}")


if __name__ == "__main__":
    main()
