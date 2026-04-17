"""Experiment 4: split dKV packed kernel into dV-only + dK-only kernels.

Hypothesis: dV computation needs no V/delta (2 matmuls, 1 accumulator).
Halving the live state per kernel should cut register spills substantially.
Cost: extra kernel launch + K loaded twice.

Tests at N=4K / 8K, Gemma4 D=512 config. Reports spills/shmem/time and
validates dK, dV match the packed reference bit-exactly (or within f32 noise).
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
    _flash_attn_gqa_bwd_dv_only_kernel,
    _flash_attn_gqa_bwd_dk_only_kernel,
    _delta_kernel,
)


def get_all_compiled(jit_fn):
    binder_cache, _, _, _, _ = jit_fn.device_caches[0]
    return list(binder_cache.values())


def kernel_stats(jit_fn, w, s):
    """Find the compiled variant matching (num_warps, num_stages)."""
    cks = [ck for ck in get_all_compiled(jit_fn) if hasattr(ck, "n_regs")]
    matched = None
    for ck in cks:
        m = ck.metadata
        nw = getattr(m, "num_warps", None)
        ns = getattr(m, "num_stages", None)
        if nw is None and hasattr(m, "get"):
            nw, ns = m.get("num_warps"), m.get("num_stages")
        if nw == w and ns == s:
            matched = ck
    if matched is None and cks:
        matched = cks[-1]
    if matched is None:
        return 0, 0, 0.0
    shmem = getattr(matched.metadata, "shared", None) or 0
    return matched.n_regs, matched.n_spills, shmem / 1024


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
    dtype = torch.float16
    device = "cuda"
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


def bench_packed(q, k, v, o, lse, do, delta, H_Q, H_KV, N, D,
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
    t = time_cuda(run, warmup, rep)
    regs, spills, shm = kernel_stats(_flash_attn_gqa_bwd_dkv_packed_kernel, w, s)
    return {"ms": t, "regs": regs, "spills": spills, "shmem_KB": shm, "dk": dk, "dv": dv}


def bench_dv_only(q, k, o, lse, do, H_Q, H_KV, N, D,
                  BQ, BKV, w, s, causal=True, warmup=8, rep=20):
    GQA_RATIO = H_Q // H_KV
    grid = (triton.cdiv(N, BKV), 1 * H_KV)
    dv = torch.empty_like(k)  # same shape as V which equals K shape
    def run():
        _flash_attn_gqa_bwd_dv_only_kernel[grid](
            q, k, do, dv, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=1.0 / math.sqrt(D),
            BLOCK_Q=BQ, BLOCK_KV=BKV, GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=causal, SLIDE_SIZE=0,
            num_warps=w, num_stages=s,
        )
    t = time_cuda(run, warmup, rep)
    regs, spills, shm = kernel_stats(_flash_attn_gqa_bwd_dv_only_kernel, w, s)
    return {"ms": t, "regs": regs, "spills": spills, "shmem_KB": shm, "dv": dv}


def bench_dk_only(q, k, v, o, lse, do, delta, H_Q, H_KV, N, D,
                  BQ, BKV, w, s, causal=True, warmup=8, rep=20):
    GQA_RATIO = H_Q // H_KV
    grid = (triton.cdiv(N, BKV), 1 * H_KV)
    dk = torch.empty_like(k)
    def run():
        _flash_attn_gqa_bwd_dk_only_kernel[grid](
            q, k, v, do, dk, lse, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=1.0 / math.sqrt(D),
            BLOCK_Q=BQ, BLOCK_KV=BKV, GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=causal, SLIDE_SIZE=0,
            num_warps=w, num_stages=s,
        )
    t = time_cuda(run, warmup, rep)
    regs, spills, shm = kernel_stats(_flash_attn_gqa_bwd_dk_only_kernel, w, s)
    return {"ms": t, "regs": regs, "spills": spills, "shmem_KB": shm, "dk": dk}


def main():
    B, H_Q, H_KV, D = 1, 32, 4, 512
    # Test baseline block config from attention.py: BQ=64, BKV=16, w=4, s=2.
    # Also try a few dV-only specific configs since its live state is smaller.
    cases = [
        # (packed_cfg, dv_cfg, dk_cfg, label)
        ((64, 16, 4, 2), (64, 16, 4, 2), (64, 16, 4, 2), "split same as packed default"),
        ((64, 16, 4, 2), (64, 16, 4, 3), (64, 16, 4, 2), "split, dv s=3 (more pipeline)"),
        ((64, 16, 4, 2), (64, 32, 4, 2), (64, 16, 4, 2), "split, dv BKV=32 (bigger tile)"),
        ((64, 16, 4, 2), (128, 16, 4, 2), (64, 16, 4, 2), "split, dv BQ=128"),
    ]
    for N in [4096, 8192]:
        print(f"\n============ N={N} ============")
        q, k, v, o, lse, do, delta = setup(B, H_Q, H_KV, N, D, causal=True)
        print(f"{'label':>34} | {'time_ms':>8} | {'regs':>4} | {'spills':>6} | {'shmem':>8} | vs packed")
        print("-" * 120)
        # Packed baseline
        (BQp, BKVp, wp, sp) = (64, 16, 4, 2)
        r_pack = bench_packed(q, k, v, o, lse, do, delta, H_Q, H_KV, N, D,
                              BQp, BKVp, wp, sp, causal=True, warmup=8, rep=20)
        print(f"{'PACKED (baseline)':>34} | {r_pack['ms']:>8.3f} | {r_pack['regs']:>4} | {r_pack['spills']:>6} | "
              f"{r_pack['shmem_KB']:>6.1f}KB | — (baseline)")
        baseline_ms = r_pack["ms"]

        for pack_cfg, dv_cfg, dk_cfg, label in cases:
            if pack_cfg != (64, 16, 4, 2):
                continue
            try:
                BQ, BKV, w, s = dv_cfg
                r_dv = bench_dv_only(q, k, o, lse, do, H_Q, H_KV, N, D,
                                     BQ, BKV, w, s, causal=True, warmup=8, rep=20)
                BQ, BKV, w, s = dk_cfg
                r_dk = bench_dk_only(q, k, v, o, lse, do, delta, H_Q, H_KV, N, D,
                                     BQ, BKV, w, s, causal=True, warmup=8, rep=20)
            except Exception as e:
                msg = type(e).__name__ + ": " + str(e).splitlines()[0][:80]
                print(f"{label:>34} | FAIL — {msg}")
                continue
            tot = r_dv["ms"] + r_dk["ms"]
            pct = 100 * (tot - baseline_ms) / baseline_ms
            print(f"{'  dV ' + label:>34} | {r_dv['ms']:>8.3f} | {r_dv['regs']:>4} | {r_dv['spills']:>6} | "
                  f"{r_dv['shmem_KB']:>6.1f}KB | ")
            print(f"{'  dK ' + label:>34} | {r_dk['ms']:>8.3f} | {r_dk['regs']:>4} | {r_dk['spills']:>6} | "
                  f"{r_dk['shmem_KB']:>6.1f}KB | ")
            print(f"{'  [total dV+dK]':>34} | {tot:>8.3f} |      |        |          | {pct:+.1f}% ({tot/baseline_ms:.3f}x)")
            dv_diff = (r_pack["dv"] - r_dv["dv"]).abs().max().item()
            dk_diff = (r_pack["dk"] - r_dk["dk"]).abs().max().item()
            print(f"{'  correctness':>34} | dK max = {dk_diff:.2e}, dV max = {dv_diff:.2e}")


if __name__ == "__main__":
    main()
