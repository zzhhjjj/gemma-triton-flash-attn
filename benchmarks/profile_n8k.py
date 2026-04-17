"""Detailed profiling for D=512 N=8K full causal (Gemma4 full config).

Goal: find the *actual* hotspot across fwd and bwd now that many rounds of
optimization have landed. Reports, per kernel:
  - elapsed ms (CUDA events, median)
  - theoretical FLOPs (causal convention)
  - achieved TFLOPS, MFU
  - theoretical HBM traffic (two bounds: naive no-reuse / perfect-reuse)
  - arithmetic intensity
  - BW utilization (both bounds)
  - rank / percent of total

Config: B=1, H_Q=32, H_KV=4, D=512, fp16, causal=True.
Same measurement setup as bwd_breakdown.py — this file adds N=8K + fwd
analysis + a unified hotspot summary.
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
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dq_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
    flash_attn_gqa_train,
    attention_gqa_ref,
)

# H100 peak specs
HBM_GB_PER_S = 3350.0  # 3.35 TB/s
FP16_TFLOPS = 989.5    # tensor core peak


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


def causal_fwd_flops(B, H_Q, N, D):
    # Two matmuls (QK^T + PV), each roughly half-N triangle.
    # Total FLOPs ≈ 2·B·H_Q·N²·D (same convention as baseline.md).
    return 2.0 * B * H_Q * N * N * D


def causal_bwd_flops(B, H_Q, N, D):
    # Convention: bwd ≈ 2× fwd (dS/dP + dQ + dK + dV matmuls)
    return 4.0 * B * H_Q * N * N * D


def fwd_hbm_bytes(B, H_Q, H_KV, N, D):
    """Two bounds on HBM traffic for fwd.

    perfect (lower bound): each tensor read/written exactly once.
      Q read + O write = 2 * B * H_Q * N * D * 2
      K read + V read  = 2 * B * H_KV * N * D * 2
      LSE write        = B * H_Q * N * 4
    naive  (upper bound): no L2 reuse at all. K/V loaded once per Q head.
      K+V traffic multiplied by GQA_RATIO = H_Q/H_KV.
    """
    bytes_fp16 = 2
    qo = 2 * B * H_Q * N * D * bytes_fp16
    kv_perfect = 2 * B * H_KV * N * D * bytes_fp16
    kv_naive = 2 * B * H_Q * N * D * bytes_fp16
    lse = B * H_Q * N * 4
    return {
        "perfect": qo + kv_perfect + lse,
        "naive":   qo + kv_naive + lse,
    }


def dq_hbm_bytes(B, H_Q, H_KV, N, D):
    """dQ kernel traffic bounds.

    Reads: Q, K, V, dO, LSE, delta. Writes: dQ.
    Q, dO, dQ indexed by H_Q. K, V indexed by H_KV.
    """
    f = 2
    q_like = B * H_Q * N * D * f  # Q / dO / dQ each
    kv_perfect = 2 * B * H_KV * N * D * f
    kv_naive = 2 * B * H_Q * N * D * f
    lse_delta = 2 * B * H_Q * N * 4  # lse + delta, fp32
    return {
        "perfect": 3 * q_like + kv_perfect + lse_delta,
        "naive":   3 * q_like + kv_naive + lse_delta,
    }


def dkv_hbm_bytes(B, H_Q, H_KV, N, D):
    """Pack-GQA dKV kernel traffic.

    Each program owns one KV tile, loops all GQA_RATIO Q heads.
    Reads: Q, K, V, dO, LSE, delta. Writes: dK, dV.
    K/V loaded ONCE per program (perfect for pack-gqa). Q/dO must be loaded
    per Q head (GQA_RATIO times in worst case, or once with L2 reuse across
    KV tiles in the same group).
    """
    f = 2
    kv = 2 * B * H_KV * N * D * f  # K+V read
    dkv_write = 2 * B * H_KV * N * D * f
    # Q and dO: loaded at least once per (B, H_Q) position
    q_do_perfect = 2 * B * H_Q * N * D * f
    # Worst case: each KV tile loads Q+dO fully — N/BLOCK_KV × BLOCK_Q × D × H_Q
    # but Q loop sees BLOCK_Q stride; the inner loop re-traverses all Q blocks
    # per KV tile for this H_KV head. We use q_do_perfect as lower bound.
    lse_delta = 2 * B * H_Q * N * 4
    return {
        "perfect": kv + dkv_write + q_do_perfect + lse_delta,
        "naive":   kv + dkv_write + q_do_perfect + lse_delta,
    }


def delta_hbm_bytes(B, H_Q, N, D):
    f = 2
    # Read do + o, write delta (fp32)
    return 2 * B * H_Q * N * D * f + B * H_Q * N * 4


def analyze_kernel(name, t_ms, flops, hbm):
    """Derive TFLOPS, BW util, arithmetic intensity."""
    tflops = flops / (t_ms * 1e-3) / 1e12
    mfu = tflops / FP16_TFLOPS * 100
    out = {"name": name, "time_ms": t_ms, "flops_G": flops / 1e9,
           "tflops": tflops, "mfu_%": mfu}
    if isinstance(hbm, dict):
        for k, bytes_ in hbm.items():
            gbs = bytes_ / (t_ms * 1e-3) / 1e9
            out[f"bw_{k}_GB/s"] = gbs
            out[f"bw_{k}_util_%"] = gbs / HBM_GB_PER_S * 100 * 1000 / 1000
            out[f"ai_{k}_F/B"] = flops / bytes_ if bytes_ > 0 else float("nan")
    else:
        gbs = hbm / (t_ms * 1e-3) / 1e9
        out["bw_GB/s"] = gbs
        out["bw_util_%"] = gbs / HBM_GB_PER_S * 100
        out["ai_F/B"] = flops / hbm
    return out


def run(N, B=1, H_Q=32, H_KV=4, D=512, causal=True, slide_size=0,
        warmup=10, rep=30):
    device = "cuda"
    dtype = torch.float16
    GQA_RATIO = H_Q // H_KV

    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

    # ----- Forward kernel standalone -----
    o = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    BLOCK_Q_F = min(64 if D >= 512 else 128, triton.next_power_of_2(N))
    BLOCK_KV_F = min(32 if D >= 512 else 64, triton.next_power_of_2(N))
    num_warps_F = 8 if D >= 256 else 4
    grid_f = (triton.cdiv(N, BLOCK_Q_F), B * H_Q)
    scale = 1.0 / math.sqrt(D)

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
            IS_CAUSAL=causal,
            SLIDE_SIZE=slide_size,
            LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
            stride_lsen=lse.stride(2), STORE_LSE=True,
            num_warps=num_warps_F, num_stages=2,
        )
    t_fwd = time_cuda(run_fwd_kernel, warmup, rep)

    # SDPA fwd reference
    def run_fwd_sdpa():
        attention_gqa_ref(q, k, v, causal=causal)
    t_sdpa_fwd = time_cuda(run_fwd_sdpa, warmup=5, rep=min(rep, 20))

    # Run fwd once to populate o / lse for bwd kernels
    run_fwd_kernel()
    torch.cuda.synchronize()

    do = torch.randn_like(o)

    # ----- Delta kernel -----
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

    # ----- dQ kernel -----
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
            IS_CAUSAL=causal,
            SLIDE_SIZE=slide_size,
            STORE_DELTA=False,
            num_warps=w_dq, num_stages=2,
        )
    t_dq = time_cuda(run_dq, warmup, rep)

    # ----- dKV packed kernel -----
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
            IS_CAUSAL=causal,
            SLIDE_SIZE=slide_size,
            Q_SPLITS=1,
            num_warps=w_dkv, num_stages=2,
        )
    t_dkv = time_cuda(run_dkv, warmup, rep)

    # ----- Triton fwd+bwd end-to-end (autograd path) -----
    qg = q.detach().clone().requires_grad_(True)
    kg = k.detach().clone().requires_grad_(True)
    vg = v.detach().clone().requires_grad_(True)
    def run_triton_fwdbwd():
        qg.grad = None; kg.grad = None; vg.grad = None
        out = flash_attn_gqa_train(qg, kg, vg, causal=causal, slide_size=slide_size)
        out.backward(do, retain_graph=False)
    t_triton_fwdbwd = time_cuda(run_triton_fwdbwd, warmup, rep)

    # SDPA fwd+bwd
    qs = q.detach().clone().requires_grad_(True)
    ks = k.detach().clone().requires_grad_(True)
    vs = v.detach().clone().requires_grad_(True)
    def run_sdpa_fwdbwd():
        qs.grad = None; ks.grad = None; vs.grad = None
        out = attention_gqa_ref(qs, ks, vs, causal=causal)
        out.backward(do, retain_graph=False)
    try:
        t_sdpa_fwdbwd = time_cuda(run_sdpa_fwdbwd, warmup=3, rep=min(rep, 8))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        t_sdpa_fwdbwd = float("nan")

    # ----- derive analyses -----
    results = {}
    results["fwd"] = analyze_kernel(
        "fwd", t_fwd, causal_fwd_flops(B, H_Q, N, D),
        fwd_hbm_bytes(B, H_Q, H_KV, N, D))
    results["delta"] = analyze_kernel(
        "delta", t_delta, 0.0, delta_hbm_bytes(B, H_Q, N, D))
    results["dQ"] = analyze_kernel(
        "dQ", t_dq, causal_bwd_flops(B, H_Q, N, D) * 0.5,  # roughly half of bwd total
        dq_hbm_bytes(B, H_Q, H_KV, N, D))
    results["dKV"] = analyze_kernel(
        "dKV", t_dkv, causal_bwd_flops(B, H_Q, N, D) * 0.5,
        dkv_hbm_bytes(B, H_Q, H_KV, N, D))

    results["meta"] = {
        "N": N, "B": B, "H_Q": H_Q, "H_KV": H_KV, "D": D,
        "causal": causal, "GQA_RATIO": GQA_RATIO,
        "block_cfg_fwd":  f"BQ={BLOCK_Q_F}, BKV={BLOCK_KV_F}, w={num_warps_F}, BLOCK_D={D}",
        "block_cfg_dQ":   f"BQ={BQ_dq}, BKV={BKV_dq}, w={w_dq}",
        "block_cfg_dKV":  f"BQ={BQ_dkv}, BKV={BKV_dkv}, w={w_dkv}, GQA_RATIO={GQA_RATIO}",
        "t_sdpa_fwd": t_sdpa_fwd,
        "t_triton_fwdbwd": t_triton_fwdbwd,
        "t_sdpa_fwdbwd": t_sdpa_fwdbwd,
        "fwd_speedup": t_sdpa_fwd / t_fwd if t_fwd > 0 else float("nan"),
        "fwdbwd_speedup": (t_sdpa_fwdbwd / t_triton_fwdbwd) if t_triton_fwdbwd > 0 and not math.isnan(t_sdpa_fwdbwd) else float("nan"),
    }
    return results


def pprint_summary(res):
    m = res["meta"]
    print("=" * 90)
    print(f"N={m['N']}  B={m['B']}  H_Q={m['H_Q']}  H_KV={m['H_KV']}  D={m['D']}  causal={m['causal']}")
    print(f"  fwd block   : {m['block_cfg_fwd']}")
    print(f"  dQ block    : {m['block_cfg_dQ']}")
    print(f"  dKV block   : {m['block_cfg_dKV']}")
    print("-" * 90)

    # Per-kernel table
    header = f"{'kernel':>8} | {'time_ms':>8} | {'TFLOPS':>8} | {'MFU%':>6} | {'BW_perfect':>12} | {'BW_naive':>10} | {'AI_perf':>8}"
    print(header)
    print("-" * len(header))
    for name in ["fwd", "delta", "dQ", "dKV"]:
        k = res[name]
        bw_p = k.get("bw_perfect_GB/s", k.get("bw_GB/s", float("nan")))
        bw_p_u = k.get("bw_perfect_util_%", k.get("bw_util_%", float("nan")))
        bw_n = k.get("bw_naive_GB/s", float("nan"))
        bw_n_u = k.get("bw_naive_util_%", float("nan"))
        ai = k.get("ai_perfect_F/B", k.get("ai_F/B", float("nan")))
        print(f"{name:>8} | {k['time_ms']:>8.3f} | {k['tflops']:>8.1f} | {k['mfu_%']:>5.1f}% | "
              f"{bw_p:>6.0f}GB/s ({bw_p_u:>4.0f}%) | {bw_n:>5.0f}GB/s ({bw_n_u:>3.0f}%) | {ai:>7.1f}")
    print("-" * 90)

    # Fwd+Bwd aggregate
    total_bwd_kernels = res["delta"]["time_ms"] + res["dQ"]["time_ms"] + res["dKV"]["time_ms"]
    total_fwdbwd = m["t_triton_fwdbwd"]
    print(f"Fwd (Triton):         {res['fwd']['time_ms']:.3f} ms   vs SDPA {m['t_sdpa_fwd']:.3f} ms   speedup {m['fwd_speedup']:.2f}x")
    print(f"Fwd+Bwd (Triton):     {total_fwdbwd:.3f} ms   vs SDPA {m['t_sdpa_fwdbwd']:.3f} ms   speedup {m['fwdbwd_speedup']:.2f}x")
    print(f"  sum of 3 bwd kernels = {total_bwd_kernels:.3f} ms    ({100*total_bwd_kernels/total_fwdbwd:.1f}% of fwd+bwd)")
    print(f"  fwd kernel inside    = {res['fwd']['time_ms']:.3f} ms ({100*res['fwd']['time_ms']/total_fwdbwd:.1f}% of fwd+bwd)")
    overhead = total_fwdbwd - total_bwd_kernels - res['fwd']['time_ms']
    print(f"  overhead / alloc     = {overhead:.3f} ms ({100*overhead/total_fwdbwd:.1f}%)")
    print("-" * 90)

    # Hotspot ranking
    ranking = sorted([
        ("fwd", res["fwd"]["time_ms"]),
        ("delta", res["delta"]["time_ms"]),
        ("dQ", res["dQ"]["time_ms"]),
        ("dKV", res["dKV"]["time_ms"]),
    ], key=lambda x: -x[1])
    fwdbwd_kernel_total = sum(v for _, v in ranking)
    print("Hotspot rank (by ms, relative to fwd+bwd kernel sum):")
    for i, (n, t) in enumerate(ranking, 1):
        print(f"  #{i}  {n:>5}  {t:>8.3f} ms  ({100*t/fwdbwd_kernel_total:.1f}%)")


def main():
    B, H_Q, H_KV, D = 1, 32, 4, 512
    # N=8K is the requested profiling target.
    # Add N=4K and N=16K as sanity anchors to see if hotspot shifts with N.
    all_results = {}
    for N in [4096, 8192, 16384]:
        if N <= 4096:
            warmup, rep = 10, 30
        elif N <= 8192:
            warmup, rep = 8, 20
        else:
            warmup, rep = 5, 12
        r = run(N, B=B, H_Q=H_Q, H_KV=H_KV, D=D,
                causal=True, slide_size=0,
                warmup=warmup, rep=rep)
        pprint_summary(r)
        all_results[N] = r

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "profile_n8k.json")
    # Make JSON-safe: strip non-scalars
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj
    with open(out_path, "w") as f:
        json.dump(sanitize(all_results), f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
