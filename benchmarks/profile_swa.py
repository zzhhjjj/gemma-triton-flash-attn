"""Detailed profiling for SWA (Sliding Window Attention) across long N.

Gemma-4-E2B sliding config: H_Q=8, H_KV=1 (GQA 8:1), D=256, sliding_window=512.
This script reports per-kernel time / TFLOPS / MFU / speedup for fwd + bwd
(delta, dQ, dKV packed) at N in {8K, 16K, 32K}, slide in {512, 1024}.

Also quantifies split-loop upper bound: fraction of KV blocks per Q block that
are fully in-window AND off-diagonal (would not need a mask op in a split-loop
variant). This tells us the ceiling benefit of extending USE_SPLIT to SWA.
"""
import math
import os
import sys
import json

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

HBM_GB_PER_S = 3350.0
FP16_TFLOPS = 989.5


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


def swa_kv_positions(N, slide):
    """sum_{i=0}^{N-1} min(i+1, slide) — number of (q,k) attend pairs."""
    if slide >= N:
        return N * (N + 1) // 2
    return slide * (slide + 1) // 2 + slide * (N - slide)


def swa_fwd_flops(B, H_Q, N, D, slide):
    pairs = swa_kv_positions(N, slide)
    return 2.0 * B * H_Q * D * pairs


def swa_bwd_flops(B, H_Q, N, D, slide):
    return 2.0 * swa_fwd_flops(B, H_Q, N, D, slide)


def fwd_hbm_bytes(B, H_Q, H_KV, N, D):
    bytes_fp16 = 2
    qo = 2 * B * H_Q * N * D * bytes_fp16
    kv_perfect = 2 * B * H_KV * N * D * bytes_fp16
    kv_naive = 2 * B * H_Q * N * D * bytes_fp16
    lse = B * H_Q * N * 4
    return {"perfect": qo + kv_perfect + lse, "naive": qo + kv_naive + lse}


def dq_hbm_bytes(B, H_Q, H_KV, N, D):
    f = 2
    q_like = B * H_Q * N * D * f
    kv_perfect = 2 * B * H_KV * N * D * f
    kv_naive = 2 * B * H_Q * N * D * f
    lse_delta = 2 * B * H_Q * N * 4
    return {"perfect": 3 * q_like + kv_perfect + lse_delta,
            "naive":   3 * q_like + kv_naive + lse_delta}


def dkv_hbm_bytes(B, H_Q, H_KV, N, D):
    f = 2
    kv = 2 * B * H_KV * N * D * f
    dkv_write = 2 * B * H_KV * N * D * f
    q_do_perfect = 2 * B * H_Q * N * D * f
    lse_delta = 2 * B * H_Q * N * 4
    return {"perfect": kv + dkv_write + q_do_perfect + lse_delta,
            "naive":   kv + dkv_write + q_do_perfect + lse_delta}


def delta_hbm_bytes(B, H_Q, N, D):
    f = 2
    return 2 * B * H_Q * N * D * f + B * H_Q * N * 4


def analyze_kernel(name, t_ms, flops, hbm):
    tflops = flops / (t_ms * 1e-3) / 1e12 if t_ms > 0 else 0.0
    mfu = tflops / FP16_TFLOPS * 100
    out = {"name": name, "time_ms": t_ms, "flops_G": flops / 1e9,
           "tflops": tflops, "mfu_%": mfu}
    if isinstance(hbm, dict):
        for k, bytes_ in hbm.items():
            gbs = bytes_ / (t_ms * 1e-3) / 1e9
            out[f"bw_{k}_GB/s"] = gbs
            out[f"bw_{k}_util_%"] = gbs / HBM_GB_PER_S * 100
            out[f"ai_{k}_F/B"] = flops / bytes_ if bytes_ > 0 else float("nan")
    else:
        gbs = hbm / (t_ms * 1e-3) / 1e9
        out["bw_GB/s"] = gbs
        out["bw_util_%"] = gbs / HBM_GB_PER_S * 100
        out["ai_F/B"] = flops / hbm if hbm > 0 else 0.0
    return out


def split_loop_ceiling(N, slide, BLOCK_Q, BLOCK_KV):
    """Fraction of KV iterations that would be fully unmasked under a SWA split-loop.

    For Q block starting at q_s (size BQ), a KV block at kv_s (size BKV) is
    fully in-window AND off-diagonal iff:
      - causal: (kv_s + BKV - 1) <= q_s         -> kv_s <= q_s - BKV + 1
      - window: (q_s + BQ - 1) - kv_s < slide   -> kv_s > q_s + BQ - 1 - slide
    Unmasked kv_s range: (q_s + BQ - 1 - slide, q_s - BKV + 1]
    Length = slide - BKV - BQ + 2
    """
    total_unmasked = 0
    total_loop = 0
    for q_blk in range(0, N, BLOCK_Q):
        q_s = q_blk
        # current kernel loop: kv_loop_start .. kv_end
        if slide > 0:
            kv_min = max(0, q_s - slide + 1)
            kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV
        else:
            kv_loop_start = 0
        kv_end = q_s + BLOCK_Q  # causal

        # unmasked range under split-loop:
        kv_unmasked_lo = q_s + BLOCK_Q - 1 - slide + 1  # exclusive lower
        kv_unmasked_hi = q_s - BLOCK_KV + 1             # inclusive upper
        # Align to BLOCK_KV grid, intersect with [kv_loop_start, kv_end)
        kv_u_start = max(kv_loop_start, max(0, ((kv_unmasked_lo + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV))
        kv_u_end = min(kv_end, (kv_unmasked_hi // BLOCK_KV + 1) * BLOCK_KV)
        unmasked = max(0, (kv_u_end - kv_u_start) // BLOCK_KV)
        looped = max(0, (kv_end - kv_loop_start + BLOCK_KV - 1) // BLOCK_KV)
        total_unmasked += unmasked
        total_loop += looped
    return total_unmasked, total_loop


def run(N, B=1, H_Q=8, H_KV=1, D=256, causal=True, slide_size=512,
        warmup=10, rep=30):
    device = "cuda"
    dtype = torch.float16
    GQA_RATIO = H_Q // H_KV

    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

    # Forward defaults match FlashAttnGQAFunction for D < 512
    BLOCK_Q_F = min(128, triton.next_power_of_2(N))
    BLOCK_KV_F = min(64, triton.next_power_of_2(N))
    num_warps_F = 8
    grid_f = (triton.cdiv(N, BLOCK_Q_F), B * H_Q)
    scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)

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

    # SDPA full-causal reference (SDPA has no native SWA; we use SDPA full-causal
    # as the comparison — matches every SWA table in baseline.md).
    def run_fwd_sdpa():
        attention_gqa_ref(q, k, v, causal=True)
    t_sdpa_fwd = time_cuda(run_fwd_sdpa, warmup=5, rep=min(rep, 20))

    run_fwd_kernel(); torch.cuda.synchronize()
    do = torch.randn_like(o)

    # Delta
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

    # dQ
    dq = torch.empty_like(q)
    # D < 512 defaults (match FlashAttnGQAFunction.backward)
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

    # dKV packed
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

    # Triton fwd+bwd end-to-end
    qg = q.detach().clone().requires_grad_(True)
    kg = k.detach().clone().requires_grad_(True)
    vg = v.detach().clone().requires_grad_(True)
    def run_triton_fwdbwd():
        qg.grad = None; kg.grad = None; vg.grad = None
        out = flash_attn_gqa_train(qg, kg, vg, causal=causal, slide_size=slide_size)
        out.backward(do, retain_graph=False)
    t_triton_fwdbwd = time_cuda(run_triton_fwdbwd, warmup, rep)

    # SDPA fwd+bwd (full-causal reference)
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

    # SWA split-loop ceiling (fwd)
    unmasked_blocks, total_blocks = split_loop_ceiling(N, slide_size, BLOCK_Q_F, BLOCK_KV_F)
    ceiling_frac = unmasked_blocks / total_blocks if total_blocks > 0 else 0.0

    # Results
    fwd_fl = swa_fwd_flops(B, H_Q, N, D, slide_size)
    bwd_fl = swa_bwd_flops(B, H_Q, N, D, slide_size)
    results = {}
    results["fwd"] = analyze_kernel("fwd", t_fwd, fwd_fl,
                                    fwd_hbm_bytes(B, H_Q, H_KV, N, D))
    results["delta"] = analyze_kernel("delta", t_delta, 0.0,
                                      delta_hbm_bytes(B, H_Q, N, D))
    results["dQ"] = analyze_kernel("dQ", t_dq, bwd_fl * 0.5,
                                   dq_hbm_bytes(B, H_Q, H_KV, N, D))
    results["dKV"] = analyze_kernel("dKV", t_dkv, bwd_fl * 0.5,
                                    dkv_hbm_bytes(B, H_Q, H_KV, N, D))

    results["meta"] = {
        "N": N, "B": B, "H_Q": H_Q, "H_KV": H_KV, "D": D,
        "causal": causal, "slide_size": slide_size, "GQA_RATIO": GQA_RATIO,
        "block_cfg_fwd": f"BQ={BLOCK_Q_F}, BKV={BLOCK_KV_F}, w={num_warps_F}, BLOCK_D={D}",
        "block_cfg_dQ":  f"BQ={BQ_dq}, BKV={BKV_dq}, w={w_dq}",
        "block_cfg_dKV": f"BQ={BQ_dkv}, BKV={BKV_dkv}, w={w_dkv}, GQA_RATIO={GQA_RATIO}",
        "t_sdpa_fwd": t_sdpa_fwd,
        "t_triton_fwdbwd": t_triton_fwdbwd,
        "t_sdpa_fwdbwd": t_sdpa_fwdbwd,
        "fwd_speedup": t_sdpa_fwd / t_fwd if t_fwd > 0 else float("nan"),
        "fwdbwd_speedup": (t_sdpa_fwdbwd / t_triton_fwdbwd) if t_triton_fwdbwd > 0 and not math.isnan(t_sdpa_fwdbwd) else float("nan"),
        "fwd_unmasked_kv_blocks": unmasked_blocks,
        "fwd_total_kv_blocks": total_blocks,
        "fwd_split_loop_ceiling_%": 100 * ceiling_frac,
    }
    return results


def pprint_summary(res):
    m = res["meta"]
    print("=" * 100)
    print(f"N={m['N']}  B={m['B']}  H_Q={m['H_Q']}  H_KV={m['H_KV']}  D={m['D']}  "
          f"causal={m['causal']}  slide={m['slide_size']}  GQA={m['GQA_RATIO']}:1")
    print(f"  fwd block : {m['block_cfg_fwd']}")
    print(f"  dQ  block : {m['block_cfg_dQ']}")
    print(f"  dKV block : {m['block_cfg_dKV']}")
    print("-" * 100)

    header = f"{'kernel':>8} | {'time_ms':>8} | {'TFLOPS':>8} | {'MFU%':>6} | {'BW_perf':>14} | {'AI_perf':>8}"
    print(header)
    print("-" * len(header))
    for name in ["fwd", "delta", "dQ", "dKV"]:
        k = res[name]
        bw_p = k.get("bw_perfect_GB/s", k.get("bw_GB/s", 0.0))
        bw_p_u = k.get("bw_perfect_util_%", k.get("bw_util_%", 0.0))
        ai = k.get("ai_perfect_F/B", k.get("ai_F/B", 0.0))
        print(f"{name:>8} | {k['time_ms']:>8.3f} | {k['tflops']:>8.1f} | {k['mfu_%']:>5.1f}% | "
              f"{bw_p:>6.0f}GB/s ({bw_p_u:>4.0f}%) | {ai:>7.1f}")
    print("-" * 100)

    total_bwd_kernels = res["delta"]["time_ms"] + res["dQ"]["time_ms"] + res["dKV"]["time_ms"]
    total_fwdbwd = m["t_triton_fwdbwd"]
    print(f"Fwd (Triton)    : {res['fwd']['time_ms']:.3f} ms  vs SDPA full-causal {m['t_sdpa_fwd']:.3f} ms  speedup {m['fwd_speedup']:.2f}x")
    print(f"Fwd+Bwd (Triton): {total_fwdbwd:.3f} ms  vs SDPA full-causal {m['t_sdpa_fwdbwd']:.3f} ms  speedup {m['fwdbwd_speedup']:.2f}x")
    print(f"  bwd kernels sum = {total_bwd_kernels:.3f} ms  ({100*total_bwd_kernels/total_fwdbwd:.1f}% of fwd+bwd)")
    print(f"  fwd kernel time = {res['fwd']['time_ms']:.3f} ms  ({100*res['fwd']['time_ms']/total_fwdbwd:.1f}% of fwd+bwd)")
    overhead = total_fwdbwd - total_bwd_kernels - res['fwd']['time_ms']
    print(f"  overhead/alloc  = {overhead:.3f} ms  ({100*overhead/total_fwdbwd:.1f}%)")
    print("-" * 100)

    ranking = sorted([
        ("fwd", res["fwd"]["time_ms"]),
        ("delta", res["delta"]["time_ms"]),
        ("dQ", res["dQ"]["time_ms"]),
        ("dKV", res["dKV"]["time_ms"]),
    ], key=lambda x: -x[1])
    total = sum(v for _, v in ranking)
    print("Hotspot rank (fraction of fwd+bwd kernel sum):")
    for i, (n, t) in enumerate(ranking, 1):
        print(f"  #{i}  {n:>5}  {t:>8.3f} ms  ({100*t/total:.1f}%)")

    print("-" * 100)
    print(f"Split-loop ceiling (fwd only): {m['fwd_unmasked_kv_blocks']}/{m['fwd_total_kv_blocks']} "
          f"KV blocks fully in-window & off-diagonal = {m['fwd_split_loop_ceiling_%']:.1f}% mask-free ceiling")


def main():
    # Gemma-4-E2B real sliding config
    B, H_Q, H_KV, D = 1, 8, 1, 256
    configs = []
    for slide in [512, 1024]:
        for N in [8192, 16384, 32768]:
            configs.append((N, slide))

    all_results = {}
    for N, slide in configs:
        if N <= 8192:
            warmup, rep = 10, 30
        elif N <= 16384:
            warmup, rep = 8, 20
        else:
            warmup, rep = 5, 12
        r = run(N, B=B, H_Q=H_Q, H_KV=H_KV, D=D,
                causal=True, slide_size=slide,
                warmup=warmup, rep=rep)
        pprint_summary(r)
        all_results[f"N={N}_slide={slide}"] = r

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "profile_swa.json")
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
