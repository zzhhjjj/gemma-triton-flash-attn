"""Sweep pack-GQA dKV kernel configs for D=256 SWA (Gemma-4-E2B sliding).

Primary:       N=16384, slide=512   (Gemma-4-E2B real config)
Cross-verify:  N=32768 slide=512, N=16384 slide=1024

Current default (attention.py:1515-1516, D<512 branch):
    BKV=32, BQ=64, warps=4, stages=2

The default was tuned for the OLD per-Q-head split kernel (2026-04-16);
pack-GQA register model is different (BKV drives accumulator, BQ does not),
so the config may be sub-optimal for SWA dKV which is currently 44-50% of
SWA fwd+bwd time with MFU just 4-5%.
"""
import math
import os
import sys
import itertools
import json

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import (
    _flash_attn_gqa_kernel,
    _delta_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
)


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


def prepare_tensors(B, H_Q, H_KV, N, D, dtype, causal, slide_size):
    device = "cuda"
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)

    o = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    BQ_F = 64 if D >= 512 else 128
    BKV_F = 32 if D >= 512 else 64
    _flash_attn_gqa_kernel[(triton.cdiv(N, BQ_F), B * H_Q)](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ_F, BLOCK_KV=BKV_F, BLOCK_D=D,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
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
    return q, k, v, do, lse, delta


def try_config(q, k, v, do, lse, delta, BQ, BKV, warps, stages, causal, slide_size,
               warmup=3, rep=10):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    GQA_RATIO = H_Q // H_KV
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    grid = (triton.cdiv(N, BKV), B * H_KV)
    scale = 1.0 / math.sqrt(D)

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
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BQ, BLOCK_KV=BKV,
            GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
            num_warps=warps, num_stages=stages,
        )
    try:
        t = time_cuda(run, warmup=warmup, rep=rep)
        return t, None
    except Exception as ex:
        return None, str(ex).split("\n")[0][:120]


def sweep_shape(B, H_Q, H_KV, N, D, dtype, causal, slide_size,
                BKVs, BQs, warps_list, stages_list, warmup, rep):
    q, k, v, do, lse, delta = prepare_tensors(B, H_Q, H_KV, N, D, dtype, causal, slide_size)
    results = []
    for BKV, BQ, w, s in itertools.product(BKVs, BQs, warps_list, stages_list):
        t, err = try_config(q, k, v, do, lse, delta, BQ, BKV, w, s, causal, slide_size,
                            warmup=warmup, rep=rep)
        results.append({"BKV": BKV, "BQ": BQ, "warps": w, "stages": s,
                        "time_ms": t, "err": err})
    del q, k, v, do, lse, delta
    torch.cuda.empty_cache()
    return results


def print_results(title, results, baseline_ms=None):
    print(f"\n=== {title} ===")
    valid = [r for r in results if r["time_ms"] is not None]
    valid.sort(key=lambda r: r["time_ms"])
    print(f"{'rank':>4} {'BKV':>4} {'BQ':>4} {'W':>3} {'S':>3} | {'time (ms)':>10} | {'vs base':>8}")
    print("-" * 60)
    for i, r in enumerate(valid[:12], 1):
        ratio = f"{baseline_ms / r['time_ms']:.2f}x" if baseline_ms else "-"
        print(f"{i:>4} {r['BKV']:>4} {r['BQ']:>4} {r['warps']:>3} {r['stages']:>3} | "
              f"{r['time_ms']:>10.3f} | {ratio:>8}")
    fails = [r for r in results if r["time_ms"] is None]
    if fails:
        print(f"\nFailed configs ({len(fails)} OOM/error):")
        seen = set()
        for r in fails:
            err_sig = r["err"][:60] if r["err"] else "unknown"
            if err_sig not in seen:
                print(f"  e.g. BKV={r['BKV']} BQ={r['BQ']} w={r['warps']} s={r['stages']}: {err_sig}")
                seen.add(err_sig)


def main():
    B, H_Q, H_KV, D = 1, 8, 1, 256
    dtype = torch.float16
    causal = True

    BKVs = [16, 32, 64]
    BQs = [32, 64, 128]
    warps_list = [4, 8]
    stages_list = [1, 2, 3]

    shapes = [
        ("primary", 16384, 512, 8, 20),
        ("cross-N32K", 32768, 512, 5, 12),
        ("cross-slide1024", 16384, 1024, 8, 20),
    ]

    print(f"D={D} H_Q={H_Q} H_KV={H_KV} GQA={H_Q//H_KV}:1 fp16 causal=True")
    print(f"Current default (attention.py D<512): BKV=32, BQ=64, warps=4, stages=2")
    print(f"Sweep space: BKV∈{BKVs} × BQ∈{BQs} × warps∈{warps_list} × stages∈{stages_list} "
          f"= {len(BKVs)*len(BQs)*len(warps_list)*len(stages_list)} configs")

    all_results = {}
    baselines = {}

    for label, N, slide, warmup, rep in shapes:
        print(f"\n{'='*70}")
        print(f"Shape: {label}  N={N} slide={slide}  warmup={warmup} rep={rep}")
        print("="*70)
        t0 = torch.cuda.Event(enable_timing=True)
        t0.record()
        results = sweep_shape(B, H_Q, H_KV, N, D, dtype, causal, slide,
                              BKVs, BQs, warps_list, stages_list, warmup, rep)
        t1 = torch.cuda.Event(enable_timing=True)
        t1.record()
        torch.cuda.synchronize()
        print(f"(sweep took {t0.elapsed_time(t1)/1000:.1f}s)")

        # find current default time for baseline
        base = next((r for r in results
                     if r["BKV"]==32 and r["BQ"]==64 and r["warps"]==4 and r["stages"]==2), None)
        baseline_ms = base["time_ms"] if base else None
        baselines[label] = baseline_ms
        print(f"Current default time: {baseline_ms:.3f} ms" if baseline_ms else "Current default: FAILED")

        print_results(f"{label} N={N} slide={slide}", results, baseline_ms)
        all_results[label] = {"N": N, "slide": slide, "baseline_ms": baseline_ms,
                              "results": results}

    # Cross-shape winner: pick config minimizing sum(time/baseline) across shapes
    print("\n" + "="*70)
    print("Cross-shape ranking (minimize geometric mean of time/baseline):")
    print("="*70)

    configs = [(BKV, BQ, w, s) for BKV, BQ, w, s in itertools.product(BKVs, BQs, warps_list, stages_list)]
    cross_ranked = []
    for cfg in configs:
        BKV, BQ, w, s = cfg
        per_shape = []
        for label, data in all_results.items():
            r = next((r for r in data["results"]
                      if r["BKV"]==BKV and r["BQ"]==BQ and r["warps"]==w and r["stages"]==s), None)
            if r is None or r["time_ms"] is None:
                per_shape = None
                break
            per_shape.append((r["time_ms"], data["baseline_ms"]))
        if per_shape is None:
            continue
        # geomean of time_ms across shapes
        import math as _m
        logs = [_m.log(t) for t, _ in per_shape]
        geom = _m.exp(sum(logs) / len(logs))
        base_logs = [_m.log(b) for _, b in per_shape if b]
        base_geom = _m.exp(sum(base_logs) / len(base_logs)) if base_logs else None
        ratio = base_geom / geom if base_geom else None
        cross_ranked.append({
            "BKV": BKV, "BQ": BQ, "warps": w, "stages": s,
            "geom_time_ms": geom, "speedup_vs_baseline": ratio,
            "times": [t for t, _ in per_shape],
        })
    cross_ranked.sort(key=lambda r: r["geom_time_ms"])
    print(f"{'rank':>4} {'BKV':>4} {'BQ':>4} {'W':>3} {'S':>3} | {'geomean (ms)':>13} | "
          f"{'vs base':>8} | per-shape times (ms)")
    print("-" * 90)
    for i, r in enumerate(cross_ranked[:8], 1):
        times_str = ", ".join(f"{t:.2f}" for t in r["times"])
        sp = f"{r['speedup_vs_baseline']:.2f}x" if r["speedup_vs_baseline"] else "-"
        print(f"{i:>4} {r['BKV']:>4} {r['BQ']:>4} {r['warps']:>3} {r['stages']:>3} | "
              f"{r['geom_time_ms']:>13.3f} | {sp:>8} | {times_str}")

    out = {
        "config": {"B": B, "H_Q": H_Q, "H_KV": H_KV, "D": D, "dtype": str(dtype)},
        "sweep_space": {"BKVs": BKVs, "BQs": BQs, "warps": warps_list, "stages": stages_list},
        "shapes": {k: v for k, v in all_results.items()},
        "baselines_ms": baselines,
        "cross_shape_ranking": cross_ranked[:8],
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dkv_swa_sweep_D256.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
