"""Sweep pack-GQA dKV kernel configs for D=512 full-causal Gemma4.

Fixed: B=1, H_Q=32, H_KV=4, D=512, fp16, causal=True, N=4096 (primary) + 16384 (verify top).
Current default: BQ=16, BKV=32, w=8, stages=2 → 12.92ms @ N=4K.
"""
import math
import os
import sys
import itertools
import json

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


def prepare_tensors(B, H_Q, H_KV, N, D, dtype, causal):
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
    return q, k, v, do, lse, delta


def try_config(q, k, v, do, lse, delta, BQ, BKV, warps, stages, causal):
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
            IS_CAUSAL=causal, SLIDE_SIZE=0,
            Q_SPLITS=1,
            num_warps=warps, num_stages=stages,
        )
    try:
        t = time_cuda(run, warmup=3, rep=10)
        return t, None
    except Exception as ex:
        return None, str(ex).split("\n")[0][:120]


def main():
    B, H_Q, H_KV, D = 1, 32, 4, 512
    N = 4096
    dtype = torch.float16
    causal = True

    print(f"Config: B={B} H_Q={H_Q} H_KV={H_KV} D={D} N={N} fp16 causal={causal}")
    print(f"Current default: BQ=16 BKV=32 warps=8 stages=2")
    print()

    q, k, v, do, lse, delta = prepare_tensors(B, H_Q, H_KV, N, D, dtype, causal)

    BQs = [16, 32, 64, 128]
    BKVs = [16, 32, 64]
    warps_list = [4, 8]
    stages_list = [2, 3]

    results = []
    print(f"{'BQ':>4} {'BKV':>4} {'W':>3} {'S':>3} | {'time (ms)':>10} | note")
    print("-" * 70)

    for BQ, BKV, w, s in itertools.product(BQs, BKVs, warps_list, stages_list):
        t, err = try_config(q, k, v, do, lse, delta, BQ, BKV, w, s, causal)
        if t is None:
            print(f"{BQ:>4} {BKV:>4} {w:>3} {s:>3} | {'-':>10} | FAIL: {err}")
        else:
            print(f"{BQ:>4} {BKV:>4} {w:>3} {s:>3} | {t:>10.3f} |")
            results.append({"BQ": BQ, "BKV": BKV, "warps": w, "stages": s, "time_ms": t})

    results.sort(key=lambda r: r["time_ms"])
    print("\n=== Top 5 @ N=4096 ===")
    for r in results[:5]:
        print(f"  BQ={r['BQ']:>3} BKV={r['BKV']:>3} w={r['warps']} s={r['stages']}: {r['time_ms']:.3f} ms")

    # Verify top-3 at N=16384
    print(f"\n=== Verify top-3 at N=16384 ===")
    N2 = 16384
    q2, k2, v2, do2, lse2, delta2 = prepare_tensors(B, H_Q, H_KV, N2, D, dtype, causal)
    top_verify = []
    for r in results[:3]:
        t, err = try_config(q2, k2, v2, do2, lse2, delta2,
                            r["BQ"], r["BKV"], r["warps"], r["stages"], causal)
        note = err if t is None else ""
        print(f"  BQ={r['BQ']:>3} BKV={r['BKV']:>3} w={r['warps']} s={r['stages']}: "
              f"{t if t is not None else '-':.3f} ms  {note}")
        top_verify.append({**r, "time_N16K": t, "err_N16K": err})

    out = {
        "config": {"B": B, "H_Q": H_Q, "H_KV": H_KV, "D": D, "dtype": str(dtype),
                   "causal": causal, "N_sweep": N, "N_verify": N2},
        "sweep_N4K": results,
        "verify_N16K": top_verify,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dkv_sweep_D512.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
