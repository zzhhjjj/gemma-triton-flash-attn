"""Focused BKV/BQ/w sweep for Config A (H_Q=32, H_KV=16, D=256) dKV kernel
at short N. Q_SPLITS stays 1 (raw_grid always ≥ 256 here).

Target: raw_grid at N=1K is 512 programs (>> 132 SMs × 2 waves = 264).
Maybe larger BKV = smaller grid = better amortization of Q loop.
"""
import math, os, sys
import torch, triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flash_attn.attention import (
    _delta_kernel, _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
)


def time_cuda(fn, warmup=8, rep=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2]


def setup(N, slide, B, H_Q, H_KV, D):
    device = "cuda"; dtype = torch.float16
    scale = 1.0 / math.sqrt(D)
    swa_slide = slide if slide < N else 0
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
    o = torch.empty_like(q); lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    BQ_F = min(128, triton.next_power_of_2(N))
    BKV_F = min(64, triton.next_power_of_2(N))
    _flash_attn_gqa_kernel[(triton.cdiv(N, BQ_F), B*H_Q)](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=scale, BLOCK_Q=BQ_F, BLOCK_KV=BKV_F, BLOCK_D=D,
        IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=True,
        num_warps=8, num_stages=2,
    )
    do = torch.randn_like(o)
    delta = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    _delta_kernel[(N, B*H_Q)](
        do, o, delta,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_HEADS=H_Q, SEQ_LEN=N, HEAD_DIM=D, num_warps=4, num_stages=2,
    )
    return q, k, v, do, lse, delta, scale, swa_slide


def time_config(q, k, v, do, lse, delta, BKV, BQ, w, st, H_Q, H_KV, N, D, scale, swa_slide, B=1):
    GQA_RATIO = H_Q // H_KV
    BKV = min(BKV, triton.next_power_of_2(N))
    BQ = min(BQ, triton.next_power_of_2(N))
    dk = torch.empty_like(k); dv = torch.empty_like(v)
    grid = (triton.cdiv(N, BKV), B * H_KV, 1)

    def run_it():
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
            BLOCK_Q=BQ, BLOCK_KV=BKV, GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
            Q_SPLITS=1,
            num_warps=w, num_stages=st,
        )
    try:
        return time_cuda(run_it)
    except Exception:
        return None


def main():
    import sys
    torch.manual_seed(0)
    B, H_Q, H_KV, D = 1, 32, 16, 256

    for N in [1024, 2048, 4096]:
        print(f"\n{'='*72}", flush=True)
        print(f" Config A  N={N}  slide=1024  (H_Q=32, H_KV=16, D=256)", flush=True)
        print('='*72, flush=True)
        q, k, v, do, lse, delta, scale, swa_slide = setup(N, 1024, B, H_Q, H_KV, D)
        rows = []
        # Focused: BKV ∈ {32, 64}, BQ ∈ {32, 64, 128}, w ∈ {4, 8}, s ∈ {1, 2}
        for BKV in [32, 64]:
            for BQ in [32, 64, 128]:
                for w in [4, 8]:
                    for st in [1, 2]:
                        t = time_config(q, k, v, do, lse, delta, BKV, BQ, w, st, H_Q, H_KV, N, D, scale, swa_slide)
                        if t:
                            rows.append({"BKV": BKV, "BQ": BQ, "w": w, "s": st, "t": t})
                            print(f"  BKV={BKV:>3} BQ={BQ:>3} w={w} s={st}  t={t:>7.3f}ms", flush=True)
        rows.sort(key=lambda r: r["t"])
        print(f"\n  TOP 5:")
        for r in rows[:5]:
            print(f"    BKV={r['BKV']:>3} BQ={r['BQ']:>3} w={r['w']} s={r['s']}  {r['t']:.3f}ms", flush=True)
        cur = [r for r in rows if r['BKV']==32 and r['BQ']==64 and r['w']==4 and r['s']==2]
        if cur:
            print(f"  Current (BKV=32, BQ=64, w=4, s=2): {cur[0]['t']:.3f}ms (rank #{rows.index(cur[0])+1})", flush=True)


if __name__ == "__main__":
    main()
