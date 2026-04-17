"""Joint sweep of BLOCK_KV × Q_SPLITS for Config B dKV kernel.

Slimmed to ~32 configs per N for fast turnaround. Core question: does a
larger BKV (halving grid) with higher Q_SPLITS compensation beat BKV=32
at Config B? The raw_grid heuristic suggests QS should scale inversely.
"""
import math, os, sys
import torch, triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flash_attn.attention import (
    _delta_kernel, _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
)


def time_cuda(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
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
    return q, k, v, do, o, lse, delta, scale, swa_slide


def time_config(q, k, v, do, lse, delta, BKV, BQ, w, QS, H_Q, H_KV, N, D, scale, swa_slide, B=1):
    GQA_RATIO = H_Q // H_KV
    BKV = min(BKV, triton.next_power_of_2(N))
    BQ = min(BQ, triton.next_power_of_2(N))
    if QS > 1:
        dk = torch.zeros_like(k); dv = torch.zeros_like(v)
    else:
        dk = torch.empty_like(k); dv = torch.empty_like(v)
    grid = (triton.cdiv(N, BKV), B * H_KV, QS)

    def run_it():
        if QS > 1:
            dk.zero_(); dv.zero_()
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
            Q_SPLITS=QS,
            num_warps=w, num_stages=2,
        )
    try:
        return time_cuda(run_it, warmup=8, rep=40)
    except Exception:
        return None


def main():
    torch.manual_seed(0)
    B, H_Q, H_KV, D = 1, 8, 1, 256
    slide = 512

    for N in [1024, 2048, 4096]:
        print(f"\n{'='*72}", flush=True)
        print(f" Config B  N={N}  slide={slide}  (Gemma-4-E2B)", flush=True)
        print('='*72, flush=True)
        tup = setup(N, slide, B, H_Q, H_KV, D)
        q, k, v, do, o, lse, delta, scale, swa_slide = tup

        # Slim grid: focus on BKV ∈ {32, 64}, BQ ∈ {64, 128}, w ∈ {4, 8}, QS ∈ {1,2,4,8}
        rows = []
        for BKV in [16, 32, 64]:
            for BQ in [64, 128]:
                for w in [4, 8]:
                    for QS in [1, 2, 4, 8]:
                        raw_grid = triton.cdiv(N, BKV) * B * H_KV
                        # Skip: BKV=16 grid is already big; >1 QS wasteful at short N
                        if BKV == 16 and raw_grid * QS > 512:
                            continue
                        if BKV == 64 and QS == 8:
                            continue  # BKV=64 grid small but QS=8 likely atomic-dominated
                        t = time_config(q, k, v, do, lse, delta, BKV, BQ, w, QS, H_Q, H_KV, N, D, scale, swa_slide)
                        if t is not None:
                            rows.append({"BKV": BKV, "BQ": BQ, "w": w, "QS": QS, "t": t, "raw_grid": raw_grid})
                            print(f"  BKV={BKV:>3} BQ={BQ:>3} w={w} QS={QS}  t={t:>7.3f}ms  grid={raw_grid*QS:>4}", flush=True)
        rows.sort(key=lambda r: r["t"])
        print(f"\n  TOP 5:")
        for r in rows[:5]:
            print(f"    BKV={r['BKV']:>3} BQ={r['BQ']:>3} w={r['w']} QS={r['QS']}  {r['t']:.3f}ms", flush=True)
        cur = [r for r in rows if r['BKV']==32 and r['BQ']==64 and r['w']==4 and r['QS'] in (1, 2, 4, 8)]
        cur.sort(key=lambda r: r['QS'])
        print(f"  Current (BKV=32, BQ=64, w=4, heuristic-selected QS):")
        for r in cur:
            print(f"    QS={r['QS']}: {r['t']:.3f}ms", flush=True)


if __name__ == "__main__":
    main()
