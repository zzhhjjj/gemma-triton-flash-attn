"""Probe BKV=64 × Q_SPLITS on Config B (H_Q=8, H_KV=1). My current gate
sends Config B short N to BKV=32 always, but BKV=64 + QS>1 might win via
different amortization. Quick check."""
import math, os, sys
import torch, triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flash_attn.attention import (
    _delta_kernel, _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
)


def time_cuda(fn, warmup=10, rep=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2]


def setup(N, slide, B=1, H_Q=8, H_KV=1, D=256):
    device="cuda"; dtype=torch.float16
    scale=1.0/math.sqrt(D)
    swa_slide = slide if slide < N else 0
    q=torch.randn(B,H_Q,N,D,dtype=dtype,device=device)
    k=torch.randn(B,H_KV,N,D,dtype=dtype,device=device)
    v=torch.randn(B,H_KV,N,D,dtype=dtype,device=device)
    o=torch.empty_like(q); lse=torch.empty(B,H_Q,N,dtype=torch.float32,device=device)
    BQ_F=min(128,triton.next_power_of_2(N)); BKV_F=min(64,triton.next_power_of_2(N))
    _flash_attn_gqa_kernel[(triton.cdiv(N,BQ_F), B*H_Q)](
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
    do=torch.randn_like(o); delta=torch.empty(B,H_Q,N,dtype=torch.float32,device=device)
    _delta_kernel[(N,B*H_Q)](
        do, o, delta,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_HEADS=H_Q, SEQ_LEN=N, HEAD_DIM=D, num_warps=4, num_stages=2,
    )
    return q,k,v,do,lse,delta,scale,swa_slide


def run_dkv(q,k,v,do,lse,delta,BKV,BQ,w,st,QS,H_Q=8,H_KV=1,D=256,B=1):
    GQA_RATIO=H_Q//H_KV
    N=q.shape[2]
    BKV=min(BKV,triton.next_power_of_2(N))
    BQ=min(BQ,triton.next_power_of_2(N))
    scale=1.0/math.sqrt(D)
    swa_slide = 512 if 512 < N else 0
    if QS>1:
        dk=torch.zeros_like(k); dv=torch.zeros_like(v)
    else:
        dk=torch.empty_like(k); dv=torch.empty_like(v)
    grid=(triton.cdiv(N,BKV), B*H_KV, QS)
    def run_it():
        if QS>1: dk.zero_(); dv.zero_()
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
            num_warps=w, num_stages=st,
        )
    try: return time_cuda(run_it)
    except Exception: return None


def main():
    torch.manual_seed(0)
    for N in [1024, 2048, 4096]:
        print(f"\n=== Config B N={N} slide=512 ===", flush=True)
        tup = setup(N, 512)
        q,k,v,do,lse,delta,scale,swa_slide = tup
        print(f"{'cfg':<28} | {'time':>8} | {'grid':>5}")
        # Current wrapper (my best per N with BKV=32)
        cur_QS = {1024: 8, 2048: 4, 4096: 2}[N]
        t = run_dkv(q,k,v,do,lse,delta, 32, 64, 4, 2, cur_QS)
        raw = triton.cdiv(N, 32)
        print(f"  BKV=32 BQ=64 w=4 s=2 QS={cur_QS:<2} (current) | {t:>7.3f}ms | {raw*cur_QS:>5}")

        # BKV=64 with various QS
        for QS in [1, 2, 4, 8]:
            # Try BQ=128 w=8 s=1 (the Config A winner) and BQ=64 w=8 s=1
            for BQ, w, st in [(128, 8, 1), (64, 8, 1), (128, 4, 2), (64, 4, 2)]:
                t = run_dkv(q,k,v,do,lse,delta, 64, BQ, w, st, QS)
                if t is not None:
                    raw = triton.cdiv(N, 64)
                    print(f"  BKV=64 BQ={BQ:<3} w={w} s={st} QS={QS:<2}          | {t:>7.3f}ms | {raw*QS:>5}")


if __name__ == "__main__":
    main()
