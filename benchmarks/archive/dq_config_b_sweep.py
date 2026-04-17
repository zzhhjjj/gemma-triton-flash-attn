"""Quick dQ sweep for Config B (H_Q=8, H_KV=1, D=256) at short N."""
import math, os, sys
import torch, triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flash_attn.attention import (
    _delta_kernel, _flash_attn_gqa_kernel, _flash_attn_gqa_bwd_dq_kernel,
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


def setup(N, slide=512, B=1, H_Q=8, H_KV=1, D=256):
    device="cuda"; dtype=torch.float16
    scale=1.0/math.sqrt(D); swa_slide = slide if slide < N else 0
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
    return q,k,v,o,do,lse,delta,scale,swa_slide


def time_dq(q,k,v,o,do,lse,delta,BQ,BKV,w,st,H_Q=8,H_KV=1,D=256,B=1):
    N=q.shape[2]
    BQ=min(BQ,triton.next_power_of_2(N))
    BKV=min(BKV,triton.next_power_of_2(N))
    scale=1.0/math.sqrt(D)
    swa_slide = 512 if 512 < N else 0
    dq=torch.empty_like(q)
    grid = (triton.cdiv(N,BQ), B*H_Q)
    def run_it():
        _flash_attn_gqa_bwd_dq_kernel[grid](
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
            BLOCK_Q=BQ, BLOCK_KV=BKV,
            IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
            STORE_DELTA=False,
            num_warps=w, num_stages=st,
        )
    try: return time_cuda(run_it)
    except Exception: return None


def main():
    torch.manual_seed(0)
    for N in [1024, 2048, 4096]:
        print(f"\n=== Config B dQ N={N} slide=512 ===", flush=True)
        q,k,v,o,do,lse,delta,scale,swa_slide = setup(N)
        rows = []
        for BQ in [32, 64, 128]:
            for BKV in [32, 64]:
                for w in [4, 8]:
                    for st in [1, 2]:
                        t = time_dq(q,k,v,o,do,lse,delta, BQ,BKV,w,st)
                        if t is not None:
                            rows.append({"BQ":BQ,"BKV":BKV,"w":w,"s":st,"t":t})
                            print(f"  BQ={BQ:>3} BKV={BKV:>3} w={w} s={st}  t={t:>7.3f}ms", flush=True)
        rows.sort(key=lambda r:r["t"])
        print("\n  TOP 3:")
        for r in rows[:3]:
            print(f"    BQ={r['BQ']:>3} BKV={r['BKV']:>3} w={r['w']} s={r['s']}  {r['t']:.3f}ms", flush=True)
        cur = [r for r in rows if r['BQ']==64 and r['BKV']==64 and r['w']==4 and r['s']==2]
        if cur:
            print(f"  Current (BQ=64,BKV=64,w=4,s=2): {cur[0]['t']:.3f}ms rank #{rows.index(cur[0])+1}", flush=True)


if __name__ == "__main__":
    main()
