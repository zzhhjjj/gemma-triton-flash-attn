"""Isolate Triton kernel launch overhead from the rest.

Hypothesis: Python autograd Function overhead (measured 101μs @ Config B N=1K
in alloc_overhead.py) decomposes as:
  - torch.autograd.Function.backward() machinery (~20-30μs)
  - Triton `fn[grid](args)` Python wrapper per launch (~30-50μs × 2 launches)
  - ctx.saved_tensors unpacking (~5-10μs)

If the Triton launch wrapper is the dominant piece (60-100μs), then neither
`fn.run()` low-level API nor argument reordering will shrink it much —
it's the specialization hash + compile cache lookup + CUDA runtime call.

This script measures:
  1. Raw kernel dispatch (single `fn[grid]` launch, no alloc, no autograd)
  2. Two back-to-back launches
  3. Same with autograd.Function wrapping
"""
import math, os, sys
import torch, triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import (
    FlashAttnGQAFunction,
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dq_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
)


def time_cuda(fn, warmup=30, rep=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2]


def main():
    # Config B N=1K where Python overhead is biggest
    B, H_Q, H_KV, N, D = 1, 8, 1, 1024, 256
    device="cuda"; dtype=torch.float16
    torch.manual_seed(0)
    q = torch.randn(B,H_Q,N,D,dtype=dtype,device=device)
    k = torch.randn(B,H_KV,N,D,dtype=dtype,device=device)
    v = torch.randn(B,H_KV,N,D,dtype=dtype,device=device)
    o = torch.empty_like(q)
    lse = torch.empty(B,H_Q,N,dtype=torch.float32,device=device)

    # Populate o, lse via fwd
    BQ_F, BKV_F = 128, 64
    _flash_attn_gqa_kernel[(triton.cdiv(N,BQ_F), B*H_Q)](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0/math.sqrt(D), BLOCK_Q=BQ_F, BLOCK_KV=BKV_F, BLOCK_D=D,
        IS_CAUSAL=True, SLIDE_SIZE=0,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=True,
        num_warps=8, num_stages=2,
    )
    torch.cuda.synchronize()

    do = torch.randn_like(o)
    delta = torch.empty(B,H_Q,N,dtype=torch.float32,device=device)
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k); dv = torch.zeros_like(v)

    scale = 1.0/math.sqrt(D)

    # dQ config
    BQ_DQ, BKV_DQ, w_dq = 64, 64, 4
    grid_dq = (triton.cdiv(N,BQ_DQ), B*H_Q)

    # dKV config (rescue path for Config B N=1K)
    BKV_DKV, BQ_DKV, w_dkv, s_dkv, QS = 64, 128, 8, 1, 8
    grid_dkv = (triton.cdiv(N,BKV_DKV), B*H_KV, QS)

    # Test 1: dQ alone
    def f_dq():
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
            BLOCK_Q=BQ_DQ, BLOCK_KV=BKV_DQ,
            IS_CAUSAL=True, SLIDE_SIZE=0,
            STORE_DELTA=True,
            num_warps=w_dq, num_stages=2,
        )
    t_dq = time_cuda(f_dq)

    # Test 2: dKV alone
    def f_dkv():
        dk.zero_(); dv.zero_()
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
            BLOCK_Q=BQ_DKV, BLOCK_KV=BKV_DKV, GQA_RATIO=H_Q//H_KV,
            IS_CAUSAL=True, SLIDE_SIZE=0, Q_SPLITS=QS,
            num_warps=w_dkv, num_stages=s_dkv,
        )
    t_dkv = time_cuda(f_dkv)

    # Test 3: Both together (sequential launch via same stream)
    def f_both():
        dk.zero_(); dv.zero_()
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
            BLOCK_Q=BQ_DQ, BLOCK_KV=BKV_DQ,
            IS_CAUSAL=True, SLIDE_SIZE=0,
            STORE_DELTA=True,
            num_warps=w_dq, num_stages=2,
        )
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
            BLOCK_Q=BQ_DKV, BLOCK_KV=BKV_DKV, GQA_RATIO=H_Q//H_KV,
            IS_CAUSAL=True, SLIDE_SIZE=0, Q_SPLITS=QS,
            num_warps=w_dkv, num_stages=s_dkv,
        )
    t_both = time_cuda(f_both)

    # Autograd-wrapped bwd
    qg=q.detach().clone().requires_grad_(True)
    kg=k.detach().clone().requires_grad_(True)
    vg=v.detach().clone().requires_grad_(True)
    def f_autograd():
        qg.grad=None; kg.grad=None; vg.grad=None
        o2 = FlashAttnGQAFunction.apply(qg, kg, vg, True, 0)
        o2.backward(do, retain_graph=False)
    t_ag_fb = time_cuda(f_autograd)
    def f_ag_fwd():
        FlashAttnGQAFunction.apply(q, k, v, True, 0)
    t_ag_fwd = time_cuda(f_ag_fwd)
    t_ag_bwd = t_ag_fb - t_ag_fwd

    print(f"Config B N=1K (H_Q=8,H_KV=1,D=256), causal=True:")
    print(f"  dQ alone       (1 launch)   : {t_dq*1000:>7.1f} us")
    print(f"  dKV alone      (1 launch)   : {t_dkv*1000:>7.1f} us (incl. 2× .zero_())")
    print(f"  dQ + dKV       (2 launches) : {t_both*1000:>7.1f} us (incl. 2× .zero_())")
    print(f"  sum (dQ + dKV)              : {(t_dq+t_dkv)*1000:>7.1f} us")
    print(f"  'launch cost' (both - sum)  : {(t_both-t_dq-t_dkv)*1000:>7.1f} us  (back-to-back vs isolated)")
    print(f"  autograd bwd only           : {t_ag_bwd*1000:>7.1f} us")
    print(f"  python overhead (autograd - both) : {(t_ag_bwd-t_both)*1000:>7.1f} us")


if __name__ == "__main__":
    main()
