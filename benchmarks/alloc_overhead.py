"""Measure allocation overhead in the bwd path.

Compare:
  1. bwd with allocations INSIDE the timed region (current wrapper behavior)
  2. bwd with allocations OUTSIDE the timed region (warm-cached)
  3. kernel launches only (no allocations at all)
  4. Python-only autograd overhead (no kernel, no alloc)

Targets:
  Config B (Gemma-4-E2B 8:1) — biggest overhead share at short N
  Config A (Gemma4 syn 2:1) — sanity check
"""
import math, os, sys, time
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


def bench(B, H_Q, H_KV, N, D, slide_size):
    device="cuda"; dtype=torch.float16
    torch.manual_seed(0)

    q=torch.randn(B,H_Q,N,D,dtype=dtype,device=device)
    k=torch.randn(B,H_KV,N,D,dtype=dtype,device=device)
    v=torch.randn(B,H_KV,N,D,dtype=dtype,device=device)

    # warmup fwd via autograd to populate ctx
    qg=q.detach().clone().requires_grad_(True)
    kg=k.detach().clone().requires_grad_(True)
    vg=v.detach().clone().requires_grad_(True)
    out = FlashAttnGQAFunction.apply(qg, kg, vg, True, slide_size)
    do = torch.randn_like(out)

    # Full autograd fwd+bwd (baseline)
    def f_autograd_fwdbwd():
        qg.grad=None; kg.grad=None; vg.grad=None
        o = FlashAttnGQAFunction.apply(qg, kg, vg, True, slide_size)
        o.backward(do, retain_graph=False)
    t_fwdbwd = time_cuda(f_autograd_fwdbwd)

    # Autograd fwd only
    def f_autograd_fwd():
        FlashAttnGQAFunction.apply(q, k, v, True, slide_size)
    t_fwd = time_cuda(f_autograd_fwd)

    # Manually construct bwd kernel launches + allocations (mirror wrapper exactly)
    # Precompute fwd outputs o, lse
    o_buf = torch.empty_like(q)
    lse_buf = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    BQ_F = min(128, triton.next_power_of_2(N))
    BKV_F = min(64, triton.next_power_of_2(N))
    grid_f = (triton.cdiv(N, BQ_F), B * H_Q)
    scale = 1.0 / math.sqrt(D)
    swa_slide = slide_size if slide_size < N else 0
    _flash_attn_gqa_kernel[grid_f](
        q, k, v, o_buf,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=scale, BLOCK_Q=BQ_F, BLOCK_KV=BKV_F, BLOCK_D=D,
        IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
        LSE_ptr=lse_buf, stride_lseb=lse_buf.stride(0), stride_lseh=lse_buf.stride(1),
        stride_lsen=lse_buf.stride(2), STORE_LSE=True,
        num_warps=8, num_stages=2,
    )
    torch.cuda.synchronize()

    GQA_RATIO = H_Q // H_KV

    # Replicate the actual heuristic from attention.py backward
    grid_at_bkv64 = triton.cdiv(N, 64) * B * H_KV
    if grid_at_bkv64 >= 128 or grid_at_bkv64 <= 16:
        BKV_DKV, BQ_DKV, w_dkv, s_dkv = 64, 128, 8, 1
    else:
        BKV_DKV, BQ_DKV, w_dkv, s_dkv = 32, 64, 4, 2
    BKV_DKV = min(BKV_DKV, triton.next_power_of_2(N))
    BQ_DKV = min(BQ_DKV, triton.next_power_of_2(N))
    raw_grid = triton.cdiv(N, BKV_DKV) * B * H_KV
    target = 128 if BKV_DKV == 64 else 256
    if raw_grid >= target: QS = 1
    elif raw_grid * 2 >= target: QS = 2
    elif raw_grid * 4 >= target: QS = 4
    else: QS = 8

    if D >= 512:
        BLOCK_Q_DQ, BLOCK_KV_DQ, num_warps_dq = 32, 64, 8
    else:
        BLOCK_Q_DQ, BLOCK_KV_DQ, num_warps_dq = 64, 64, 4
    BLOCK_Q_DQ = min(BLOCK_Q_DQ, triton.next_power_of_2(N))
    BLOCK_KV_DQ = min(BLOCK_KV_DQ, triton.next_power_of_2(N))

    grid_dq = (triton.cdiv(N, BLOCK_Q_DQ), B * H_Q)
    grid_dkv = (triton.cdiv(N, BKV_DKV), B * H_KV, QS)

    # --- Variant A: allocations INSIDE timed region (matches wrapper)
    def f_kernels_with_alloc():
        delta = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
        dq = torch.empty_like(q)
        if QS > 1:
            dk = torch.zeros_like(k); dv = torch.zeros_like(v)
        else:
            dk = torch.empty_like(k); dv = torch.empty_like(v)
        _flash_attn_gqa_bwd_dq_kernel[grid_dq](
            q, k, v, do, o_buf, dq, lse_buf, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            lse_buf.stride(0), lse_buf.stride(1), lse_buf.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BLOCK_Q_DQ, BLOCK_KV=BLOCK_KV_DQ,
            IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
            STORE_DELTA=True,
            num_warps=num_warps_dq, num_stages=2,
        )
        _flash_attn_gqa_bwd_dkv_packed_kernel[grid_dkv](
            q, k, v, do, dk, dv, lse_buf, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            lse_buf.stride(0), lse_buf.stride(1), lse_buf.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BQ_DKV, BLOCK_KV=BKV_DKV, GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
            Q_SPLITS=QS,
            num_warps=w_dkv, num_stages=s_dkv,
        )
    t_k_alloc = time_cuda(f_kernels_with_alloc)

    # --- Variant B: allocations OUTSIDE timed region (pre-alloc, reuse)
    delta_pre = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
    dq_pre = torch.empty_like(q)
    dk_pre = torch.empty_like(k)
    dv_pre = torch.empty_like(v)
    def f_kernels_no_alloc():
        if QS > 1:
            dk_pre.zero_(); dv_pre.zero_()
        _flash_attn_gqa_bwd_dq_kernel[grid_dq](
            q, k, v, do, o_buf, dq_pre, lse_buf, delta_pre,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
            dq_pre.stride(0), dq_pre.stride(1), dq_pre.stride(2), dq_pre.stride(3),
            lse_buf.stride(0), lse_buf.stride(1), lse_buf.stride(2),
            delta_pre.stride(0), delta_pre.stride(1), delta_pre.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BLOCK_Q_DQ, BLOCK_KV=BLOCK_KV_DQ,
            IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
            STORE_DELTA=True,
            num_warps=num_warps_dq, num_stages=2,
        )
        _flash_attn_gqa_bwd_dkv_packed_kernel[grid_dkv](
            q, k, v, do, dk_pre, dv_pre, lse_buf, delta_pre,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk_pre.stride(0), dk_pre.stride(1), dk_pre.stride(2), dk_pre.stride(3),
            dv_pre.stride(0), dv_pre.stride(1), dv_pre.stride(2), dv_pre.stride(3),
            lse_buf.stride(0), lse_buf.stride(1), lse_buf.stride(2),
            delta_pre.stride(0), delta_pre.stride(1), delta_pre.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BQ_DKV, BLOCK_KV=BKV_DKV, GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=True, SLIDE_SIZE=swa_slide,
            Q_SPLITS=QS,
            num_warps=w_dkv, num_stages=s_dkv,
        )
    t_k_pre = time_cuda(f_kernels_no_alloc)

    # --- Measure just allocations (no kernels)
    def f_alloc_only():
        delta = torch.empty(B, H_Q, N, dtype=torch.float32, device=device)
        dq = torch.empty_like(q)
        if QS > 1:
            dk = torch.zeros_like(k); dv = torch.zeros_like(v)
        else:
            dk = torch.empty_like(k); dv = torch.empty_like(v)
    t_alloc = time_cuda(f_alloc_only)

    # --- Measure autograd bwd only (exclude fwd)
    t_bwd_only = t_fwdbwd - t_fwd

    return {
        "config": dict(B=B,H_Q=H_Q,H_KV=H_KV,N=N,D=D,slide=slide_size,BKV_DKV=BKV_DKV,QS=QS),
        "autograd_fwdbwd": t_fwdbwd,
        "autograd_fwd": t_fwd,
        "autograd_bwd_only": t_bwd_only,
        "kernels_with_alloc": t_k_alloc,
        "kernels_no_alloc": t_k_pre,
        "alloc_only": t_alloc,
        "alloc_overhead_est": t_k_alloc - t_k_pre,
        "autograd_py_overhead": t_bwd_only - t_k_alloc,
    }


def main():
    for tag, cfg in [
        ("A N=1K",  dict(B=1,H_Q=32,H_KV=16,N=1024,D=256,slide_size=1024)),
        ("A N=2K",  dict(B=1,H_Q=32,H_KV=16,N=2048,D=256,slide_size=1024)),
        ("B N=1K",  dict(B=1,H_Q=8, H_KV=1, N=1024,D=256,slide_size=512)),
        ("B N=2K",  dict(B=1,H_Q=8, H_KV=1, N=2048,D=256,slide_size=512)),
        ("B N=4K",  dict(B=1,H_Q=8, H_KV=1, N=4096,D=256,slide_size=512)),
    ]:
        r = bench(**cfg)
        c = r["config"]
        print(f"\n=== {tag}  BKV_DKV={c['BKV_DKV']} QS={c['QS']} ===")
        print(f"  autograd fwd+bwd total  : {r['autograd_fwdbwd']:.4f} ms")
        print(f"  autograd fwd only       : {r['autograd_fwd']:.4f} ms")
        print(f"  autograd bwd only       : {r['autograd_bwd_only']:.4f} ms")
        print(f"  kernels with alloc      : {r['kernels_with_alloc']:.4f} ms")
        print(f"  kernels WITHOUT alloc   : {r['kernels_no_alloc']:.4f} ms  <-- lower bound")
        print(f"  alloc only (no kernels) : {r['alloc_only']:.4f} ms")
        print(f"  alloc overhead (alloc-noalloc) : {r['alloc_overhead_est']*1000:.1f} us")
        print(f"  autograd Python overhead (bwd - kernels_w_alloc): {r['autograd_py_overhead']*1000:.1f} us")


if __name__ == "__main__":
    main()
