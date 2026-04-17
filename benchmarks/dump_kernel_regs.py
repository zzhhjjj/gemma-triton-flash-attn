"""Dump num_regs / num_spills / shared_memory for fwd / dQ / dKV kernels
at the Gemma4 D=512 default configs. Used to test hypothesis (a):
'dKV MFU is half of dQ because of register spill'.

H100 compute capability 9.0: 65536 regs/SM, max 255 regs/thread, 228KB shmem/SM.
"""
import math
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import (
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_bwd_dq_kernel,
    _flash_attn_gqa_bwd_dkv_packed_kernel,
    _delta_kernel,
)


def get_compiled_kernel(jit_fn):
    """Return the single CompiledKernel just registered in device_caches[0]."""
    dc = jit_fn.device_caches
    _, _, _, _, _ = dc[0]  # validate format
    binder_cache, _hash_cache, _target, _backend, _fn = dc[0]
    assert len(binder_cache) == 1, f"expected 1 compiled variant, got {len(binder_cache)}"
    return next(iter(binder_cache.values()))


def pprint(name, ck, shmem_budget_kb=232):
    meta = ck.metadata
    shared = getattr(meta, "shared", None)
    if shared is None:
        shared = meta.get("shared", None) if hasattr(meta, "get") else None
    nregs = ck.n_regs
    spills = ck.n_spills
    num_warps = getattr(meta, "num_warps", None) or meta.get("num_warps", "?")
    num_stages = getattr(meta, "num_stages", None) or meta.get("num_stages", "?")
    shmem_kb = shared / 1024 if shared is not None else float("nan")
    shmem_pct = 100 * shmem_kb / shmem_budget_kb if shared is not None else float("nan")
    print(f"{name:>24} | regs={nregs:>4} | spills={spills:>4} | "
          f"shmem={shmem_kb:>6.1f}KB ({shmem_pct:>5.1f}% of {shmem_budget_kb}KB) | "
          f"warps={num_warps}, stages={num_stages}")


def run_fwd(B, H_Q, H_KV, N, D, causal=True):
    q = torch.randn(B, H_Q, N, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda")
    o = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device="cuda")
    BQ = min(64 if D >= 512 else 128, triton.next_power_of_2(N))
    BKV = min(32 if D >= 512 else 64, triton.next_power_of_2(N))
    w = 8 if D >= 256 else 4
    grid = (triton.cdiv(N, BQ), B * H_Q)
    _flash_attn_gqa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ, BLOCK_KV=BKV, BLOCK_D=D,
        IS_CAUSAL=causal, SLIDE_SIZE=0,
        LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
        stride_lsen=lse.stride(2), STORE_LSE=True,
        num_warps=w, num_stages=2,
    )
    torch.cuda.synchronize()
    return (BQ, BKV, w), q, k, v, o, lse


def run_dq(q, k, v, o, lse, H_Q, H_KV, N, D, causal=True):
    do = torch.randn_like(o)
    delta = torch.empty(1, H_Q, N, dtype=torch.float32, device="cuda")
    # populate delta first
    _delta_kernel[(N, 1 * H_Q)](
        do, o, delta,
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_HEADS=H_Q, SEQ_LEN=N, HEAD_DIM=D,
        num_warps=4, num_stages=2,
    )
    dq = torch.empty_like(q)
    if D >= 512:
        BQ, BKV, w = 32, 64, 8
    else:
        BQ, BKV, w = 64, 64, 4
    BQ = min(BQ, triton.next_power_of_2(N))
    BKV = min(BKV, triton.next_power_of_2(N))
    grid = (triton.cdiv(N, BQ), 1 * H_Q)
    _flash_attn_gqa_bwd_dq_kernel[grid](
        q, k, v, do, dq, lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
        HEAD_DIM=D, scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ, BLOCK_KV=BKV,
        IS_CAUSAL=causal, SLIDE_SIZE=0,
        num_warps=w, num_stages=2,
    )
    torch.cuda.synchronize()
    return (BQ, BKV, w), do, delta


def run_dkv(q, k, v, o, lse, do, delta, H_Q, H_KV, N, D, causal=True):
    if D >= 512:
        BKV, BQ, w = 16, 64, 4
    else:
        BKV, BQ, w = 32, 64, 4
    BKV = min(BKV, triton.next_power_of_2(N))
    BQ = min(BQ, triton.next_power_of_2(N))
    GQA_RATIO = H_Q // H_KV
    grid = (triton.cdiv(N, BKV), 1 * H_KV)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
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
        HEAD_DIM=D, scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ, BLOCK_KV=BKV, GQA_RATIO=GQA_RATIO,
        IS_CAUSAL=causal, SLIDE_SIZE=0,
        num_warps=w, num_stages=2,
    )
    torch.cuda.synchronize()
    return (BKV, BQ, w)


def main():
    B, H_Q, H_KV, D = 1, 32, 4, 512
    N = 8192  # profiling target; kernel metadata is config-dependent, not N-dependent

    # Compile each kernel once with its default config
    cfg_fwd, q, k, v, o, lse = run_fwd(B, H_Q, H_KV, N, D, causal=True)
    cfg_dq, do, delta = run_dq(q, k, v, o, lse, H_Q, H_KV, N, D, causal=True)
    cfg_dkv = run_dkv(q, k, v, o, lse, do, delta, H_Q, H_KV, N, D, causal=True)

    print(f"Config: B={B} H_Q={H_Q} H_KV={H_KV} D={D} N={N} causal=True, fp16, H100")
    print(f"  fwd  block: BQ={cfg_fwd[0]}, BKV={cfg_fwd[1]}, warps={cfg_fwd[2]}")
    print(f"  dQ   block: BQ={cfg_dq[0]}, BKV={cfg_dq[1]}, warps={cfg_dq[2]}")
    print(f"  dKV  block: BKV={cfg_dkv[0]}, BQ={cfg_dkv[1]}, warps={cfg_dkv[2]}")
    print()
    print(f"{'kernel':>24} | {'regs':>8} | {'spills':>8} | {'shmem':>33} | config")
    print("-" * 120)

    ck_fwd = get_compiled_kernel(_flash_attn_gqa_kernel)
    ck_delta = get_compiled_kernel(_delta_kernel)
    ck_dq = get_compiled_kernel(_flash_attn_gqa_bwd_dq_kernel)
    ck_dkv = get_compiled_kernel(_flash_attn_gqa_bwd_dkv_packed_kernel)

    pprint("fwd_gqa", ck_fwd)
    pprint("delta", ck_delta)
    pprint("dQ", ck_dq)
    pprint("dKV_packed", ck_dkv)

    print()
    print("Reference:")
    print("  H100 (cc90): 65536 regs/SM, max 255 regs/thread, 232KB dyn shmem/SM")
    print("  Register spill > 0 ⇒ local memory (slow) used; 0 is ideal")
    print("  Occupancy drop kicks in around regs/thread > 128 (warp-level)")


if __name__ == "__main__":
    main()
