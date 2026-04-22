"""Split-D forward attention prototype — Approach A.

Split only the first matmul (QK^T) across HEAD_DIM, keep the second (P@V)
as a single full-D matmul. This is the cheap/easy variant.

Rationale:
  - The first matmul buffers K[BKV, D]; splitting to BD=128 shrinks
    per-stage K SMEM to BKV × 128 × 2 bytes.
  - The second matmul still needs V[BKV, D] in full — if we also D-split
    this, we'd need to update acc[:, d_chunk] per chunk, which Triton
    cannot express as sliced assignment on a 2D tile.

SMEM budget at BQ=64, BKV=64, stages=2, bf16:
  Q full:          64 × 512 × 2 = 64 KB
  K staged (BD=128): 2 × (64 × 128 × 2) = 32 KB
  V staged full:    2 × (64 × 512 × 2) = 128 KB
  Total: 224 KB  (vs 232 KB budget — just fits)

Compare to baseline (BKV=32, stages=2): 192 KB, 7.4 ms @ N=4096.
"""
import math
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import _flash_attn_gqa_kernel


@triton.jit
def _split_d_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sqb, sqh, sqn, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sob, soh, son, sod,
    N_Q_HEADS, N_KV_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh = tl.program_id(1)
    qh = bh % N_Q_HEADS
    b = bh // N_Q_HEADS
    kvh = qh * N_KV_HEADS // N_Q_HEADS

    q_base = Q_ptr + b * sqb + qh * sqh
    k_base = K_ptr + b * skb + kvh * skh
    v_base = V_ptr + b * svb + kvh * svh
    o_base = O_ptr + b * sob + qh * soh

    q_off = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_off < SEQ_LEN
    d_full = tl.arange(0, HEAD_DIM)

    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E

    m_i = tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        kv_end = (q_block_idx + 1) * BLOCK_Q
    else:
        kv_end = SEQ_LEN

    for kv_start in range(0, kv_end, BLOCK_KV):
        kv_off = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_off < SEQ_LEN

        # Split-D QK^T: accumulate scores over BLOCK_D chunks.
        scores = tl.zeros([BLOCK_Q, BLOCK_KV], dtype=tl.float32)
        for d0 in tl.static_range(0, HEAD_DIM, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            q_chunk = tl.load(
                q_base + q_off[:, None] * sqn + d[None, :] * sqd,
                mask=q_mask[:, None], other=0.0)
            k_chunk = tl.load(
                k_base + kv_off[:, None] * skn + d[None, :] * skd,
                mask=kv_mask[:, None], other=0.0)
            scores += tl.dot(q_chunk, tl.trans(k_chunk))

        scores *= scale_log2e
        if IS_CAUSAL:
            valid = (kv_off[None, :] <= q_off[:, None]) & kv_mask[None, :]
            scores = tl.where(valid, scores, -float("inf"))
        else:
            scores = tl.where(kv_mask[None, :], scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, block_max)
        alpha = tl.math.exp2(m_i - new_max)
        p = tl.math.exp2(scores - new_max[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Full-D V matmul.
        v_full = tl.load(
            v_base + kv_off[:, None] * svn + d_full[None, :] * svd,
            mask=kv_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v_full.dtype), v_full)

        m_i = new_max

    acc = acc / l_i[:, None]
    o_ptrs = o_base + q_off[:, None] * son + d_full[None, :] * sod
    tl.store(o_ptrs, acc, mask=q_mask[:, None])


def run_baseline(q, k, v, BQ=64, BKV=32, warps=8, stages=2, causal=True):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    o = torch.empty_like(q)
    lse = torch.empty(B, H_Q, N, dtype=torch.float32, device="cuda")
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
        stride_lsen=lse.stride(2), STORE_LSE=False,
        num_warps=warps, num_stages=stages,
    )
    return o


def run_split(q, k, v, BQ, BKV, BD, warps, stages, causal=True):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BQ), B * H_Q)
    _split_d_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BQ, BLOCK_KV=BKV, BLOCK_D=BD,
        IS_CAUSAL=causal,
        num_warps=warps, num_stages=stages,
    )
    return o


def time_fn(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def get_ck(jit_fn):
    dc = jit_fn.device_caches
    bc, *_ = dc[0]
    return list(bc.values())[-1]


def main():
    B, H_Q, H_KV, D = 1, 32, 4, 512
    torch.manual_seed(0)

    Nc = 256
    qc = torch.randn(B, H_Q, Nc, D, dtype=torch.bfloat16, device="cuda")
    kc = torch.randn(B, H_KV, Nc, D, dtype=torch.bfloat16, device="cuda")
    vc = torch.randn(B, H_KV, Nc, D, dtype=torch.bfloat16, device="cuda")
    o_ref = run_baseline(qc, kc, vc, causal=True)
    o_sp = run_split(qc, kc, vc, BQ=64, BKV=32, BD=128, warps=8, stages=2, causal=True)
    diff = (o_ref.float() - o_sp.float()).abs()
    ok = diff.max().item() < 1e-2
    print(f"[correctness] max|Δ|={diff.max().item():.2e}  mean|Δ|={diff.mean().item():.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        return

    for N in [4096, 8192]:
        print(f"\n=== N={N}, D={D}, H_Q={H_Q}, H_KV={H_KV}, causal, bf16 ===")
        q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")

        print(f"{'config':<42} | {'ms':>7} | {'shmem':>8} | {'regs':>4} | {'spills':>6}")
        print("-" * 92)

        ms = time_fn(lambda: run_baseline(q, k, v, BQ=64, BKV=32, warps=8, stages=2, causal=True))
        ck = get_ck(_flash_attn_gqa_kernel)
        print(f"{'baseline BQ=64 BKV=32 s=2':<42} | {ms:>6.3f} | "
              f"{ck.metadata.shared/1024:>6.1f}KB | {ck.n_regs:>4} | {ck.n_spills:>5}")

        for BQ, BKV, BD, w, st in [
            (64, 32, 128, 8, 2),
            (64, 32, 256, 8, 2),
            (64, 64, 128, 8, 2),
            (64, 64,  64, 8, 2),
            (64, 64, 128, 8, 3),
            (64, 64,  64, 8, 3),
            (64, 128, 128, 8, 2),
            (64, 128,  64, 8, 2),
            (32, 64, 128, 4, 3),
            (32, 128, 128, 4, 2),
        ]:
            try:
                ms = time_fn(lambda: run_split(q, k, v, BQ=BQ, BKV=BKV, BD=BD,
                                               warps=w, stages=st, causal=True))
                ck = get_ck(_split_d_kernel)
                tag = f"split-D BQ={BQ} BKV={BKV} BD={BD} s={st}"
                print(f"{tag:<42} | {ms:>6.3f} | "
                      f"{ck.metadata.shared/1024:>6.1f}KB | {ck.n_regs:>4} | {ck.n_spills:>5}")
            except Exception as e:
                tag = f"split-D BQ={BQ} BKV={BKV} BD={BD} s={st}"
                print(f"{tag:<42} |   FAIL  | {str(e)[:70]}")


if __name__ == "__main__":
    main()
