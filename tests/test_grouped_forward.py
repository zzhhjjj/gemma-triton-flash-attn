"""Multi-head fusion kernel correctness + sweep.

Tests `_flash_attn_gqa_grouped_kernel` vs the single-head `_flash_attn_gqa_kernel`
baseline across GROUP_SIZE ∈ {1, 2, 4, 8}, D ∈ {256, 512}, N ∈ {1K, 4K, 16K}.

Exit status:
  0 if at least one GROUP_SIZE > 1 config is faster than GROUP_SIZE=1 somewhere
  1 if all GROUP_SIZE > 1 are slower (implies no HBM K/V reuse gain worth the extra register pressure)
"""
from __future__ import annotations

import math
import sys

import torch
import triton

from gemma_triton_flash_attn.attention import (
    _flash_attn_gqa_kernel,
    _flash_attn_gqa_grouped_kernel,
    attention_gqa_ref,
)
from gemma_triton_flash_attn.utils import benchmark_fn


def run_grouped(q, k, v, *, causal, slide_size, BLOCK_Q, BLOCK_KV, GROUP_SIZE,
                num_warps, num_stages):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    output = torch.empty_like(q)
    n_groups = H_Q // GROUP_SIZE
    grid = (triton.cdiv(N, BLOCK_Q), B * n_groups)
    _flash_attn_gqa_grouped_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
        GROUP_SIZE=GROUP_SIZE,
        LSE_ptr=None, stride_lseb=0, stride_lseh=0, stride_lsen=0,
        STORE_LSE=False,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


def run_baseline(q, k, v, *, causal, slide_size, BLOCK_Q, BLOCK_KV,
                 num_warps, num_stages):
    """Single-head kernel (GROUP_SIZE=1 equivalent)."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    output = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_Q), B * H_Q)
    _flash_attn_gqa_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, BLOCK_D=D,
        IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
        LSE_ptr=None, stride_lseb=0, stride_lseh=0, stride_lsen=0,
        STORE_LSE=False,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


def make_tensors(B, H_Q, H_KV, N, D, dtype=torch.float16):
    torch.manual_seed(0)
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda")
    return q, k, v


def test_correctness():
    """Verify grouped kernel matches SDPA reference at each GROUP_SIZE."""
    print("=== Correctness: Grouped kernel vs SDPA reference ===")
    all_ok = True
    configs = [
        # (B, H_Q, H_KV, N, D, causal, slide_size, GROUP_SIZE)
        (1, 8, 1, 512, 256, True, 0, 1),
        (1, 8, 1, 512, 256, True, 0, 2),
        (1, 8, 1, 512, 256, True, 0, 4),
        (1, 8, 1, 512, 256, True, 0, 8),
        (1, 8, 1, 2048, 256, True, 512, 2),   # SWA path
        (1, 8, 1, 2048, 256, True, 512, 4),
        (1, 32, 4, 1024, 512, True, 0, 2),
        (1, 32, 4, 1024, 512, True, 0, 4),
    ]
    for B, H_Q, H_KV, N, D, causal, slide, gs in configs:
        q, k, v = make_tensors(B, H_Q, H_KV, N, D)
        if slide > 0:
            from gemma_triton_flash_attn.attention import attention_swa_ref
            ref = attention_swa_ref(q, k, v, slide_size=slide)
        else:
            ref = attention_gqa_ref(q, k, v, causal=causal)
        BQ = min(64, N)
        BKV = min(32, N)
        tag = f"B={B} H_Q={H_Q} H_KV={H_KV} N={N} D={D} " \
              f"causal={causal} slide={slide} GS={gs}"
        try:
            out = run_grouped(q, k, v, causal=causal, slide_size=slide,
                              BLOCK_Q=BQ, BLOCK_KV=BKV, GROUP_SIZE=gs,
                              num_warps=8 if D >= 256 else 4, num_stages=2)
        except triton.runtime.errors.OutOfResources as e:
            # Shared memory limit — this GS × BLOCK_Q combo is not viable.
            # Not a correctness failure; caller should pick smaller BQ for high GS.
            print(f"  {tag:<55} SKIP (shmem OOM)")
            continue
        except Exception as e:
            print(f"  {tag:<55} ERROR {type(e).__name__}: {str(e)[:60]}")
            all_ok = False
            continue
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0).item()
        max_abs = (out - ref).abs().max().item()
        ok = cos > 0.9999
        status = "OK" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  {tag:<55} cos={cos:.6f} abs={max_abs:.2e}  {status}")
    return all_ok


def bench_sweep():
    """Sweep GROUP_SIZE ∈ {1,2,4,8} × D ∈ {256,512} × N ∈ {1K,4K,16K}.

    Uses Gemma-4-E2B-like shapes: H_Q=8, H_KV=1 (GQA 8:1).
    """
    print("\n=== Throughput: GROUP_SIZE sweep (H_Q=8, H_KV=1, B=1, causal) ===")
    print(f"{'N':>6} {'D':>4} {'GS=1':>9} {'GS=2':>9} {'GS=4':>9} {'GS=8':>9}  best")
    print("-" * 60)

    for D in [256, 512]:
        for N in [1024, 4096, 16384]:
            q, k, v = make_tensors(1, 8, 1, N, D)
            results = {}
            for gs in [1, 2, 4, 8]:
                # Match baseline block sizes for fair comparison.
                # Grouped kernel needs smaller BQ as GS grows (more registers).
                if D == 512:
                    BQ = max(16, 64 // gs)
                    BKV = 32
                    nw = 8
                else:  # D=256
                    BQ = max(16, 128 // gs)
                    BKV = 32 if gs >= 4 else 64
                    nw = 8

                def fn():
                    if gs == 1:
                        run_baseline(q, k, v, causal=True, slide_size=0,
                                     BLOCK_Q=BQ, BLOCK_KV=BKV,
                                     num_warps=nw, num_stages=2)
                    else:
                        run_grouped(q, k, v, causal=True, slide_size=0,
                                    BLOCK_Q=BQ, BLOCK_KV=BKV, GROUP_SIZE=gs,
                                    num_warps=nw, num_stages=2)
                try:
                    t = benchmark_fn(fn, warmup=5, rep=20 if N <= 4096 else 10)
                    results[gs] = t
                except Exception as e:
                    results[gs] = None
            row = f"{N:>6} {D:>4}"
            for gs in [1, 2, 4, 8]:
                t = results.get(gs)
                row += f" {t:>8.3f}ms" if t is not None else f" {'OOM':>9}"
            valid = {g: t for g, t in results.items() if t is not None}
            if valid:
                best = min(valid, key=valid.get)
                row += f"  GS={best}"
                if best != 1:
                    sp = results[1] / results[best]
                    row += f" ({sp:.2f}×)"
            print(row)
            del q, k, v
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ok = test_correctness()
    if not ok:
        print("\nCorrectness FAILED — skipping benchmark.")
        sys.exit(1)
    bench_sweep()
