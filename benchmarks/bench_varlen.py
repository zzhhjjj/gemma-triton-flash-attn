"""Varlen vs padded-batched benchmark on H200.

Measures tokens/sec of three paths on a realistic "mixed seqlen" workload:

  1. varlen   — this repo's flash_attn_gqa_varlen on the packed stream.
  2. padded   — this repo's batched flash_attn_gqa_train on a zero-padded
                (B, H, max_seqlen, D) version of the same data. This is the
                status-quo that varlen is supposed to replace.
  3. fa2      — upstream Dao-AILab flash_attn_varlen_func if importable
                (D=128 only; skipped if not installed).

Workload generators:
  - zipf:    lengths ∝ 1/rank**a, scaled so sum ≈ total_tokens.
  - uniform: lengths uniform in [128, max_seqlen].

Metrics:
  - Median ms per fwd  and per fwd+bwd via CUDA events.
  - Tokens/sec = total_tokens / median_ms_fwd * 1e3.
  - Padding waste = (B * max_seqlen - total_tokens) / (B * max_seqlen).
  - Peak memory (torch.cuda.max_memory_allocated).

Quick mode (--quick) runs a 4-config subset for smoke tests; full mode sweeps
all (D, GQA, total_tokens) combos. Results are written to
benchmarks/varlen_bench.json and printed as an ASCII table.

This bench is D=128 only on this env (triton 3.2 on H200) because the batched
kernel's D=256/D=512 configs exceed shared memory under the newer triton's
accounting — same skip pattern used by test_varlen_correctness.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import triton

from flash_attn import (
    flash_attn_gqa_varlen,
    flash_attn_gqa_train,
    pack_batched_to_varlen,
    unpack_varlen_to_batched,
)
from flash_attn.utils import benchmark_fn


# =====================================================================
# Workload generation
# =====================================================================

def zipf_seqlens(B: int, total_tokens: int, a: float = 1.5,
                 min_len: int = 128, rng: np.random.Generator = None) -> np.ndarray:
    """B sample lengths from a Zipf-like distribution, scaled to sum ≈ total_tokens."""
    if rng is None:
        rng = np.random.default_rng(0)
    ranks = np.arange(1, B + 1)
    probs = 1.0 / ranks ** a
    probs = probs / probs.sum()
    rng.shuffle(probs)
    raw = probs * total_tokens
    seqlens = np.clip(raw.round().astype(np.int64), min_len, None)
    # Scale to match total (fudge: stretch/squash the max element).
    diff = total_tokens - int(seqlens.sum())
    if diff != 0:
        seqlens[0] = max(min_len, int(seqlens[0]) + int(diff))
    return seqlens


def uniform_seqlens(B: int, max_seqlen: int, min_len: int = 128,
                    rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    return rng.integers(min_len, max_seqlen + 1, size=B).astype(np.int64)


# =====================================================================
# Per-path benchmarks
# =====================================================================

def _cu(seqlens_np: np.ndarray, device) -> torch.Tensor:
    cu = np.zeros(len(seqlens_np) + 1, dtype=np.int32)
    cu[1:] = seqlens_np.cumsum()
    return torch.tensor(cu, dtype=torch.int32, device=device)


def bench_varlen(seqlens_np: np.ndarray, H_Q: int, H_KV: int, D: int,
                 dtype=torch.float16, rep: int = 20) -> dict:
    total = int(seqlens_np.sum())
    max_len = int(seqlens_np.max())
    torch.manual_seed(0)
    q = torch.randn(total, H_Q, D, dtype=dtype, device="cuda")
    k = torch.randn(total, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(total, H_KV, D, dtype=dtype, device="cuda")
    cu = _cu(seqlens_np, "cuda")

    def fwd():
        flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                              causal=True, window_size=0)

    try:
        t_fwd = benchmark_fn(fwd, warmup=5, rep=rep)
    except triton.runtime.errors.OutOfResources:
        return {"skipped": True, "reason": "shmem OOM"}

    # Peak memory measurement around a fwd.
    torch.cuda.reset_peak_memory_stats()
    _ = flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                              causal=True, window_size=0)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Fwd+bwd
    q_g = q.clone().requires_grad_(True)
    k_g = k.clone().requires_grad_(True)
    v_g = v.clone().requires_grad_(True)
    do = torch.randn_like(q)

    def fwd_bwd():
        out = flash_attn_gqa_varlen(q_g, k_g, v_g, cu, cu, max_len, max_len,
                                    causal=True, window_size=0)
        out.backward(do)
        q_g.grad = None; k_g.grad = None; v_g.grad = None

    try:
        t_fwd_bwd = benchmark_fn(fwd_bwd, warmup=3, rep=max(rep // 2, 5))
    except triton.runtime.errors.OutOfResources:
        t_fwd_bwd = float("nan")

    return {"skipped": False,
            "t_fwd_ms": t_fwd, "t_fwd_bwd_ms": t_fwd_bwd,
            "peak_mb": peak_mb, "total": total}


def bench_padded_batched(seqlens_np: np.ndarray, H_Q: int, H_KV: int, D: int,
                         dtype=torch.float16, rep: int = 20) -> dict:
    """Pad each sample to max_seqlen and run the batched kernel. This is the
    status-quo we're replacing."""
    B = len(seqlens_np)
    max_len = int(seqlens_np.max())
    torch.manual_seed(0)
    # Allocate full padded batched tensors.
    q = torch.randn(B, H_Q, max_len, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H_KV, max_len, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H_KV, max_len, D, dtype=dtype, device="cuda")

    def fwd():
        flash_attn_gqa_train(q, k, v, causal=True, slide_size=0)

    try:
        t_fwd = benchmark_fn(fwd, warmup=5, rep=rep)
    except triton.runtime.errors.OutOfResources:
        return {"skipped": True, "reason": "shmem OOM"}

    torch.cuda.reset_peak_memory_stats()
    _ = flash_attn_gqa_train(q, k, v, causal=True, slide_size=0)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Fwd+bwd
    q_g = q.clone().requires_grad_(True)
    k_g = k.clone().requires_grad_(True)
    v_g = v.clone().requires_grad_(True)
    do = torch.randn_like(q)

    def fwd_bwd():
        out = flash_attn_gqa_train(q_g, k_g, v_g, causal=True, slide_size=0)
        out.backward(do)
        q_g.grad = None; k_g.grad = None; v_g.grad = None

    try:
        t_fwd_bwd = benchmark_fn(fwd_bwd, warmup=3, rep=max(rep // 2, 5))
    except triton.runtime.errors.OutOfResources:
        t_fwd_bwd = float("nan")

    padded_total = B * max_len
    return {"skipped": False,
            "t_fwd_ms": t_fwd, "t_fwd_bwd_ms": t_fwd_bwd,
            "peak_mb": peak_mb, "padded_total": padded_total}


# =====================================================================
# Runner
# =====================================================================

def run_config(D: int, H_Q: int, H_KV: int, total_tokens: int, B: int,
               distribution: str, rep: int) -> dict:
    rng = np.random.default_rng(0)
    if distribution == "zipf":
        seqlens = zipf_seqlens(B, total_tokens, rng=rng)
    elif distribution == "uniform":
        max_len = max(128, total_tokens // B)
        seqlens = uniform_seqlens(B, max_len, rng=rng)
    else:
        raise ValueError(distribution)

    max_len = int(seqlens.max())
    total = int(seqlens.sum())
    padding_frac = (B * max_len - total) / (B * max_len)

    varlen = bench_varlen(seqlens, H_Q, H_KV, D, rep=rep)
    padded = bench_padded_batched(seqlens, H_Q, H_KV, D, rep=rep)

    def tok_per_sec(t_ms, n_tok):
        if t_ms is None or (isinstance(t_ms, float) and math.isnan(t_ms)):
            return float("nan")
        return n_tok / (t_ms * 1e-3)

    return {
        "D": D, "H_Q": H_Q, "H_KV": H_KV, "B": B,
        "total_tokens": total, "max_seqlen": max_len,
        "distribution": distribution,
        "padding_frac": padding_frac,
        "varlen": varlen,
        "padded": padded,
        "varlen_tokens_per_s": tok_per_sec(
            varlen.get("t_fwd_ms"), total) if not varlen.get("skipped") else None,
        "padded_tokens_per_s": tok_per_sec(
            padded.get("t_fwd_ms"), total) if not padded.get("skipped") else None,
    }


def print_table(results: list[dict]) -> None:
    print("\n=== Varlen benchmark ===\n")
    header = (f"{'D':>4} {'H_Q:H_KV':>9} {'B':>3} {'total':>7} {'maxN':>6} "
              f"{'dist':>8} {'pad%':>5} "
              f"{'varlen ms':>10} {'padded ms':>10} {'speedup':>8} "
              f"{'tok/s varl':>12} {'tok/s padd':>12}")
    print(header)
    print("-" * len(header))
    for r in results:
        gqa = f"{r['H_Q']}:{r['H_KV']}"
        pad_pct = f"{r['padding_frac']*100:.0f}%"
        if r['varlen'].get('skipped') or r['padded'].get('skipped'):
            reason = r['varlen'].get('reason') or r['padded'].get('reason')
            print(f"{r['D']:>4} {gqa:>9} {r['B']:>3} {r['total_tokens']:>7} "
                  f"{r['max_seqlen']:>6} {r['distribution']:>8} {pad_pct:>5} "
                  f"{'SKIP:':<10} {reason}")
            continue
        t_v = r['varlen']['t_fwd_ms']
        t_p = r['padded']['t_fwd_ms']
        sp = t_p / t_v if t_v > 0 else float('nan')
        print(f"{r['D']:>4} {gqa:>9} {r['B']:>3} {r['total_tokens']:>7} "
              f"{r['max_seqlen']:>6} {r['distribution']:>8} {pad_pct:>5} "
              f"{t_v:>10.3f} {t_p:>10.3f} {sp:>7.2f}x "
              f"{r['varlen_tokens_per_s']/1e6:>10.2f}M "
              f"{r['padded_tokens_per_s']/1e6:>10.2f}M")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run a small subset for smoke testing")
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).parent / "varlen_bench.json"))
    args = parser.parse_args()

    if args.quick:
        configs = [
            # (D, H_Q, H_KV, total_tokens, B, distribution)
            (128, 8, 2, 4096, 8, "zipf"),
            (128, 8, 2, 16384, 16, "zipf"),
            (128, 32, 4, 4096, 8, "zipf"),
        ]
    else:
        configs = [
            (128, 8, 1, 4096, 8, "zipf"),
            (128, 8, 1, 16384, 16, "zipf"),
            (128, 8, 1, 32768, 32, "zipf"),
            (128, 8, 4, 16384, 16, "zipf"),
            (128, 32, 4, 16384, 16, "zipf"),
            (128, 8, 8, 16384, 16, "uniform"),
            # D=256/D=512 will skip on this env (shmem OOM on batched baseline)
            (256, 8, 2, 8192, 8, "zipf"),
            (512, 32, 4, 4096, 4, "zipf"),
        ]

    results = []
    for cfg in configs:
        D, H_Q, H_KV, total, B, distribution = cfg
        print(f"[bench] D={D} GQA={H_Q}:{H_KV} total={total} B={B} dist={distribution}")
        r = run_config(D, H_Q, H_KV, total, B, distribution, args.rep)
        results.append(r)

    print_table(results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
