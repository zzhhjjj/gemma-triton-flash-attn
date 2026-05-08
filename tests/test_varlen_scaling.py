"""Varlen correctness + efficiency at scale.

Sweeps total-token count N from 2K to 256K in steps of 2K (128 data points).
At each N, builds a packed stream of Zipf-distributed sample lengths summing
to N (each sample capped at 4K so the SDPA reference stays tractable), then:

  * Correctness: varlen output vs per-sample SDPA reference. fp32 cos_sim
    must exceed 0.999 and max_abs_diff stays within fp16 rounding bounds.
  * Efficiency: median fwd time of varlen vs padded-batched (same data
    reshaped into (B, H, max_seqlen, D)). Reports tokens/sec and speedup.

Runs at HEAD_DIM=128 GQA 8:2 to keep the 128-point sweep tractable (D=256/
D=512 configs remain covered by test_varlen_correctness.py).

Runtime: ~3-5 min on H200 (varlen-fa conda env).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton

from flash_attn import (
    flash_attn_gqa_varlen,
    flash_attn_gqa_train,
    attention_gqa_varlen_ref,
    unpack_varlen_to_batched,
)
from flash_attn.utils import benchmark_fn


# Cap per-sample length so the SDPA reference stays cheap even when total N is
# large. Packed training data hits this distribution shape naturally.
MAX_SAMPLE_LEN = 4096


def _zipf_seqlens(total: int, a: float = 1.5, min_len: int = 64,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Return sample lengths (int64) that sum to ~total, each in
    [min_len, MAX_SAMPLE_LEN], roughly Zipf-distributed."""
    if rng is None:
        rng = np.random.default_rng(0)
    # Each sample averages ~1/2 of the cap; rounding down means more samples.
    B_guess = max(1, total // (MAX_SAMPLE_LEN // 2))
    ranks = np.arange(1, B_guess + 1)
    probs = 1.0 / ranks ** a
    probs /= probs.sum()
    rng.shuffle(probs)
    raw = probs * total
    seqlens = np.clip(raw.round().astype(np.int64), min_len, MAX_SAMPLE_LEN)

    # Fix up total — absorb / redistribute the rounding delta.
    diff = int(total - seqlens.sum())
    if diff > 0:
        # Add tokens to samples that still have headroom.
        idx = np.argsort(MAX_SAMPLE_LEN - seqlens)[::-1]
        i = 0
        while diff > 0 and i < len(idx):
            room = MAX_SAMPLE_LEN - int(seqlens[idx[i]])
            add = min(room, diff)
            seqlens[idx[i]] += add
            diff -= add
            i += 1
        if diff > 0:
            # Still short: append more samples.
            extra = min(MAX_SAMPLE_LEN, diff)
            seqlens = np.append(seqlens, extra)
            diff -= extra
            while diff > 0:
                extra = min(MAX_SAMPLE_LEN, diff)
                seqlens = np.append(seqlens, extra)
                diff -= extra
    elif diff < 0:
        # Trim from the longest sample(s).
        idx = np.argsort(seqlens)[::-1]
        i = 0
        while diff < 0 and i < len(idx):
            slack = int(seqlens[idx[i]]) - min_len
            cut = min(slack, -diff)
            seqlens[idx[i]] -= cut
            diff += cut
            i += 1
    assert seqlens.sum() == total, (seqlens.sum(), total)
    assert seqlens.min() >= min_len
    assert seqlens.max() <= MAX_SAMPLE_LEN
    return seqlens


def _make_cu(seqlens: np.ndarray, device) -> torch.Tensor:
    cu = np.zeros(len(seqlens) + 1, dtype=np.int32)
    cu[1:] = seqlens.cumsum()
    return torch.tensor(cu, dtype=torch.int32, device=device)


def run_one(total_tokens: int, H_Q: int, H_KV: int, D: int,
            dtype: torch.dtype, rep: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + total_tokens)
    seqlens = _zipf_seqlens(total_tokens, rng=rng)
    B = len(seqlens)
    max_len = int(seqlens.max())
    total = int(seqlens.sum())

    torch.manual_seed(seed)
    q = torch.randn(total, H_Q, D, dtype=dtype, device="cuda")
    k = torch.randn(total, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(total, H_KV, D, dtype=dtype, device="cuda")
    cu = _make_cu(seqlens, "cuda")

    # --- correctness (fwd only; bwd at 128 points would take hours) ---
    try:
        tri_out = flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                                        causal=True, window_size=0)
    except triton.runtime.errors.OutOfResources as e:
        return {"total": total, "B": B, "max_len": max_len,
                "skipped": True, "reason": f"shmem OOM: {e}"}

    ref_out = attention_gqa_varlen_ref(q, k, v, cu, cu, max_len, max_len,
                                       causal=True, window_size=0)
    max_abs = (tri_out.float() - ref_out.float()).abs().max().item()
    cos = F.cosine_similarity(tri_out.float().flatten(),
                              ref_out.float().flatten(), dim=0).item()
    correct = cos > 0.999 and max_abs < 5e-2

    del ref_out
    torch.cuda.empty_cache()

    # --- efficiency: varlen fwd ---
    def varlen_fwd():
        flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                              causal=True, window_size=0)
    try:
        t_varlen = benchmark_fn(varlen_fwd, warmup=2, rep=rep)
    except triton.runtime.errors.OutOfResources:
        t_varlen = float("nan")

    # --- efficiency: padded-batched baseline (fit-check, then bench) ---
    # Reshape the packed stream into (B, H, max_len, D) with zero padding.
    # For large (B, max_len), this allocation can itself be huge; skip with nan
    # if it would exceed ~20 GB.
    batched_mem_gb = B * (H_Q + 2 * H_KV) * max_len * D * dtype.itemsize / 1e9
    if batched_mem_gb > 20.0:
        t_padded = float("nan")
        skipped_padded = "too large to alloc"
    else:
        q_bhnd = unpack_varlen_to_batched(q, cu, max_len, H_Q)
        k_bhnd = unpack_varlen_to_batched(k, cu, max_len, H_KV)
        v_bhnd = unpack_varlen_to_batched(v, cu, max_len, H_KV)

        def padded_fwd():
            flash_attn_gqa_train(q_bhnd, k_bhnd, v_bhnd, causal=True, slide_size=0)

        try:
            t_padded = benchmark_fn(padded_fwd, warmup=2, rep=rep)
            skipped_padded = None
        except triton.runtime.errors.OutOfResources:
            t_padded = float("nan")
            skipped_padded = "shmem OOM"

        del q_bhnd, k_bhnd, v_bhnd
        torch.cuda.empty_cache()

    del q, k, v, cu, tri_out
    torch.cuda.empty_cache()

    padded_total = B * max_len
    padding_frac = (padded_total - total) / padded_total
    speedup = (t_padded / t_varlen) if (t_varlen == t_varlen and t_padded == t_padded) else float("nan")
    tok_per_s = total / (t_varlen * 1e-3) if t_varlen == t_varlen else float("nan")

    return {
        "total": total, "B": B, "max_len": max_len,
        "skipped": False,
        "max_abs": max_abs, "cos": cos, "correct": correct,
        "t_varlen_ms": t_varlen, "t_padded_ms": t_padded,
        "speedup": speedup,
        "padding_frac": padding_frac,
        "varlen_tokens_per_s": tok_per_s,
        "skipped_padded": skipped_padded,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2048)
    parser.add_argument("--stop", type=int, default=262144,
                        help="inclusive upper bound")
    parser.add_argument("--step", type=int, default=2048)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--H_Q", type=int, default=8)
    parser.add_argument("--H_KV", type=int, default=2)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).parent.parent /
                                    "benchmarks" / "varlen_scaling.json"))
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 8 points at 2K, 32K, 64K, ..., 256K")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if args.quick:
        ns = [2048, 16384, 32768, 65536, 131072, 196608, 262144]
    else:
        ns = list(range(args.start, args.stop + 1, args.step))
    print(f"[scaling] {len(ns)} points from {args.start} to {args.stop} step {args.step}")
    print(f"[scaling] config: H_Q={args.H_Q} H_KV={args.H_KV} D={args.D} "
          f"dtype={args.dtype}")

    header = (f"{'total':>7} {'B':>4} {'maxN':>5} {'pad%':>5}"
              f" {'max_abs':>9} {'cos':>10} {'corr':>5}"
              f" {'varlen ms':>10} {'padded ms':>10} {'speedup':>8}"
              f" {'tok/s':>10}")
    print(header)
    print("-" * len(header))

    results = []
    fail_ct = 0
    t_start = time.time()
    for N in ns:
        r = run_one(N, args.H_Q, args.H_KV, args.D, dtype, args.rep, seed=0)
        if r.get("skipped"):
            print(f"{r['total']:>7} {r['B']:>4} {r['max_len']:>5}"
                  f"  SKIP {r['reason']}")
            results.append(r)
            continue
        pad_pct = f"{r['padding_frac']*100:.0f}%"
        corr_tag = "OK" if r["correct"] else "FAIL"
        if not r["correct"]:
            fail_ct += 1
        v = r["t_varlen_ms"]
        p = r["t_padded_ms"]
        sp = r["speedup"]
        tps = r["varlen_tokens_per_s"]
        v_str = f"{v:>10.3f}" if v == v else f"{'—':>10}"
        p_str = f"{p:>10.3f}" if p == p else f"{'—':>10}"
        sp_str = f"{sp:>7.2f}x" if sp == sp else f"{'—':>8}"
        tps_str = f"{tps/1e6:>8.2f}M" if tps == tps else f"{'—':>9}"
        print(f"{r['total']:>7} {r['B']:>4} {r['max_len']:>5} {pad_pct:>5}"
              f" {r['max_abs']:>9.2e} {r['cos']:>10.6f} {corr_tag:>5}"
              f" {v_str} {p_str} {sp_str} {tps_str}")
        results.append(r)
    elapsed = time.time() - t_start
    print(f"[scaling] done in {elapsed:.1f}s, {fail_ct} correctness FAILs of "
          f"{len([r for r in results if not r.get('skipped')])} points")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f,
                  indent=2, default=str)
    print(f"[scaling] wrote {args.out}")

    return 0 if fail_ct == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
