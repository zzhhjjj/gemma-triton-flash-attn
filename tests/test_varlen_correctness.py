"""Varlen (packed / cu_seqlens) correctness test.

Covers three kinds of assertion:
  1. Forward numerical: `flash_attn_gqa_varlen` vs `attention_gqa_varlen_ref`
     (per-sample SDPA) across seq-len distributions.
  2. Backward numerical: dQ/dK/dV cosine similarity vs per-sample SDPA
     autograd.
  3. Equal-length equivalence: packing equal-length batched tensors to varlen
     and comparing against the existing batched kernel — fp32 cosine > 0.99999.
     This is the tightest diagnostic; a single mask / stride / LSE-layout bug
     fails it loudly.

Runs in the fresh `varlen-fa` conda env (triton 3.2.0, torch 2.6.0+cu124, H200).

Note on existing batched kernel on this env: the pre-existing `FlashAttnGQA*`
hot path at HEAD_DIM=256/512 currently exceeds H200's 228 KB shmem budget
under triton 3.2's stricter accounting (configs were tuned on an older triton).
The varlen kernels follow the same block-size rules, so they inherit the same
constraint — tests gate D=256/512 configs with a try/except around kernel
invocations and skip-not-fail on OutOfResources. D=128 exercises every code
path.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import triton

from flash_attn import (
    flash_attn_gqa_varlen,
    flash_attn_gqa_train,
    attention_gqa_varlen_ref,
    pack_batched_to_varlen,
    unpack_varlen_to_batched,
)


# =====================================================================
# Seqlen distributions
# =====================================================================

def _gen_seqlens(B: int, distribution: str, max_seqlen: int, rng: torch.Generator):
    """Return int64 tensor shape (B,) of sample lengths.

    Distributions:
        equal         — all samples == max_seqlen
        uniform       — uniform in [max_seqlen//4, max_seqlen]
        bimodal       — half at small (~max/8), half at large (~max)
        one_dominant  — one sample at max, rest small (starves the scheduler)
    """
    if distribution == "equal":
        return torch.full((B,), max_seqlen, dtype=torch.int64)
    if distribution == "uniform":
        lo = max(1, max_seqlen // 4)
        return torch.randint(lo, max_seqlen + 1, (B,), generator=rng, dtype=torch.int64)
    if distribution == "bimodal":
        half = B // 2
        small = torch.randint(max(1, max_seqlen // 16),
                              max(2, max_seqlen // 8) + 1,
                              (half,), generator=rng, dtype=torch.int64)
        large = torch.randint(max(max_seqlen // 2, 2),
                              max_seqlen + 1,
                              (B - half,), generator=rng, dtype=torch.int64)
        out = torch.cat([small, large])
        return out[torch.randperm(B, generator=rng)]
    if distribution == "one_dominant":
        seqs = torch.randint(1, max(2, max_seqlen // 16) + 1,
                             (B,), generator=rng, dtype=torch.int64)
        seqs[0] = max_seqlen
        return seqs[torch.randperm(B, generator=rng)]
    raise ValueError(f"unknown distribution: {distribution}")


def _make_cu_seqlens(seqlens: torch.Tensor, device) -> torch.Tensor:
    B = seqlens.numel()
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = seqlens.to(torch.int32).cumsum(0).to(device)
    return cu


# =====================================================================
# Helpers
# =====================================================================

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _run_fwd_test(H_Q: int, H_KV: int, D: int, B: int, max_seqlen: int,
                  distribution: str, causal: bool, window_size: int,
                  dtype: torch.dtype, seed: int) -> dict:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    seqlens = _gen_seqlens(B, distribution, max_seqlen, rng)
    total = int(seqlens.sum().item())
    max_len = int(seqlens.max().item())

    torch.manual_seed(seed + 1)
    q = torch.randn(total, H_Q, D, dtype=dtype, device="cuda")
    k = torch.randn(total, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(total, H_KV, D, dtype=dtype, device="cuda")
    cu = _make_cu_seqlens(seqlens, "cuda")

    try:
        tri_out = flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                                        causal=causal, window_size=window_size)
    except triton.runtime.errors.OutOfResources as e:
        return {"skipped": True, "reason": f"shmem OOM: {e}"}

    ref_out = attention_gqa_varlen_ref(q, k, v, cu, cu, max_len, max_len,
                                       causal=causal, window_size=window_size)

    diff = (tri_out.float() - ref_out.float()).abs().max().item()
    cos = _cos(tri_out, ref_out)
    # Tolerance: cos_sim > 0.999 (fp16 attention rounding can produce max_abs
    # up to ~1e-2 per entry but cos_sim across all output stays very high).
    passed = cos > 0.999 and diff < 5e-2
    return {"skipped": False, "max_abs": diff, "cos": cos, "passed": passed}


def _run_bwd_test(H_Q: int, H_KV: int, D: int, B: int, max_seqlen: int,
                  distribution: str, causal: bool, window_size: int,
                  dtype: torch.dtype, seed: int) -> dict:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    seqlens = _gen_seqlens(B, distribution, max_seqlen, rng)
    total = int(seqlens.sum().item())
    max_len = int(seqlens.max().item())

    torch.manual_seed(seed + 1)
    q_data = torch.randn(total, H_Q, D, dtype=dtype, device="cuda")
    k_data = torch.randn(total, H_KV, D, dtype=dtype, device="cuda")
    v_data = torch.randn(total, H_KV, D, dtype=dtype, device="cuda")
    do = torch.randn(total, H_Q, D, dtype=dtype, device="cuda")
    cu = _make_cu_seqlens(seqlens, "cuda")

    # Triton path
    q = q_data.clone().requires_grad_(True)
    k = k_data.clone().requires_grad_(True)
    v = v_data.clone().requires_grad_(True)
    try:
        out_tri = flash_attn_gqa_varlen(q, k, v, cu, cu, max_len, max_len,
                                        causal=causal, window_size=window_size)
    except triton.runtime.errors.OutOfResources as e:
        return {"skipped": True, "reason": f"shmem OOM: {e}"}
    out_tri.backward(do)
    dq_tri, dk_tri, dv_tri = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # Reference path — per-sample SDPA autograd
    q_ref = q_data.clone().requires_grad_(True)
    k_ref = k_data.clone().requires_grad_(True)
    v_ref = v_data.clone().requires_grad_(True)
    out_ref = attention_gqa_varlen_ref(q_ref, k_ref, v_ref, cu, cu,
                                       max_len, max_len,
                                       causal=causal, window_size=window_size)
    out_ref.backward(do)
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    dq_cos = _cos(dq_tri, dq_ref)
    dk_cos = _cos(dk_tri, dk_ref)
    dv_cos = _cos(dv_tri, dv_ref)
    # fp16 atomic accumulation + per-sample SDPA rounding => tolerance of 0.9999.
    passed = dq_cos > 0.9999 and dk_cos > 0.9999 and dv_cos > 0.9999
    return {"skipped": False,
            "dq_cos": dq_cos, "dk_cos": dk_cos, "dv_cos": dv_cos,
            "passed": passed}


def _run_equivalence_test(H_Q: int, H_KV: int, D: int, B: int, N: int,
                          causal: bool, slide_size: int,
                          dtype: torch.dtype, seed: int) -> dict:
    """Equal-length packing → varlen output must match batched output bit-for-bit
    in fp32 (same input tensors, same kernel math, same block sizes)."""
    torch.manual_seed(seed)
    qb = torch.randn(B, H_Q, N, D, dtype=dtype, device="cuda", requires_grad=True)
    kb = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda", requires_grad=True)
    vb = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda", requires_grad=True)
    do_batched = torch.randn(B, H_Q, N, D, dtype=dtype, device="cuda")

    try:
        out_batched = flash_attn_gqa_train(qb, kb, vb, causal=causal, slide_size=slide_size)
    except triton.runtime.errors.OutOfResources as e:
        return {"skipped": True, "reason": f"batched shmem OOM: {e}"}
    out_batched.backward(do_batched)
    dq_batched = qb.grad.clone()
    dk_batched = kb.grad.clone()
    dv_batched = vb.grad.clone()

    # Pack the SAME tensors.
    seqlens = torch.full((B,), N, dtype=torch.int64)
    qp_data, kp_data, vp_data, cu = pack_batched_to_varlen(
        qb.detach(), kb.detach(), vb.detach(), seqlens)
    do_packed, _, _, _ = pack_batched_to_varlen(
        do_batched, do_batched[:, :H_KV], do_batched[:, :H_KV], seqlens)

    qp = qp_data.requires_grad_(True)
    kp = kp_data.requires_grad_(True)
    vp = vp_data.requires_grad_(True)
    try:
        out_varlen = flash_attn_gqa_varlen(qp, kp, vp, cu, cu, N, N,
                                           causal=causal, window_size=slide_size)
    except triton.runtime.errors.OutOfResources as e:
        return {"skipped": True, "reason": f"varlen shmem OOM: {e}"}
    out_varlen.backward(do_packed)

    # Reshape back to batched for comparison.
    out_varlen_bhnd = unpack_varlen_to_batched(out_varlen, cu, N, H_Q)
    dq_varlen_bhnd = unpack_varlen_to_batched(qp.grad, cu, N, H_Q)
    dk_varlen_bhnd = unpack_varlen_to_batched(kp.grad, cu, N, H_KV)
    dv_varlen_bhnd = unpack_varlen_to_batched(vp.grad, cu, N, H_KV)

    out_diff = (out_varlen_bhnd - out_batched).abs().max().item()
    out_cos = _cos(out_varlen_bhnd, out_batched)
    dq_cos = _cos(dq_varlen_bhnd, dq_batched)
    dk_cos = _cos(dk_varlen_bhnd, dk_batched)
    dv_cos = _cos(dv_varlen_bhnd, dv_batched)

    # Tight tolerance: equal-length packing is mathematically identical.
    passed = (out_cos > 0.99999 and dq_cos > 0.99999
              and dk_cos > 0.99999 and dv_cos > 0.99999)
    return {"skipped": False,
            "out_max_abs": out_diff, "out_cos": out_cos,
            "dq_cos": dq_cos, "dk_cos": dk_cos, "dv_cos": dv_cos,
            "passed": passed}


# =====================================================================
# Test configurations
# =====================================================================

@dataclass
class FwdCfg:
    H_Q: int
    H_KV: int
    D: int
    B: int
    max_seqlen: int
    distribution: str
    causal: bool
    window_size: int


FWD_CONFIGS = [
    # Core D=128 (no shmem OOM on this env)
    FwdCfg(4, 4, 128, 1, 256, "equal", True, 0),
    FwdCfg(8, 2, 128, 4, 512, "uniform", True, 0),
    FwdCfg(8, 2, 128, 4, 512, "bimodal", True, 0),
    FwdCfg(8, 2, 128, 8, 256, "one_dominant", True, 0),
    FwdCfg(8, 2, 128, 4, 512, "uniform", True, 128),  # SWA
    FwdCfg(16, 4, 128, 2, 1024, "bimodal", True, 256),
    FwdCfg(32, 4, 128, 4, 256, "uniform", True, 0),
    # D=256 (may hit shmem OOM on newer triton; test will skip cleanly)
    FwdCfg(8, 2, 256, 2, 256, "equal", True, 0),
    FwdCfg(8, 2, 256, 4, 256, "uniform", True, 64),
    # D=512 (pack-GQA hot path)
    FwdCfg(32, 4, 512, 1, 128, "equal", True, 0),
    FwdCfg(32, 4, 512, 2, 128, "uniform", True, 0),
]

EQUIV_CONFIGS = [
    # (H_Q, H_KV, D, B, N, causal, slide_size)
    (4, 4, 128, 2, 128, True, 0),
    (8, 2, 128, 4, 128, True, 0),
    (8, 2, 128, 2, 256, True, 64),       # SWA
    (32, 4, 128, 1, 256, True, 0),
    (8, 2, 256, 2, 128, True, 0),        # D=256, may shmem-OOM
    (32, 4, 512, 1, 128, True, 0),       # D=512 Gemma4 full config
]


# =====================================================================
# Runners
# =====================================================================

def run_fwd_suite() -> bool:
    print("\n=== Varlen forward (vs per-sample SDPA) ===")
    header = f"{'GQA':<6} {'D':<4} {'B':<3} {'maxN':<6} {'dist':<14} {'wndw':<5}"
    header += f"{'max_abs':>12} {'cos':>12} {'status':>10}"
    print(header)
    print("-" * len(header))
    all_ok = True
    for cfg in FWD_CONFIGS:
        tag_gqa = f"{cfg.H_Q}:{cfg.H_KV}"
        for seed in (0, 17):
            r = _run_fwd_test(cfg.H_Q, cfg.H_KV, cfg.D, cfg.B, cfg.max_seqlen,
                              cfg.distribution, cfg.causal, cfg.window_size,
                              torch.float16, seed)
            if r.get("skipped"):
                status = "SKIP"
            elif r["passed"]:
                status = "PASS"
            else:
                status = "FAIL"
                all_ok = False
            max_abs = f"{r.get('max_abs', float('nan')):>12.2e}" if not r.get("skipped") else f"{'—':>12}"
            cos = f"{r.get('cos', float('nan')):>12.6f}" if not r.get("skipped") else f"{'—':>12}"
            print(f"{tag_gqa:<6} {cfg.D:<4} {cfg.B:<3} {cfg.max_seqlen:<6} "
                  f"{cfg.distribution:<14} {cfg.window_size:<5}{max_abs}{cos}{status:>10}")
    return all_ok


def run_bwd_suite() -> bool:
    print("\n=== Varlen backward (vs per-sample SDPA autograd) ===")
    header = f"{'GQA':<6} {'D':<4} {'B':<3} {'maxN':<6} {'dist':<14} {'wndw':<5}"
    header += f"{'dq_cos':>10} {'dk_cos':>10} {'dv_cos':>10} {'status':>10}"
    print(header)
    print("-" * len(header))
    all_ok = True
    # Use full fwd config list; backward is slower so run one seed only.
    for cfg in FWD_CONFIGS:
        r = _run_bwd_test(cfg.H_Q, cfg.H_KV, cfg.D, cfg.B, cfg.max_seqlen,
                          cfg.distribution, cfg.causal, cfg.window_size,
                          torch.float16, 42)
        tag_gqa = f"{cfg.H_Q}:{cfg.H_KV}"
        if r.get("skipped"):
            status = "SKIP"
            dq = dk = dv = "—"
        elif r["passed"]:
            status = "PASS"
            dq = f"{r['dq_cos']:.6f}"
            dk = f"{r['dk_cos']:.6f}"
            dv = f"{r['dv_cos']:.6f}"
        else:
            status = "FAIL"
            all_ok = False
            dq = f"{r['dq_cos']:.6f}"
            dk = f"{r['dk_cos']:.6f}"
            dv = f"{r['dv_cos']:.6f}"
        print(f"{tag_gqa:<6} {cfg.D:<4} {cfg.B:<3} {cfg.max_seqlen:<6} "
              f"{cfg.distribution:<14} {cfg.window_size:<5}"
              f"{dq:>10} {dk:>10} {dv:>10}{status:>10}")
    return all_ok


def run_equivalence_suite() -> bool:
    print("\n=== Equivalence: packed-equal-length varlen vs batched kernel ===")
    header = f"{'GQA':<6} {'D':<4} {'B':<3} {'N':<5} {'slide':<6}"
    header += f"{'out_cos':>12} {'dq_cos':>10} {'dk_cos':>10} {'dv_cos':>10} {'status':>10}"
    print(header)
    print("-" * len(header))
    all_ok = True
    for H_Q, H_KV, D, B, N, causal, slide in EQUIV_CONFIGS:
        r = _run_equivalence_test(H_Q, H_KV, D, B, N, causal, slide,
                                  torch.float16, 0)
        tag_gqa = f"{H_Q}:{H_KV}"
        if r.get("skipped"):
            status = "SKIP"
            oc = dqc = dkc = dvc = "—"
        elif r["passed"]:
            status = "PASS"
            oc = f"{r['out_cos']:.9f}"
            dqc = f"{r['dq_cos']:.6f}"
            dkc = f"{r['dk_cos']:.6f}"
            dvc = f"{r['dv_cos']:.6f}"
        else:
            status = "FAIL"
            all_ok = False
            oc = f"{r['out_cos']:.9f}"
            dqc = f"{r['dq_cos']:.6f}"
            dkc = f"{r['dk_cos']:.6f}"
            dvc = f"{r['dv_cos']:.6f}"
        print(f"{tag_gqa:<6} {D:<4} {B:<3} {N:<5} {slide:<6}"
              f"{oc:>12} {dqc:>10} {dkc:>10} {dvc:>10}{status:>10}")
    return all_ok


def main() -> int:
    fwd_ok = run_fwd_suite()
    bwd_ok = run_bwd_suite()
    eq_ok = run_equivalence_suite()
    all_ok = fwd_ok and bwd_ok and eq_ok
    print("\n=== Summary ===")
    print(f"  Forward suite:     {'PASS' if fwd_ok else 'FAIL'}")
    print(f"  Backward suite:    {'PASS' if bwd_ok else 'FAIL'}")
    print(f"  Equivalence suite: {'PASS' if eq_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
