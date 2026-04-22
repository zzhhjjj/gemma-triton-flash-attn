"""B200 fwd-config sweep for the real Gemma-4-26B-A4B MoE attention shapes.

26B MoE config (from cossim load log): H_Q=16 H_KV=8 (GQA 2:1)
  - Full causal:  D=512, slide=0
  - Sliding:      D=256, slide=1024

Each config runs in its OWN subprocess via b200_moe_fwd_runone.py — the
wgmma misaligned-address fault on B200 D=512 corrupts CUDA driver state
(even empty_cache fails), so isolation is the only way to keep going.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNONE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "b200_moe_fwd_runone.py")


def time_sdpa(B, H_Q, H_KV, N, D, slide, n_warmup=3, n_rep=10):
    q = torch.randn(B, H_Q, N, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda")
    if H_Q != H_KV:
        gqa = H_Q // H_KV
        k = k.repeat_interleave(gqa, dim=1)
        v = v.repeat_interleave(gqa, dim=1)
    if slide > 0:
        idx = torch.arange(N, device="cuda")
        mask = (idx[:, None] - idx[None, :]).abs() < slide
        mask = mask & (idx[:, None] >= idx[None, :])
        attn_kw = {"attn_mask": mask}
    else:
        attn_kw = {"is_causal": True}

    fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, **attn_kw)
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / n_rep
    del q, k, v
    torch.cuda.empty_cache()
    return ms


def run_one_isolated(H_Q, H_KV, N, D, slide, BQ, BKV, w, st, timeout=60):
    """Returns (ms, status) — status is 'ok' / 'fault' / 'skip' / 'error'."""
    args = [sys.executable, RUNONE, str(H_Q), str(H_KV), str(N), str(D),
            str(slide), str(BQ), str(BKV), str(w), str(st)]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return float('inf'), 'timeout'
    if r.returncode != 0:
        # Match the kind of fault — we don't actually need the full stderr.
        if 'misaligned' in r.stderr.lower():
            return float('inf'), 'misaligned'
        if 'OutOfResources' in r.stderr or 'out of resource' in r.stderr.lower():
            return float('inf'), 'oor'
        if 'CUDA error' in r.stderr or 'CUDA_ERROR' in r.stderr:
            return float('inf'), 'cuda'
        return float('inf'), 'error'
    out = r.stdout.strip().splitlines()
    for line in out:
        if line.startswith("RESULT"):
            parts = line.split()
            if parts[1] == "skip":
                return float('inf'), 'skip'
            if parts[1] == "ok":
                return float(parts[2]), 'ok'
    return float('inf'), 'noparse'


# Curated configs.

CONFIGS_D512 = [
    # The known-good baseline + variations of warps/stages within the same block.
    (32, 32, 4, 1),
    (32, 32, 8, 1),   # current B200 default
    (32, 32, 8, 2),
    (32, 32, 16, 1),
    (32, 32, 4, 2),
    (32, 32, 16, 2),
    # The other small blocks that *might* dodge the fault.
    (16, 32, 4, 1),
    (16, 32, 8, 1),
    (16, 64, 4, 1),
    (16, 64, 8, 1),
    (16, 128, 4, 1),
    (16, 128, 8, 1),
    (32, 64, 4, 1),
    (32, 64, 8, 1),
    (32, 16, 4, 1),
    (32, 16, 8, 1),
    # Try larger BQ once each — expected to fault but worth the data.
    (64, 16, 4, 1),
    (64, 32, 4, 1),
    (128, 16, 8, 1),
]

CONFIGS_D256 = [
    # Current default = (BQ=128, BKV=64, w=8, s=2 → s=1 on B200)
    (128, 64, 8, 1),
    (128, 64, 8, 2),
    (128, 64, 4, 1),
    (128, 64, 4, 2),
    # wider Q
    (256, 32, 8, 1),
    (256, 32, 8, 2),
    (256, 64, 8, 1),
    (256, 128, 8, 1),
    # narrower Q
    (64, 64, 4, 1),
    (64, 64, 4, 2),
    (64, 64, 8, 1),
    (64, 128, 4, 1),
    (64, 128, 8, 1),
    (64, 128, 4, 2),
    # smaller BKV
    (128, 32, 8, 1),
    (128, 32, 8, 2),
    (128, 32, 4, 2),
    # bigger BKV
    (128, 128, 8, 1),
    (256, 128, 8, 2),
]


def sweep_one(label, B, H_Q, H_KV, D, slide, N, configs):
    sdpa_ms = time_sdpa(B, H_Q, H_KV, N, D, slide)

    rows = []
    for BQ, BKV, w, st in configs:
        ms, status = run_one_isolated(H_Q, H_KV, N, D, slide, BQ, BKV, w, st)
        rows.append((ms, BQ, BKV, w, st, status))
    rows_sorted = sorted(rows)
    n_ok = sum(1 for r in rows if r[5] == 'ok')

    print(f"\n--- {label}  N={N} ---")
    print(f"  SDPA fwd: {sdpa_ms:.3f} ms")
    print(f"  Top 5 ok configs:")
    for r in rows_sorted[:5]:
        ms, BQ, BKV, w, st, status = r
        if status != 'ok':
            continue
        print(f"    BQ={BQ:>3} BKV={BKV:>3} w={w:>2} s={st}  {ms:.3f} ms  ({sdpa_ms/ms:.2f}x SDPA)")
    print(f"  {n_ok}/{len(rows)} configs ran ok")
    print(f"  Fault summary: " + ", ".join(
        f"{tag}={sum(1 for r in rows if r[5]==tag)}"
        for tag in ('misaligned', 'oor', 'cuda', 'error', 'skip', 'timeout')
        if any(r[5] == tag for r in rows)
    ))
    if n_ok > 0:
        best = rows_sorted[0]
        ms, BQ, BKV, w, st, _ = best
        sp = sdpa_ms / ms
        print(f"  BEST: BQ={BQ} BKV={BKV} w={w} s={st}  →  {ms:.3f} ms  ({sp:.2f}x SDPA)")
        best_dict = {"BQ": BQ, "BKV": BKV, "warps": w, "stages": st, "ms": ms, "sp": sp}
    else:
        best_dict = None

    return {"label": label, "N": N, "sdpa_ms": sdpa_ms, "best": best_dict,
            "all": [{"BQ": r[1], "BKV": r[2], "w": r[3], "s": r[4],
                     "ms": (None if r[0] == float('inf') else r[0]),
                     "status": r[5]} for r in rows]}


def main():
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[smoke] device: {torch.cuda.get_device_name(0)}  cap: {cap}")

    H_Q, H_KV = 16, 8  # Gemma-4-26B-A4B MoE
    N_list = [1024, 2048, 4096, 8192, 16384]

    # Run D=256 SWA first (no faults expected) so we always get that data.
    results = {"D256_swa": [], "D512_full": []}
    for N in N_list:
        results["D256_swa"].append(
            sweep_one("MoE D=256 SWA slide=1024", 1, H_Q, H_KV, 256, 1024, N,
                       CONFIGS_D256))
    for N in N_list:
        results["D512_full"].append(
            sweep_one("MoE D=512 full causal", 1, H_Q, H_KV, 512, 0, N,
                       CONFIGS_D512))

    print("\n" + "=" * 70)
    print("SUMMARY: best configs by N")
    print("=" * 70)
    for cfg_name in ("D256_swa", "D512_full"):
        print(f"\n{cfg_name}:")
        print(f"  {'N':>6} {'SDPA':>7} {'BEST':>7} {'spd':>5}  config")
        for r in results[cfg_name]:
            if r["best"] is None:
                print(f"  {r['N']:>6} {r['sdpa_ms']:>6.3f}    --     -- (no ok config)")
                continue
            b = r["best"]
            print(f"  {r['N']:>6} {r['sdpa_ms']:>6.3f}  {b['ms']:>6.3f} {b['sp']:>5.2f}x  "
                  f"BQ={b['BQ']} BKV={b['BKV']} w={b['warps']} s={b['stages']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "b200_moe_fwd_tune.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
