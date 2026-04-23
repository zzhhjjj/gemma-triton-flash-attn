"""B200 dQ + dKV config sweep at the MoE shape (Gemma-4-26B-A4B).

Mirrors b200_moe_fwd_tune.py — each config in its own subprocess so wgmma
faults at D=512 don't take down the sweep.

Targets the production hot path:
  Full causal D=512: H_Q=16 H_KV=8, slide=0
  SWA D=256:        H_Q=16 H_KV=8, slide=1024

Two kernels per shape:
  dQ:  grid (N/BQ, B*H_Q)
  dKV: grid (N/BKV, B*H_KV, Q_SPLITS) — packed GQA inner loop
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

DQ_RUNONE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "b200_moe_bwd_dq_runone.py")
DKV_RUNONE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "b200_moe_bwd_dkv_runone.py")


def run_one(runner, args, timeout=90):
    args_str = [sys.executable, runner] + [str(a) for a in args]
    try:
        r = subprocess.run(args_str, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return float('inf'), 'timeout'
    if r.returncode != 0:
        s = r.stderr.lower()
        if 'misaligned' in s: return float('inf'), 'misaligned'
        if 'out of resource' in s or 'outofresources' in s: return float('inf'), 'oor'
        if 'cuda error' in s or 'cuda_error' in s: return float('inf'), 'cuda'
        return float('inf'), 'error'
    for line in r.stdout.strip().splitlines():
        if line.startswith("RESULT"):
            p = line.split()
            if p[1] == "skip": return float('inf'), 'skip'
            if p[1] == "ok": return float(p[2]), 'ok'
    return float('inf'), 'noparse'


# dQ configs (BQ, BKV, warps, stages). dQ is BQ-major: one program per Q block,
# inner loop streams KV. Current B200 default is (BQ=32 BKV=32 w=8 s=1).
# At D=512 BQ>=64 may fault (same wgmma layout bug as fwd); driver isolates it.
DQ_D512 = [
    (32, 16, 4, 2), (32, 16, 8, 1),
    (32, 32, 4, 2), (32, 32, 8, 1), (32, 32, 8, 2),
    (32, 64, 4, 2), (32, 64, 8, 1),
    (64, 16, 4, 1), (64, 32, 4, 1), (64, 64, 4, 1),
    (64, 16, 8, 1), (64, 32, 8, 1),
]
DQ_D256 = [
    (64, 64, 4, 2), (64, 128, 4, 2),
    (128, 32, 8, 2), (128, 64, 4, 1), (128, 64, 8, 1), (128, 64, 8, 2),
    (128, 128, 8, 1), (128, 128, 8, 2),
]

# dKV (packed): grid is BKV-major; inner loop walks GQA_RATIO*Q heads.
# Current B200 default: D=512 (BKV=16 BQ=32 w=4); D<512 (BKV=64 BQ=128 w=8).
DKV_D512 = [
    (32, 16, 4, 1), (32, 16, 4, 2), (32, 16, 8, 1),
    (32, 32, 4, 1), (32, 32, 4, 2), (32, 32, 8, 1),
    (64, 16, 4, 1), (64, 16, 4, 2), (64, 16, 8, 1),
    (64, 32, 4, 1), (64, 32, 4, 2),
    (16, 32, 4, 1), (16, 32, 8, 1),
]
DQ_QS = 1
DKV_D512 = [(BQ, BKV, w, st, 1) for (BQ, BKV, w, st) in DKV_D512]
DKV_D256 = [
    (BQ, BKV, w, st, 1)
    for (BQ, BKV, w, st) in [
        (64, 64, 4, 1), (64, 64, 4, 2),
        (128, 32, 8, 2), (128, 64, 8, 1), (128, 64, 8, 2),
        (128, 128, 8, 1), (128, 128, 8, 2),
    ]
]


def sweep_kernel(label, runner, configs, H_Q, H_KV, N, D, slide):
    rows = []
    for cfg in configs:
        ms, status = run_one(runner, [H_Q, H_KV, N, D, slide] + list(cfg))
        rows.append((ms, *cfg, status))
    rows_sorted = sorted(rows)
    n_ok = sum(1 for r in rows if r[-1] == 'ok')
    print(f"\n--- {label}  N={N} ---")
    print(f"  Top 5 ok configs:")
    shown = 0
    for r in rows_sorted:
        if shown >= 5: break
        if r[-1] != 'ok': continue
        ms = r[0]; cfg = r[1:-1]
        cfg_s = " ".join(f"{x:>3}" for x in cfg)
        print(f"    {cfg_s}  {ms:.3f} ms")
        shown += 1
    print(f"  {n_ok}/{len(rows)} ok")
    fault_tags = ('misaligned','oor','cuda','error','skip','timeout')
    fs = ", ".join(f"{t}={sum(1 for r in rows if r[-1]==t)}" for t in fault_tags
                   if any(r[-1]==t for r in rows))
    if fs: print(f"  Faults: {fs}")
    return rows_sorted


def main():
    cap = torch.cuda.get_device_capability(0)
    print(f"[smoke] {torch.cuda.get_device_name(0)} cap={cap}")
    H_Q, H_KV = 16, 8  # MoE shape
    # 2 N values keeps sweep ~10-15 min. Best configs are stable across N for
    # the same shape; small/large pair catches grid-occupancy regime change.
    N_list = [4096, 16384]

    results = {"dq_d512": [], "dq_d256": [],
               "dkv_d512": [], "dkv_d256": []}

    # dQ first (no faults at D=256, fewer configs at D=512)
    print("\n" + "="*60)
    print("dQ kernel sweep")
    print("="*60)
    for N in N_list:
        rows = sweep_kernel(f"dQ D=256 SWA slide=1024", DQ_RUNONE, DQ_D256,
                            H_Q, H_KV, N, 256, 1024)
        results["dq_d256"].append({"N": N, "rows": [(r[0], *r[1:]) for r in rows]})
    for N in N_list:
        rows = sweep_kernel(f"dQ D=512 full", DQ_RUNONE, DQ_D512,
                            H_Q, H_KV, N, 512, 0)
        results["dq_d512"].append({"N": N, "rows": [(r[0], *r[1:]) for r in rows]})

    # dKV
    print("\n" + "="*60)
    print("dKV (packed) kernel sweep")
    print("="*60)
    for N in N_list:
        rows = sweep_kernel(f"dKV D=256 SWA slide=1024", DKV_RUNONE, DKV_D256,
                            H_Q, H_KV, N, 256, 1024)
        results["dkv_d256"].append({"N": N, "rows": [(r[0], *r[1:]) for r in rows]})
    for N in N_list:
        rows = sweep_kernel(f"dKV D=512 full", DKV_RUNONE, DKV_D512,
                            H_Q, H_KV, N, 512, 0)
        results["dkv_d512"].append({"N": N, "rows": [(r[0], *r[1:]) for r in rows]})

    print("\n" + "="*60)
    print("BEST configs by N")
    print("="*60)
    for k in ("dq_d256", "dq_d512", "dkv_d256", "dkv_d512"):
        print(f"\n{k}:")
        for entry in results[k]:
            ok = [r for r in entry["rows"] if r[-1] == "ok"]
            if not ok:
                print(f"  N={entry['N']:>6}  no ok")
                continue
            best = min(ok)
            ms = best[0]; cfg = best[1:-1]
            print(f"  N={entry['N']:>6}  {ms:>7.3f} ms   cfg={cfg}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "b200_moe_bwd_tune.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
