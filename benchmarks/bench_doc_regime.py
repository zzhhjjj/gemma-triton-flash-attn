"""Doc-clamped (packed multi-document) regime benchmark for B300.

Reproduces the e2e training regime: 32K packed sequence of ~154-token docs,
doc bounds from position_ids. Reports per-kernel time (fwd / dQ / dKV) via
torch.profiler for both gemma4 geometries, plus numerics vs an fp32
block-diagonal reference at small S.

Env knobs (GTFA_DQ_*, GTFA_DKV_*) are honored by the patched attention.py.
"""

import math
import os

import torch
import torch.nn.functional as F
from flash_attn.attention import doc_bounds_from_position_ids, flash_attn_gqa_train

DEV, DTYPE = "cuda", torch.bfloat16


def packed_position_ids(S, doc_len=154, seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = []
    total = 0
    while total < S:
        n = int(torch.randint(doc_len // 2, doc_len * 2, (1,), generator=g))
        n = min(n, S - total)
        lens.append(n)
        total += n
    pos = torch.cat([torch.arange(n) for n in lens])
    return pos[None].to(DEV), lens


def numerics_check(Hq, Hkv, D, W, S=4096):
    pos, lens = packed_position_ids(S)
    lo, hi = doc_bounds_from_position_ids(pos)
    torch.manual_seed(0)
    q = torch.randn(1, Hq, S, D, device=DEV, dtype=DTYPE, requires_grad=True)
    k = torch.randn(1, Hkv, S, D, device=DEV, dtype=DTYPE, requires_grad=True)
    v = torch.randn(1, Hkv, S, D, device=DEV, dtype=DTYPE, requires_grad=True)
    dout = torch.randn(1, Hq, S, D, device=DEV, dtype=DTYPE)
    out = flash_attn_gqa_train(q, k, v, causal=True, slide_size=W, doc_lo=lo, doc_hi_excl=hi)
    gq, gk, gv = torch.autograd.grad(out, (q, k, v), dout)

    n = Hq // Hkv
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    sc = qf @ kf.repeat_interleave(n, 1).transpose(-1, -2) / math.sqrt(D)
    qi = torch.arange(S, device=DEV)[:, None]
    ki = torch.arange(S, device=DEV)[None, :]
    keep = (ki <= qi) & (lo[0][:, None] <= ki)
    if W:
        keep &= (qi - ki) < W
    ref = F.softmax(sc.masked_fill(~keep, float("-inf")), -1) @ vf.repeat_interleave(n, 1)
    rq, rk, rv = torch.autograd.grad(ref, (qf, kf, vf), dout.float())
    for nm, g_, r in (("out", out, ref), ("dq", gq, rq), ("dk", gk, rk), ("dv", gv, rv)):
        rel = (g_.float() - r).abs().max() / r.abs().max().clamp_min(1e-6)
        print(f"    numerics {nm}: rel-to-max {rel.item():.2e}")


def profile_case(name, Hq, Hkv, D, W, S=32768):
    pos, lens = packed_position_ids(S)
    lo, hi = doc_bounds_from_position_ids(pos)
    torch.manual_seed(0)
    q = torch.randn(1, Hq, S, D, device=DEV, dtype=DTYPE)
    k = torch.randn(1, Hkv, S, D, device=DEV, dtype=DTYPE)
    v = torch.randn(1, Hkv, S, D, device=DEV, dtype=DTYPE)
    dout = torch.randn_like(q)

    def run():
        q_, k_, v_ = (t.detach().requires_grad_(True) for t in (q, k, v))
        o = flash_attn_gqa_train(q_, k_, v_, causal=True, slide_size=W, doc_lo=lo, doc_hi_excl=hi)
        torch.autograd.grad(o, (q_, k_, v_), dout)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            run()
        torch.cuda.synchronize()
    agg = {}
    for e in prof.key_averages():
        if "_flash_attn" in e.key:
            agg[e.key.split("<")[0]] = e.device_time_total / 1e3 / 5
    total = sum(agg.values())
    docs = len(lens)
    print(f"  {name} S={S} docs={docs}: total {total:8.2f} ms/iter | " +
          " | ".join(f"{k.replace('_flash_attn_gqa_', '')}: {v:.2f}" for k, v in sorted(agg.items())))


if __name__ == "__main__":
    print(f"GTFA env: DQ={os.getenv('GTFA_DQ_BQ','-')}/{os.getenv('GTFA_DQ_BKV','-')}/w{os.getenv('GTFA_DQ_W','-')} "
          f"DKV={os.getenv('GTFA_DKV_BKV','-')}/{os.getenv('GTFA_DKV_BQ','-')}/w{os.getenv('GTFA_DKV_W','-')}/s{os.getenv('GTFA_DKV_ST','-')}")
    if os.getenv("GTFA_NUMERICS") == "1":
        print("  [sliding numerics]")
        numerics_check(32, 16, 256, 1024)
        print("  [global numerics]")
        numerics_check(32, 4, 512, 0)
    profile_case("sliding D256 W1024", 32, 16, 256, 1024)
    profile_case("global  D512 causal", 32, 4, 512, 0)
