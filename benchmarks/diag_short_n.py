"""Short-N diagnostics:
  1) dKV grid size vs H100 132 SMs (SM occupancy).
  2) SDPA backend check at N=1K/2K/4K for GQA 32:16 and 8:1 configs.
  3) Launch-overhead floor (empty kernel time) on this machine.
"""
import os
import sys
import math

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H100_SMS = 132


def grid_occupancy():
    print("=" * 90)
    print("dKV grid vs H100 SMs (dKV kernel grid = (cdiv(N, BKV), B*H_KV))")
    print("=" * 90)
    print(f"{'Config':<35} | {'N':>5} | {'grid':>8} | {'SM util':>8}")
    print("-" * 90)
    # Current dKV config for D<512, N<8K: BKV=32
    BKV = 32
    cfgs = [
        ("A. H_Q=32,H_KV=16 (syn Gemma4)", 32, 16),
        ("B. H_Q=8 ,H_KV=1  (Gemma-4-E2B)", 8, 1),
    ]
    for name, HQ, HKV in cfgs:
        for N in [1024, 2048, 4096, 8192]:
            g = triton.cdiv(N, BKV) * HKV
            occ = 100.0 * g / H100_SMS
            print(f"{name:<35} | {N:>5} | {g:>8} | {occ:>6.1f}%")

    print("\n" + "=" * 90)
    print("fwd grid = (cdiv(N, BQ_F=128), B*H_Q)  — compare for reference")
    print("=" * 90)
    for name, HQ, HKV in cfgs:
        for N in [1024, 2048, 4096, 8192]:
            g = triton.cdiv(N, 128) * HQ
            occ = 100.0 * g / H100_SMS
            print(f"{name:<35} | {N:>5} | {g:>8} | {occ:>6.1f}%")

    print("\n" + "=" * 90)
    print("dQ grid = (cdiv(N, BQ=64), B*H_Q)")
    print("=" * 90)
    for name, HQ, HKV in cfgs:
        for N in [1024, 2048, 4096, 8192]:
            g = triton.cdiv(N, 64) * HQ
            occ = 100.0 * g / H100_SMS
            print(f"{name:<35} | {N:>5} | {g:>8} | {occ:>6.1f}%")


def sdpa_backend():
    """Detect which SDPA backend runs for GQA short-N."""
    from flash_attn.attention import attention_gqa_ref
    print("\n" + "=" * 90)
    print("SDPA backend detection (ref expands KV via repeat_interleave)")
    print("=" * 90)
    device = "cuda"
    dtype = torch.float16

    from torch.nn.attention import sdpa_kernel, SDPBackend
    backends = {
        "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
        "MATH": SDPBackend.MATH,
        "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
    }

    def time_cuda(fn, warmup=10, rep=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(rep):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts)//2]

    cfgs = [
        ("A syn  32:16", 1, 32, 16, 256),
        ("B E2B   8:1 ", 1, 8,  1,  256),
    ]
    for tag, B, HQ, HKV, D in cfgs:
        print(f"\nconfig: {tag}")
        hdr = f"{'N':>5} | " + " | ".join(f"{n:>12}" for n in backends) + f" | {'default':>12}"
        print(hdr); print("-" * len(hdr))
        for N in [1024, 2048, 4096]:
            q = torch.randn(B, HQ, N, D, dtype=dtype, device=device)
            k = torch.randn(B, HKV, N, D, dtype=dtype, device=device)
            v = torch.randn(B, HKV, N, D, dtype=dtype, device=device)

            row = [f"{N:>5}"]
            for bname, b in backends.items():
                try:
                    with sdpa_kernel([b]):
                        t = time_cuda(lambda: attention_gqa_ref(q, k, v, causal=True))
                    row.append(f"{t:>10.3f}ms")
                except Exception as ex:
                    row.append(f"{'N/A':>12}")
            # default
            t_def = time_cuda(lambda: attention_gqa_ref(q, k, v, causal=True))
            row.append(f"{t_def:>10.3f}ms")
            print(" | ".join(row))


def launch_overhead_floor():
    """Measure an empty / near-empty Triton kernel launch overhead floor."""
    print("\n" + "=" * 90)
    print("Launch overhead floor (noop Triton kernel)")
    print("=" * 90)

    @triton.jit
    def _noop_kernel(x_ptr):
        pid = tl.program_id(0)
        # Minimum work to avoid DCE
        tl.store(x_ptr + pid, pid)

    x = torch.zeros(65536, dtype=torch.int32, device="cuda")

    def t(grid_n, rep=300):
        fn = lambda: _noop_kernel[(grid_n,)](x)
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(rep):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts)//2]

    for g in [1, 8, 32, 64, 128, 256]:
        print(f"  grid={g:>4}  t={t(g)*1000:>6.1f}us")


def main():
    grid_occupancy()
    sdpa_backend()
    launch_overhead_floor()


if __name__ == "__main__":
    main()
