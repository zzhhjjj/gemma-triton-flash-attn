# Optimization notes

This document records the optimization strategies that were implemented and
measured. One ("pack-GQA dKV") succeeded and is now the default backward
path; two others ("multi-head fusion fwd", "atomic fused bwd") turned out
worse than the baseline and are kept in source only as cautionary reference.

Further quantitative tables live in [`../context/baseline.md`](../context/baseline.md).

## ✅ Pack-GQA style backward (default)

**Borrowed from** `flash-attention/flash_attn/cute/pack_gqa.py` — the CuTeDSL
implementation that FA4 uses on Hopper/Blackwell.

**Idea.** Instead of one program per (KV block, Q head) with atomic-add into
a shared dK/dV tile, one program per KV block **loops over all GQA Q heads
internally** and accumulates into a single `dk_acc / dv_acc` register tile
before writing directly. No atomic, no fp32 expand buffer, no reduce kernel.

**Kernel.** `_flash_attn_gqa_bwd_dkv_packed_kernel` in
[`../flash_attn/attention.py`](../flash_attn/attention.py). Grid shape is
`(cdiv(N, BKV), B * N_KV_HEADS)`; inner `tl.static_range(GQA_RATIO)` loop.

**Results (SWA D=256, H_Q=32, H_KV=16, slide=1024, fp16, H100):**

| N       | Split dKV + reduce | Pack-GQA dKV | Δ       |
|---------|--------------------|--------------|---------|
| 4,096   | 3.64 ms            | 1.91 ms      | **-47%** |
| 8,192   | 7.50 ms            | 3.19 ms      | **-57%** |
| 16,384  | 14.10 ms           | 5.93 ms      | **-58%** |
| 32,768  | 27.20 ms           | 11.67 ms     | **-57%** |

Also saves ~1 GB activation peak at N=32K on Gemma-4-E2B sliding config
because we no longer allocate the expanded `(H_Q, N, D)` fp32 scratch.

## ❌ Multi-head fusion forward (`_flash_attn_gqa_grouped_kernel`)

**Idea.** One program processes `GROUP_SIZE` Q heads that share the same KV
head, so K/V is loaded once and fed to all `GROUP_SIZE` `tl.dot`s — HBM K/V
traffic reduced by `GROUP_SIZE×`.

**Result (Gemma-4-E2B GQA 8:1 shapes):**

| N     | D   | GS=1        | GS=2    | GS=4    |
|-------|-----|-------------|---------|---------|
| 4K    | 256 | **0.36 ms** | 0.85 ms | 1.53 ms |
| 16K   | 256 | **3.27 ms** | 8.39 ms | 16.14 ms |
| 1K    | 512 | **0.14 ms** | 0.24 ms | OOM     |

Every `GROUP_SIZE > 1` was **2–5× slower** than the baseline.

**Root cause.** `GS × BQ × D × 4` fp32 accumulators must stay live across
the KV loop, forcing register spill to local memory. The baseline's L2
cache was already doing most of the K/V reuse implicitly, leaving no HBM
headroom to claw back.

**Triton-specific caveat during implementation.** `list[i] = ...` is not
supported; we had to rebuild tuples via `new_m_is = new_m_is + (new_max,)`
inside the KV loop. That works but is another source of extra register
pressure.

Reproducer: `tests/test_grouped_forward.py`.

## ❌ Fused dQ + dKV backward (`_flash_attn_gqa_bwd_fused_kernel`)

**Idea.** A single kernel computes dQ, dK, dV in one pass. Eliminates
redundant Q@K^T / dO@V^T recomputation and saves a kernel launch + the
reduce step. dK/dV accumulated via `tl.atomic_add` into a shared fp32
scratch buffer (cast to fp16/bf16 at the end).

**Result:**

| N    | D   | Split (ms) | Fused (ms) | Speedup |
|------|-----|------------|------------|---------|
| 1K   | 256 | 0.45       | 2.75       | **0.16×** |
| 16K  | 256 | 7.50       | 62.57      | 0.12×   |
| 8K   | 512 | 75.4       | 490        | 0.15×   |

6–8× **slower**.

**Root cause.** With GQA 8:1, 8 Q heads × N/BQ Q-blocks (≈256 programs at
N=2K) all contend on the same dK/dV tiles. At N=2048 that's ~69M fp32
`atomic_add` ops per bwd call — hardware serialisation of contended atomics
eats all the savings. fp16 atomics would halve the bytes but lose ~2e-2
precision over 8-way accumulation (unacceptable for training).

There was also a transient shmem-budget issue at D=512 (BQ=32, BKV=32
exceeded the 232 KB SM budget); reducing to BQ=16 fixed the launch failure
but didn't help the atomic contention.

Reproducer: `tests/test_fused_backward.py`.

## Takeaway

Under Triton's abstraction, the split-dQ + pack-GQA-dKV design with
L2-level K/V reuse appears to be a local optimum for GQA + SWA shapes on
Hopper. Further improvement would need CUDA-level tools (warp
specialisation, async TMA, cluster barriers) that Triton 3.x exposes only
partially. The upstream `flash-attention/flash_attn/cute/` stack (CuTeDSL)
can go further specifically because it has those primitives.
