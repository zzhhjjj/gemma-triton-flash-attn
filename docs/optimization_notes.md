# Optimization notes

This document records the optimization strategies that were implemented and
measured. Wins that shipped: pack-GQA dKV kernel, softmax `exp2`, split
causal-mask loop. Dead ends kept in source for reference: multi-head fusion
fwd, atomic fused bwd. Tiling/pipeline probes at D=512 (extra `num_stages`,
Split-D) also lost — see below.

Further quantitative tables live in [`../context/baseline.md`](../context/baseline.md).

## ✅ Softmax `exp2` with folded log2(e) scale

**Idea (from FA2/FA3).** `tl.exp(x)` on Hopper expands to several PTX
instructions; `tl.math.exp2(x)` maps to the single `ex2.approx.ftz.f32`
instruction. Substitution requires staying in log2 domain:

1. Fold `log2(e) / sqrt(D)` into the score scale (applied after the QK^T matmul).
2. Track the running max `m_i` in log2 domain and use `tl.math.exp2` for the
   softmax rescale + normalization.
3. Convert back at store time: `lse_natural = m_i * ln(2) + ln(l_i)` so bwd
   kernels consume the same natural-log LSE as before. (Bwd kernels then
   redo the log2 conversion internally — `lse_log2 = lse * log2(e)`.)

**Results on full-causal forward, D=512, B=1, H_Q=32, H_KV=4 (small):**
modest at D=512 because the per-tile matmul dominates. Bigger wins at
smaller D and on SWA where softmax is a larger fraction of runtime.

**Microbench:** pure `tl.exp` vs `tl.math.exp2` loop: **1.13× faster** per op.

## ✅ Split causal-mask loop (off-diagonal unmasked + diagonal masked)

**Idea (from FA2/FA3).** For causal attention without SWA, only the
diagonal block `kv_b == q_b` needs the `kv_pos ≤ q_pos` check. All KV
blocks with `kv_b < q_b` are fully unmasked. The current Triton wrapper
splits the KV loop into two phases:

- Phase 1 (off-diagonal): `range(0, kv_end_unmasked, BLOCK_KV)` with no
  `tl.where` and no mask in `tl.load`. Even K/V loads skip the mask arg.
- Phase 2 (diagonal + seq-end): runs the mask path.

**Results at D=128, H_Q=32, H_KV=8, full causal FP16, H100:**

| N | Before (ms) | After (ms) | Speedup |
|---|-------------|-----------|---------|
| 4,096 | 0.69 | 0.48 | **1.45×** |
| 16,384 | 8.08 | 5.57 | **1.45×** |
| 32,768 | 31.34 | 20.89 | **1.50×** |

This takes Triton from 0.88× of SDPA to **1.31× of SDPA** at D=128 — crossing
above the cuDNN/FA3 baseline on FA's home turf.

**Compile-time guard at D≥512:** the two-loop code bloats register usage
enough that on D=512 the gain from skipping mask ops is lost to spills
(matmul is already the bottleneck). The wrapper sets
`USE_SPLIT: tl.constexpr = (HEAD_DIM < 512)`, so D=512 uses a single-loop
body while D<512 uses the split.

## MFU at production shapes (H100, bf16, N=8K)

Measured forward attention MFU (peak = 989 TFLOPS).
Reproducer: `benchmarks/mfu_sweep.py`. Per-shape breakdown and config table
in [`architecture.md`](architecture.md#target-model-attention-shapes).

| Path (per layer) | Triton MFU | SDPA MFU | speedup |
|---|---:|---:|---:|
| D=256 sliding (E2B/E4B/MoE bulk) | 24-32% | 0.7-1.4% | 23-38× |
| D=256 full causal (Gemma-3 ref)   | 47%    | 27%      | 1.8×   |
| **D=512 full causal (E2B/E4B/MoE)** | **17-19%** | 10-11%   | 1.7×  |

The D=512 full path is the lowest MFU and where remaining headroom lives.
Per-call cost is ≈24× a sliding layer at N=8K, so even though only 5-7 of
30-42 layers run on this path, it dominates per-step attention time.

## Remaining gap to FA2/FA3

FA2/FA3 still roughly 1.5–2× ahead on D=128 (~650 TFLOPS/s vs our
421 TFLOPS/s). Closing this gap needs CUDA-level tools Triton exposes only
partially: explicit warp specialization (producer/consumer warps), async
TMA loads, cluster barriers, software-pipelined mbarrier phases. The
upstream `flash-attention/flash_attn/cute/` uses those primitives via
CuTeDSL. Same story on D=512 where the gap is smaller (FA2 doesn't even
support D=512 natively; FA3 tops out at D=256) but the architectural ceiling
for Triton holds.

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

## ❌ Pipeline stages at D=512 forward (`num_stages ≥ 3`)

**Idea.** Current baseline @ D=512 is pinned to `num_stages=2` by the 232 KB
SMEM budget: `Q (64 KB) + 2 × (K 32 KB + V 32 KB) = 192 KB`. Going to
`num_stages=3` needs another 64 KB → 256 KB, exceeds the budget. Workaround:
shrink `BLOCK_KV` so a third stage fits.

**Result (H_Q=32, H_KV=4, D=512, causal, bf16, H100):**

| config                     | N=4K ms  | N=8K ms  | SMEM    | regs/spills |
|----------------------------|----------|----------|---------|-------------|
| **baseline BKV=32 s=2**    | **7.41** | **27.81**| 192 KB  | 255 / 4     |
| BKV=16 s=2                 | 11.06    | 41.77    | 128 KB  | 255 / 12    |
| BKV=16 s=3                 | 10.42    | 39.37    | 160 KB  | 255 / 14    |
| BKV=16 s=4                 | 10.44    | 39.50    | 192 KB  | 255 / 14    |
| BQ=32 BKV=32 s=3           | 10.26    | 39.69    | 162 KB  | 255 / 2     |
| BQ=32 BKV=64 s=2           | 9.12     | 39.71    | 164 KB  | 255 / 4     |
| BKV=32 s=3 / BKV=64 s=2    | OOS      | OOS      | —       | —           |

Shrinking `BLOCK_KV` to 16 to pay for a third stage costs more than the
pipeline wins: `tl.dot` on `[BQ=64, BKV=16]` output gives the H100 tensor
core a tile too small to hide its own issue latency. Every config lost to
baseline.

Reproducer: `benchmarks/split_d_probe.py`.

## ❌ Split-D forward at D=512

**Idea (from the FA2/FA3 lineage).** Tile `HEAD_DIM` into chunks of size
`BLOCK_D` (e.g., 128). For each KV tile, accumulate `scores = ΣchunksQ_d @
K_d^T` over a `tl.static_range(0, D, BLOCK_D)` inner loop, then update
`acc[:, d_chunk]` per chunk. The advertised win is that per-stage K/V SMEM
buffers shrink to `BKV × BLOCK_D × 2` bytes, unlocking `BLOCK_KV=128
num_stages=2` (192 KB est.) or `BLOCK_KV=64 num_stages=3` (160 KB est.) —
4× the current KV tile with better tensor-core utilization.

**Result: the SMEM win does not materialize in Triton 3.x.**

Tested both "Split-D on QK^T only, full-D V matmul" (Approach A) and "Split-D
on both matmuls" (Approach B). A compiled correctly and produced bit-exact
output vs baseline, but any `BLOCK_KV ≥ 64` config compiled to 2× the
expected SMEM usage. Example: `BQ=64, BKV=64, BD=128, s=2` theoretical SMEM
is `Q 64 KB + 2×(K-chunk 16 KB + V-full 64 KB) = 224 KB`, but Triton
requested 320 KB. At `BKV=32, BD=128, s=2` the kernel ran in 6.88 ms vs
baseline 6.81 ms — no regression, no win.

**Root cause.** Triton's software pipeliner treats `tl.static_range` as a
fully-unrolled block inside each pipeline stage, so it stages *all* D-chunk
K loads simultaneously rather than serializing them. The per-stage SMEM
ends up ≈ the sum over all D-chunks, defeating the entire SMEM reduction
that Split-D is designed to provide. CUTLASS/CuTeDSL can express a true
inner D-loop with independent stages; Triton 3.x cannot.

Approach B (per-chunk `acc[:, d]` update) also hits a Triton language wall:
sliced assignment on a 2D register tile (`acc[:, d0:d1] = ...`) is not
expressible, and the arithmetic-masked-scatter workaround compiles to the
same unrolled block the pipeliner chokes on.

Reproducers: `benchmarks/split_d_probe.py` (tiling sweep),
`benchmarks/split_d_proto.py` (Split-D kernel).

## Takeaway

Under Triton's abstraction, the split-dQ + pack-GQA-dKV design with
L2-level K/V reuse appears to be a local optimum for GQA + SWA shapes on
Hopper. Further improvement would need CUDA-level tools (warp
specialisation, async TMA, cluster barriers) that Triton 3.x exposes only
partially. The upstream `flash-attention/flash_attn/cute/` stack (CuTeDSL)
can go further specifically because it has those primitives.
