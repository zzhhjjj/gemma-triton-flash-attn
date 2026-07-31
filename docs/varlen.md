# Variable-length (packed) sequences

`flash_attn_gqa_varlen` is a Triton kernel for attention over a packed batch
of variable-length sequences — the FA2-canonical "cu_seqlens" trick. Instead
of padding every sample to `max_seqlen` and wasting compute on padding tokens,
samples are concatenated into a single packed tensor and the kernel uses
offset tables to keep attention block-diagonal (no cross-sample leakage).

## When to use varlen

Use varlen when your training / prefill workload has **mixed sequence lengths
within a batch**. Typical cases:

- Chat / instruction data — responses vary from 64 to 4096 tokens.
- Code completion — function bodies range from ~100 to ~8k tokens.
- Packed pretrain — document packing produces tightly-packed 32k-token streams
  with synthetic boundaries.

For fixed-length batches (e.g. uniform 2k-token chunks), varlen and batched
produce identical work; use whichever fits your data pipeline.

Expected perf win scales with padding waste:

| Padding waste | Expected varlen speedup (D=128, H200) |
|--------------:|---------------------------------------|
| 0% (equal)    | 1.0×                                  |
| 25%           | ~1.3×                                 |
| 50%           | ~2×                                   |
| 75%           | ~4×                                   |
| 85%+          | ~10× (measured)                       |

## API

```python
def flash_attn_gqa_varlen(
    q,                    # (total_q, N_Q_HEADS, D)       fp16 or bf16, CUDA
    k,                    # (total_k, N_KV_HEADS, D)      same dtype as q
    v,                    # (total_k, N_KV_HEADS, D)      same dtype as q
    cu_seqlens_q,         # (B+1,) int32  cumulative q offsets
    cu_seqlens_k,         # (B+1,) int32  cumulative k/v offsets
    max_seqlen_q,         # int — max over q sample lengths
    max_seqlen_k,         # int — max over k sample lengths
    causal=False,         # bool
    window_size=0,        # 0 = no SWA; >0 = left window (like slide_size)
) -> Tensor:              # (total_q, N_Q_HEADS, D)
```

v1 assumes `cu_seqlens_q == cu_seqlens_k` (same packing for Q and K/V) —
the standard training case. Distinct Q/K packing is a v2 extension.

## cu_seqlens semantics

`cu_seqlens[b]` is the starting token index of sample `b` in the packed
stream. `cu_seqlens[b+1] - cu_seqlens[b]` is the length of sample `b`.

Example: three samples of lengths 512, 1024, 256:

```
cu_seqlens = [0, 512, 1536, 1792]
                ^--sample 0: [0, 512)
                     ^--sample 1: [512, 1536)
                           ^--sample 2: [1536, 1792)
```

Samples pack contiguously along the token axis of `q`, `k`, `v`.

## GQA

Pack-GQA design preserved from the batched kernel: the dK/dV backward runs
one program per KV block, unrolls all `GQA_RATIO` Q heads internally, and
writes a single accumulator per KV tile — no expand buffer, no atomic when
`Q_SPLITS=1`, no reduce kernel.

## SWA interaction

`window_size > 0` applies a **per-sample** left window. Each token in sample
`b` attends only to the preceding `window_size` tokens within sample `b`; no
cross-sample attention. If `window_size >= max_seqlen_k`, the kernel
normalizes to full causal internally (same fast path).

## LSE layout

Forward writes log-sum-exp in packed layout `(total_q, H_Q)` fp32, indexed
by absolute token index. dQ and dKV read the same layout — single source
of truth. Packed beats `(B, H_Q, max_seqlen_q)` on skewed distributions
(no padding waste for LSE).

## Atomic safety across samples

When `Q_SPLITS > 1`, dKV programs write via `atomic_add` to a pre-zeroed
packed dK/dV buffer. Because `kv_global = cu_seqlens_k[b] + kv_local_idx`
is strictly monotone across samples (cu_seqlens is monotone), programs
from different samples **never** write to the same packed row. The only
atomic contention is within a single sample's (kv_h, Q_SPLITS) cohort —
identical to the batched kernel.

## Test commands (on H200, varlen-fa conda env)

```bash
# Correctness: fwd + bwd vs per-sample SDPA, plus equal-length equivalence vs batched
python tests/test_varlen_correctness.py

# Edge cases (single-token, skewed [1,1,1,N], etc.)
python tests/test_varlen_edge_cases.py

# Oracle against upstream flash-attn (skips if not importable — see test docstring)
python tests/test_varlen_vs_flash_attn.py
```

Expected on a clean env: all three scripts exit 0.

## Benchmark command

```bash
# Quick smoke (3 configs)
python benchmarks/bench_varlen.py --quick

# Full sweep: (D, GQA, total_tokens) combos, JSON output
python benchmarks/bench_varlen.py --out benchmarks/varlen_bench.json
```

Sample output (H200, Triton 3.2, Zipf-distributed lengths):

```
 D  H_Q:H_KV   B   total   maxN  pad%  varlen ms  padded ms   speedup
128     8:1    8    4096   2126   76%      0.209      0.657     3.14×
128     8:1   16   16384   7728   87%      1.070     12.006    11.22×
128     8:1   32   32768  14489   93%      3.252     80.529    24.76×
128    32:4   16   16384   7728   87%      3.727     46.673    12.52×
256     8:2    8    8192   4252   76%      0.563      2.838     5.04×
512    32:4    4    4096   2451   58%      2.329      7.616     3.27×
```

## v1 limitations

- **Text-only.** No image-group OR-mask (Gemma-4 multimodal vision-bidirectional
  path stays in the batched kernel only).
- **Same Q/K packing.** `cu_seqlens_q == cu_seqlens_k` required. Cross-attention
  with different packings is a v2 extension.
- **Hooks into HF adapter are intentionally absent.** HF's
  `ALL_ATTENTION_FUNCTIONS` interface doesn't pass `cu_seqlens` through, so
  varlen is exposed as a standalone training API. Callers that need varlen
  inside HF models should build Q/K/V at the model-forward boundary and call
  `flash_attn_gqa_varlen` directly.
- **H200 + Triton 3.2 tuning.** v1 uses a conservative block-size table that
  fits Triton 3.2's 228 KB shmem cap on H200 (stricter accounting than Triton
  3.0-3.1):
    - D=128: `BQ=128, BKV=64, w=4, s=2`
    - D=256: `BQ=64,  BKV=64, w=4, s=2`   (was BQ=128 on older Triton)
    - D=512: `BQ=32,  BKV=32, w=4, s=2`   (was BQ=64 on older Triton)
  Expected ~10–20% slower than the H100 peak on D=256/D=512; a full sweep
  (especially larger BKV at D=128 given H200's 4.8 TB/s HBM) is a follow-up.

## Source

- Kernels: `flash_attn/attention.py`
  - `_flash_attn_gqa_varlen_fwd_kernel`
  - `_flash_attn_gqa_varlen_bwd_dq_kernel`     (keeps `STORE_DELTA=True` fusion)
  - `_flash_attn_gqa_varlen_bwd_dkv_packed_kernel`  (pack-GQA + Q_SPLITS)
- Autograd: `FlashAttnGQAVarlenFunction`
- Reference: `attention_gqa_varlen_ref` (per-sample SDPA loop)
- Pack helpers: `pack_batched_to_varlen`, `unpack_varlen_to_batched`
