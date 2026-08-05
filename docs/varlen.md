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

## 当前测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_varlen_numerics.py --run-gpu
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_semantic_invariants.py --run-gpu
```

这两个 pytest 文件是当前门禁，覆盖 output/dQ/dK/dV、不同长度分布、边界和语义不变量。资源不足不得转换为 required case 的成功。

`tests/test_varlen_correctness.py`、`test_varlen_edge_cases.py` 和 `test_varlen_scaling.py` 是 H200/Triton 3.2 阶段历史脚本；`test_varlen_vs_flash_attn.py` 尚未实现实际 oracle 断言，均不属于当前门禁。

## 当前 benchmark

```bash
python benchmarks/benchmark_varlen_registry.py \
  --profile gemma4_e2b_text_full \
  --lengths 2048,2048,2048,2048 \
  --phase forward_backward \
  --dtype bfloat16
```

该入口走 production public API 与 registry，先验证同语义 PyTorch reference，再保存 selection、latency 分布、MFU 和环境。旧 `bench_varlen.py` 及其 JSON 只保留为 H200 历史证据。

## H200 历史结论（保留）

以下结果来自 H200、Torch 2.6.0+cu124、Triton 3.2.0 的 `varlen-fa`
环境。它们不是当前 release gate，也不能外推到 H100/B200，但仍是 H200
实现与调优的重要证据。

历史测试命令：

```bash
python tests/test_varlen_correctness.py
python tests/test_varlen_edge_cases.py
python tests/test_varlen_vs_flash_attn.py
```

`test_varlen_correctness.py` 覆盖 forward/backward 对 per-sample SDPA，
以及 equal-length packed 与 batched kernel 的等价性；后者要求 fp32 cosine
>0.99999，是当时最紧的诊断。需要保留两个已知限制：

- D256/D512 遇到 Triton shared-memory OutOfResources 时，旧脚本会记为 skip；
- `test_varlen_vs_flash_attn.py` 的 upstream oracle 比较没有真正实现，
  找到或找不到 upstream 模块都会返回 0。

历史 benchmark 命令：

```bash
python benchmarks/bench_varlen.py --quick
python benchmarks/bench_varlen.py --out benchmarks/varlen_bench.json
```

H200、Triton 3.2、Zipf 长度分布的保存结果：

```
 D  H_Q:H_KV   B   total   maxN  pad%  varlen ms  padded ms   speedup
128     8:1    8    4096   2126   76%      0.209      0.657     3.14×
128     8:1   16   16384   7728   87%      1.070     12.006    11.22×
128     8:1   32   32768  14489   93%      3.252     80.529    24.76×
128    32:4   16   16384   7728   87%      3.727     46.673    12.52×
256     8:2    8    8192   4252   76%      0.563      2.838     5.04×
512    32:4    4    4096   2451   58%      2.329      7.616     3.27×
```

当时的结论是：varlen 收益随 padding waste 增长，D128 在 85%+ padding
waste 下可达到约 10×，保存的极端 cell 达到 24.76×。

H200/Triton 3.2 为适配 228 KB shared-memory 上限采用的保守表：

- D128：`BQ=128, BKV=64, w=4, s=2`；
- D256：`BQ=64, BKV=64, w=4, s=2`，旧 Triton 曾使用 BQ128；
- D512：`BQ=32, BKV=32, w=4, s=2`，旧 Triton 曾使用 BQ64。

当时估计 D256/D512 比 H100 peak 慢约 10–20%，并记录了后续需要在
H200 4.8 TB/s HBM 上重扫 D128 BKV。当前 registry 的 `sm90` base 是
compile-safe 口径，不表示这些历史 tuning 已完成新软件栈复认证。

## 支持范围

- v1 要求 `cu_seqlens_q == cu_seqlens_k`；
- kernel 本身不支持 image-group OR-mask；
- HF integration 已提供 `triton_gqa_varlen_attention`、注册函数和 Ulysses varlen adapter，不再是“完全没有 HF hook”；
- `sm90` 保留 H100/H200 compile-safe base；H200 尚无独立 tuned override；
- `sm100` 保留安全 base，B200 varlen D512 dKV 使用已验证 product override；
- 各硬件性能结论必须在对应实机重新采集，不能复用旧 H200 或 B200 数字。

## Source

- Kernels: `flash_attn/attention.py`
  - `_flash_attn_gqa_varlen_fwd_kernel`
  - `_flash_attn_gqa_varlen_bwd_dq_kernel`     (keeps `STORE_DELTA=True` fusion)
  - `_flash_attn_gqa_varlen_bwd_dkv_packed_kernel`  (pack-GQA + Q_SPLITS)
- Autograd: `FlashAttnGQAVarlenFunction`
- Reference: `attention_gqa_varlen_ref` (per-sample SDPA loop)
- Pack helpers: `pack_batched_to_varlen`, `unpack_varlen_to_batched`
