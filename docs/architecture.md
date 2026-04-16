# Architecture

## Repository layout

```
flash_attn/
  __init__.py            # public API exports
  attention.py           # all Triton kernels + wrappers (fwd + bwd + SWA)
  hf_integration.py      # HF attention adapter + register_triton_attention()
  gemma4_e2e.py          # hand-built Gemma4-style stack benchmark (no HF)
  utils.py               # benchmark utilities

tests/gemma4_integration/
  test_adapter.py        # adapter unit test: GQA × SWA × D — no model download
  test_gemma4.py         # real google/gemma-4-E2B E2E test (correctness + perf)
  test_memory.py         # peak memory benchmark: SDPA vs Triton, max context
  README.md              # how to run the tests
  pyproject.toml         # uv workspace member

benchmarks/
  run_final_benchmark.py # combined speed + memory benchmark
  replot.py              # regenerate plots from cached results.json
  results.json           # raw benchmark data
  *.png                  # generated plots

docs/                    # technical documentation
context/baseline.md      # full quantitative history (internal)
pyproject.toml           # package config (PyPI name: gemma-triton-flash-attn)
requirements.txt         # integration test deps
```

## Kernel structure in `flash_attn/attention.py`

| Symbol | Role | Used in wrappers |
|--------|------|------------------|
| `_flash_attn_gqa_kernel` | Forward; `STORE_LSE` flag doubles as inference and training fwd | ✅ always |
| `_flash_attn_gqa_bwd_dq_kernel` | Backward dQ (one program per Q block, iterates KV) | ✅ default bwd |
| `_flash_attn_gqa_bwd_dkv_packed_kernel` | Backward dK/dV (pack-GQA style, no atomics) | ✅ default bwd |
| `_flash_attn_gqa_bwd_dkv_kernel` | Backward dK/dV (old split + reduce) | ⚪ kept for reference |
| `_flash_attn_gqa_grouped_kernel` | Failed multi-head fusion fwd | ⚪ kept for reference |
| `_flash_attn_gqa_bwd_fused_kernel` | Failed atomic fused bwd | ⚪ kept for reference |
| `_delta_kernel` | Preprocess: computes `rowsum(dO * O)` for bwd | ✅ always |
| `FlashAttnGQAFunction` | `torch.autograd.Function` tying fwd + bwd | ✅ training |

## Wrappers

```python
attention_flash_gqa(q, k, v, causal=False, slide_size=0)          # inference fwd
flash_attn_gqa_train(q, k, v, causal=False, slide_size=0)         # training fwd (autograd)
attention_gqa_ref / attention_swa_ref                              # eager PyTorch refs
```

## Data flow for training

```
user tensors (B, H_Q|H_KV, N, D)
  → FlashAttnGQAFunction.forward
      → _delta_kernel (precompute dO·O rowsum) [only on backward path]
      → _flash_attn_gqa_kernel (STORE_LSE=True)     [forward]
  → save for backward: q, k, v, o, lse
  → loss.backward() triggers:
      → _delta_kernel                               [now run with output]
      → _flash_attn_gqa_bwd_dq_kernel               [dQ]
      → _flash_attn_gqa_bwd_dkv_packed_kernel       [dK, dV, no atomics]
  → return dq, dk, dv
```

## Design choices

**Why pack-GQA for dK/dV, not dQ?**
The GQA ratio is between Q heads and KV heads. For dQ, each Q has exactly one
owning KV block, so there's no atomic contention — a plain split works.
For dK/dV, each KV block is touched by `GQA_RATIO` Q heads, which without
packing means `GQA_RATIO` programs contending on the same tile. Pack-GQA
collapses those into one program with an internal `tl.static_range` loop.

**Why is the forward kernel shared between inference and training?**
A single `STORE_LSE: tl.constexpr` flag switches whether the LSE output is
emitted. Inference skips the HBM write (`~5%` faster); training needs it for
the bwd pass. One compilation per (dtype, D, causal, slide) pair — two would
be wasteful.

**Why autograd.Function instead of compile/torch.func?**
We want deterministic kernel selection per call (D-aware block sizes), which
doesn't play nicely with torch.compile's shape polymorphism. The Function
wrapper also lets us save exactly the tensors needed (q, k, v, o, lse) with
zero copy.

## Block sizes

Block sizes are chosen per D to fit shared memory on H100 (228 KB usable):

| D | BQ (fwd) | BKV (fwd) | BQ (bwd dQ) | BKV (bwd dKV) |
|---|----------|-----------|-------------|---------------|
| 64 / 96 / 128 | 128 | 64 | 64 | 64 |
| 256 | 128 | 64 | 64 | 64 |
| 512 | 64 | 32 | 32 | 32 |

Larger D forces smaller tiles because shared memory is ~`(BQ + 2·BKV) × D`
fp16 bytes + fp32 accumulators.

## For more detail

- [`integration.md`](integration.md) — adapter internals and registry mechanism
- [`optimization_notes.md`](optimization_notes.md) — what was tried and why
- [`api.md`](api.md) — every public function and its signature
- [`../context/baseline.md`](../context/baseline.md) — full benchmark history
