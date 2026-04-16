# Testing

## Adapter unit test — `tests/gemma4_integration/test_adapter.py`

Exercises the registered `triton_gqa` adapter directly against an SDPA
reference. No model download, runs in seconds.

**Coverage: 24 parameterised cases, all PASS with cos sim > 0.999987.**

| Dimension | Values |
|-----------|--------|
| GQA ratio | 1:1 (MHA), 2:1, 4:1, 8:1 |
| Pattern   | full causal + sliding window (slide=512) |
| SWA regime | N ≤ slide (window covers full seq) + N > slide (real truncation) |
| HEAD_DIM  | 256 (Gemma4 sliding) + 512 (Gemma4 full) |
| Batch     | 1 + 2 |

This guarantees both sliding and full-causal code paths produce correct
numerics for every supported shape — no accidental regressions from one
path hiding behind the other.

Run:

```bash
python tests/gemma4_integration/test_adapter.py
```

## Real model E2E test — `tests/gemma4_integration/test_gemma4.py`

Loads `google/gemma-4-E2B` (5.1B params, 35 layers: 7 full + 28 sliding),
swaps attention to Triton via registry, compares logits vs SDPA.

**Correctness at N=1024:**
- cos similarity: **0.999745**
- top-1 token match (last position): **100 %**
- top-5 token overlap (last position): **5 / 5**

**Call instrumentation:** the adapter is hit exactly 35× per forward
(7 full + 28 sliding). Confirms every layer routes through Triton — SWA is
not silently falling back.

Run (Gemma is gated on HF, need a token):

```bash
export HF_TOKEN="hf_..."
python tests/gemma4_integration/test_gemma4.py --seq-len 1024
```

## Memory benchmark — `tests/gemma4_integration/test_memory.py`

Measures peak GPU memory during Gemma-4-E2B forward, SDPA vs Triton, across
N. Demonstrates that Triton runs where SDPA OOMs.

Run:

```bash
python tests/gemma4_integration/test_memory.py
```

## Combined benchmark — `benchmarks/run_final_benchmark.py`

Re-runs everything (kernel-fwd, kernel-fwd+bwd, E2E fwd, peak memory) and
generates the plots shown in the main README.

Run:

```bash
export HF_TOKEN="hf_..."
python benchmarks/run_final_benchmark.py
```

Outputs `benchmarks/{results.json, flops_vs_sdpa.png, e2e_latency_vs_sdpa.png, memory_vs_sdpa.png}`.

To regenerate just the plots from cached data:

```bash
python benchmarks/replot.py
```

## Optimization correctness tests

- `tests/test_packed_dkv.py` — pack-GQA dKV correctness + micro-benchmark
  (the successful optimisation). Verifies max-abs diff < 1e-3 vs the
  split-and-reduce baseline.
- `tests/test_grouped_forward.py` — multi-head fusion (dead end) repro.
- `tests/test_fused_backward.py` — atomic fused bwd (dead end) repro.
