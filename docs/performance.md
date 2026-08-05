# Performance measurement and regression gates

Performance claims in this repository must come from the registry-backed
public API and carry correctness, selection, hardware, software, source, and
measurement provenance in the same result record.

## Canonical benchmark

Run one model profile explicitly; GPU architecture and product are detected
from the CUDA device:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_registry.py \
  --profile gemma4_e2b_text_full \
  --seq-len 1024 4096 \
  --phase forward_backward \
  --dtype bfloat16 \
  --warmup 5 --repetitions 20
```

Each cell:

1. constructs the exact profile semantics;
2. compares output and, for `forward_backward`, dQ/dK/dV against the matching
   PyTorch reference with FP32 metrics and cosine floor `0.9999`;
3. captures debug registry telemetry for the production public API;
4. records every CUDA-event latency sample plus min/p20/median/p80/max;
5. reports semantic TFLOP/s and attention-kernel MFU;
6. writes a timestamp/GPU/SM/commit-qualified result directory with exclusive
   file creation, so an earlier record cannot be replaced.

Sliding profiles use the explicit sliding-window reference, not full-causal
SDPA. The eager reference constructs/expands masks and KV heads as needed and
is therefore a semantic baseline, not necessarily an optimized competitor.

## FLOPs and MFU convention

One multiply-add is two FLOPs. For every attended query/key pair and Q head:

- forward counts QKᵀ and PV: `4 × head_dim` FLOPs;
- forward+backward also counts dV, dP, dQ, and dK: `12 × head_dim` total.

Softmax scalar operations and implementation-specific recomputation are
excluded. The reported value is therefore comparable *algorithmic attention
work*, not a claim about every instruction executed by a particular kernel.

```text
semantic TFLOP/s = algorithmic FLOPs / median seconds / 1e12
attention-kernel MFU = semantic TFLOP/s / dense Tensor Core peak
```

This is not model MFU. Model MFU requires a separately documented full-model
training-FLOPs convention and end-to-end step time.

## Hardware peak catalog

`flash_attn.performance.HARDWARE_PEAKS` contains product-qualified dense
FP16/BF16 Tensor Core and HBM ceilings for B200 HGX, H200 SXM/NVL, and H100
SXM/NVL/PCIe. Sparse table values are divided by two. B200 per-GPU values are
derived from the official eight-GPU HGX totals.

The performance catalog never participates in registry dispatch. An unknown or
ambiguous product fails instead of inheriting H100 numbers. For a new product,
prefer adding a sourced catalog record; a diagnostic run may supply both
`--dense-peak-tflops` and `--hbm-bandwidth-gbps`, which is recorded as an
explicit override.

## Regression comparison

Compare result files or their containing directories:

```bash
python benchmarks/compare_registry_results.py \
  <baseline-result-or-dir> <candidate-result-or-dir> \
  --max-latency-regression 0.05 \
  --output <new-exclusive-comparison.json>
```

The comparison exits nonzero for:

- different GPU, driver, CUDA, Torch, Triton, dtype/peak, warmup, or repetition
  policy;
- missing, extra, or semantically different cells;
- any candidate correctness failure;
- median latency regression beyond the declared threshold;
- implementation/config/tile drift without a new distinct `verified` evidence
  record in debug telemetry.

It reports median change and relative p20–p80 span for both runs. A passing P2
comparison is required before a tuned candidate replaces an architecture base
or product override in the registry.

## Current limitations

- v1 benchmarks direct batched model profiles; image-group metadata, varlen,
  real-model execution, and distributed runs retain separate gates.
- the benchmark records HBM peak but does not yet claim an achieved-bandwidth
  number because exact FlashAttention byte traffic depends on tiling and cache
  reuse;
- compile time, peak memory, NCU/NSYS captures, real-model MFU, and FSDP2
  parity are later gates, not inferred from this benchmark.
- very short smoke shapes validate wiring but are not performance evidence.
