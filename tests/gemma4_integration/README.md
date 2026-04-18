# Gemma-4 Integration Tests

End-to-end tests swapping the attention implementation in real Gemma-4 models
with our Triton kernel:

- **`google/gemma-4-E2B`** (5.1B dense, 35 layers) — `test_gemma4.py`
- **`google/gemma-4-26B-A4B`** (26B MoE, 30 layers, 128 experts, top-k=8) —
  `test_gemma4_moe.py`. Requires multi-GPU (shards via `device_map="auto"`).

## Setup

```bash
# 1. Install deps into a clean env
python -m venv /opt/tiger/flash_gemma           # or uv venv
source /opt/tiger/flash_gemma/bin/activate
uv pip install -r ../../requirements.txt
uv pip install -e ../..                          # the kernel package

# 2. Set HuggingFace token (Gemma is gated)
export HF_TOKEN="hf_..."
```

## Run

```bash
# Adapter unit test — no model download, covers SWA + full-causal + GQA ratios
python test_adapter.py

# Real Gemma-4-E2B — correctness only (fast)
python test_gemma4.py --skip-perf

# Real Gemma-4-E2B — correctness + throughput
python test_gemma4.py --seq-len 1024

# Real Gemma-4-26B-A4B MoE — shards across all visible GPUs
python test_gemma4_moe.py --seq-len 1024

# Memory benchmark — SDPA vs Triton peak memory, max context length
python test_memory.py
```

## Three tests, three purposes

- **`test_adapter.py`** — Unit-tests the HF adapter itself against an SDPA
  reference. 24 cases covering: GQA ratios {1:1, 2:1, 4:1, 8:1}, SWA with both
  N≤slide (window covers full seq) and N>slide (real window truncation),
  HEAD_DIM ∈ {256, 512}, batch sizes {1, 2}. Guarantees both **sliding** and
  **full-causal** code paths produce correct numerics. Runs in seconds, no HF
  download required.

- **`test_gemma4.py`** — Full end-to-end test: loads `google/gemma-4-E2B`
  (5.1B params, 35 layers: 7 full + 28 sliding), swaps attention to the Triton
  kernel via the registry, compares logits vs SDPA, benchmarks throughput.
  Confirms the integration works on a real production-sized model.

- **`test_gemma4_moe.py`** — MoE version: loads `google/gemma-4-26B-A4B`
  (26B params, 30 layers: 24 sliding + 6 full, 128 experts, top-k=8), shards
  across all visible GPUs via `device_map="auto"`, and compares last-token
  logits vs SDPA. Full-sequence logits (1×N×262K) won't fit next to the 52 GB
  weights, so only the last-token row is compared. Requires the multi-GPU
  device-context fix in `hf_integration.py` — Triton otherwise launches every
  layer's kernel on cuda:0 regardless of tensor device, silently producing NaN.

- **`test_memory.py`** — Peak-memory benchmark during forward pass at
  increasing seq lengths. Quantifies (a) the point where SDPA starts
  materialising attention scratch (N ≈ 16K), and (b) the max N each
  implementation can run on the hardware. On 80 GB H100, SDPA OOMs at N=32K
  while Triton still runs — **2× longer context on the same hardware**.

## How it works

transformers 5.5.4 exposes a pluggable `ALL_ATTENTION_FUNCTIONS` registry.
The test registers a `triton_gqa` entry pointing to our Triton kernel, then
sets `model.config._attn_implementation = "triton_gqa"`. Every attention
layer in the model routes through our kernel.

```python
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from gemma_triton_flash_attn import flash_attn_gqa_train

def triton_gqa_attention(module, query, key, value, attention_mask,
                        scaling=None, sliding_window=None, **kwargs):
    # Bake scaling into q, call kernel, transpose output
    ...

ALL_ATTENTION_FUNCTIONS["triton_gqa"] = triton_gqa_attention
model.config._attn_implementation = "triton_gqa"
```

## Known workaround

transformers 5.5.4 has a bug where `PACKAGE_DISTRIBUTION_MAPPING["flash_attn"]`
raises KeyError during any model config load, even with `attn_implementation="sdpa"`.
The test patches this at import time:

```python
import transformers
from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
```

## Results (H100, Gemma-4-E2B bf16, forward-only)

| seq_len | SDPA (ms) | Triton (ms) | Speedup |
|---------|-----------|-------------|---------|
| 512     | 42.70     | 43.09       | 0.99x   |
| 1024    | 43.68     | 43.45       | 1.01x   |
| 2048    | 64.09     | 48.27       | 1.33x   |
| 4096    | 159.85    | 81.80       | **1.95x** |
| 8192    | 483.71    | 168.86      | **2.86x** |
| 16384   | 1639.18   | 366.50      | **4.47x** |
| 32768   | **OOM**   | 902.65      | (SDPA OOMs) |

Correctness (vs SDPA, N=1024):
- Max logits abs diff: 2.36e+00 (on logits of magnitude ~1e2 × vocab_size=262144)
- Rel Frobenius diff:  2.28e-02
- Cosine similarity:   **0.999758**
- Top-1 token match (last pos): **100%**
- Top-5 overlap (last pos): **5/5**

At short N, the attention layers are a small fraction of end-to-end latency
(linear projections and RMSNorms dominate). Speedup grows with N as attention
becomes O(N²) while projections stay O(N).

## Files

```
test_gemma4.py      # main test script
pyproject.toml      # uv workspace member
README.md           # this file
```
