# Gemma-4-E2B Integration Test

End-to-end test swapping the attention implementation in the real
`google/gemma-4-E2B` model (5.1B params, 35 layers) with our Triton kernel.

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
| 512     | 49.04     | 49.74       | 0.99x   |
| 1024    | 46.48     | 46.61       | 1.00x   |
| 2048    | 63.51     | 48.00       | 1.32x   |
| 4096    | 158.38    | 81.03       | **1.95x** |
| 8192    | 480.99    | 167.31      | **2.87x** |
| 16384   | 1638.54   | 363.37      | **4.51x** |

Correctness (vs SDPA, N=1024):
- Max logits abs diff: 3.23e+00 (on logits of magnitude ~1e2 × vocab_size=262144)
- Rel Frobenius diff:  2.27e-02
- Cosine similarity:   **0.999745**
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
