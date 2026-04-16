# gemma-triton-flash-attn

Drop-in Triton Flash Attention for HuggingFace transformers. One function call
replaces the attention kernel in every layer of your model — no subclassing,
no model surgery.

Optimised for **Gemma4-style models** (GQA with alternating **full causal**
`HEAD_DIM=512` and **sliding window** `HEAD_DIM=256` layers), where SDPA's
cuDNN / FlashAttention-3 paths either miss the config or lack SWA support.

## Quickstart: swap attention in 3 lines

```python
from gemma_triton_flash_attn import register_triton_attention
from transformers import AutoModelForCausalLM

register_triton_attention()                                   # 1. register "triton_gqa"
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B", dtype="bfloat16", device_map="cuda")
model.config._attn_implementation = "triton_gqa"              # 2. opt in
if hasattr(model.config, "text_config"):                      # 3. also opt in for nested configs
    model.config.text_config._attn_implementation = "triton_gqa"

# Every attention layer now uses the Triton kernel. Forward / backward / generate
# all continue to work — the rest of the transformers stack is untouched.
out = model(input_ids)
```

### How it works

transformers ≥ 5.5 exposes a pluggable dict, `ALL_ATTENTION_FUNCTIONS`, keyed by
implementation name (`"sdpa"`, `"eager"`, `"flash_attention_2"`, …). Every
attention layer looks up its kernel by
`ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`. We register one extra
entry, `"triton_gqa"`, pointing at an adapter that:

1. Reconciles the `scaling` argument (Gemma4 passes 1.0 with scaling folded into
   `q_norm`; most other models pass `1/√d`). The adapter pre-multiplies `q` so
   the kernel's internal `1/√d` cancels out to the requested scale.
2. Maps `sliding_window` → kernel's `slide_size` argument (0 = full causal).
3. Ignores HuggingFace's additive `attention_mask` — the kernel builds its own
   causal + sliding-window mask internally and covers the same cases.
4. Transposes the output from our `(B, H, N, D)` convention to transformers'
   `(B, N, H, D)` so the downstream `.reshape(B, N, H*D)` works.
5. Fails loudly on unsupported features (`softcap`, nonzero `dropout`) rather
   than silently producing wrong numerics.

The full adapter is ~40 lines: see
[`flash_attn/hf_integration.py`](flash_attn/hf_integration.py).

## Why this package

PyTorch SDPA (cuDNN / FlashAttention-3) is heavily optimized for standard
`HEAD_DIM` values (64, 128, 256). Gemma4's two attention variants fall outside
the fast path:

| Config | HEAD_DIM | H_Q / H_KV | GQA ratio | SDPA status |
|--------|----------|------------|-----------|-------------|
| Full causal (Gemma4 global) | **512** | 32 / 4 | 8:1 | generic fallback, slow |
| Sliding (Gemma4 local) | 256 | 32 / 16 | 2:1 | fast at short N, **no SWA support** |

For models that alternate these (Gemma4, MoE hybrids), this kernel is typically
1.3×–4.5× faster end-to-end on H100.

## Tests

### Adapter unit test — `tests/gemma4_integration/test_adapter.py`

Exercises the registered `triton_gqa` adapter directly against an SDPA reference.
No model download, runs in seconds.

**Coverage (24 parameterised cases, all PASS cos sim > 0.999987)**:

| Dimension | Values |
|-----------|--------|
| **GQA ratio** | 1:1 (MHA), 2:1, 4:1, 8:1 |
| **Pattern**   | full causal + sliding window (slide=512) |
| **SWA regime** | N ≤ slide (window covers full seq) + N > slide (real truncation) |
| **HEAD_DIM**  | 256 (Gemma4 sliding) + 512 (Gemma4 full) |
| **Batch**     | 1 + 2 |

This guarantees both **sliding** and **full-causal** code paths produce correct
numerics for every supported shape — no accidental regressions from one path
hiding behind the other.

### Real model E2E test — `tests/gemma4_integration/test_gemma4.py`

Loads `google/gemma-4-E2B` (5.1B params, 35 layers: **7 full + 28 sliding**),
swaps attention to Triton via registry, compares logits vs SDPA.

**Correctness** (N=1024 vs SDPA):
- cos similarity: **0.999745**
- top-1 token match (last pos): **100 %**
- top-5 token overlap (last pos): **5 / 5**

**Call instrumentation**: adapter hit exactly 35× per forward (7 full + 28 sliding).
Confirms every layer routes through Triton — SWA is not silently falling back.

## Performance

### Real model: Gemma-4-E2B forward pass, BF16, H100

| seq_len | SDPA (ms) | Triton (ms) | Speedup |
|---------|-----------|-------------|---------|
| 512    | 49.0    | 49.7     | 0.99× |
| 1,024  | 46.5    | 46.6     | 1.00× |
| 2,048  | 63.5    | 48.0     | **1.32×** |
| 4,096  | 158.4   | 81.0     | **1.95×** |
| 8,192  | 481.0   | 167.3    | **2.87×** |
| 16,384 | 1638.5  | 363.4    | **4.51×** |

Short N is dominated by linear projections (35 layers × 4 projs each);
attention becomes the bottleneck once N ≥ 2K, where our kernel shines.

### Kernel-level: Full causal Fwd+Bwd (D=512, H_Q=32, H_KV=4)

| N | Speedup vs SDPA |
|---|---------|
| 1,024 | 2.6× |
| 4,096 | 2.1× |
| 16,384 | 1.9× |

### Kernel-level: Sliding Fwd+Bwd (D=256, H_Q=32, H_KV=16, slide=1024)

| N | Speedup vs SDPA |
|---|---------|
| 4,096 | 1.25× |
| 8,192 | 2.0× |
| 16,384 | 3.9× |
| 32,768 | **7.7×** |
| 131,072 | **22.8×** |

Peak: **36× forward speedup at N=128K** (sliding, kernel-only, single layer).

### Memory reduction on Gemma-4-E2B (forward, BF16, 80 GB H100)

Peak GPU allocation above the model weights during one forward pass:

| seq_len | SDPA peak | Triton peak | Reduction | Saved |
|---------|-----------|-------------|-----------|-------|
| ≤ 8,192 | ~equal    | ~equal      | 1.00×     | 0     |
| 16,384  | 22.0 GB   | 16.7 GB     | **1.32×** | **5.3 GB** |
| 32,768  | **OOM**   | 33.4 GB     | —         | SDPA can't run |
| 65,536  | OOM       | OOM         | —         | beyond 80 GB |

**Key result: Triton runs at 32K where SDPA OOMs — 2× usable context length on
the same hardware.** At short N, both are tied because SDPA uses an internal
flash-attention backend. At N≥16K, SDPA starts materialising attention scratch
space while our kernel's online softmax keeps peak bounded. The H_KV=1 → H_Q=8
GQA expansion that SDPA does via `repeat_kv` is also avoided.

Reproduce: `python tests/gemma4_integration/test_memory.py`

## Installation

```bash
git clone <repo>
cd gemma-triton-flash-attn
pip install -e .
```

Requires: `torch>=2.0`, `triton>=3.0`, CUDA GPU (tested on H100).

### Running the tests

```bash
pip install -r requirements.txt                       # transformers 5.5.4, accelerate, etc.

# 1) Adapter unit test (seconds, no download)
python tests/gemma4_integration/test_adapter.py
# → 24/24 PASS, cos sim > 0.999987 on every case

# 2) Real Gemma-4-E2B end-to-end (downloads 5 GB on first run)
export HF_TOKEN="hf_..."                              # Gemma is gated on HF
python tests/gemma4_integration/test_gemma4.py --seq-len 1024
# → cos sim 0.9997 vs SDPA, 100% top-1 match, throughput 4.51× at N=16K
```

Note: transformers 5.5.4 has a bug where any model config load raises
`KeyError: 'flash_attn'`. We ship a one-line workaround:

```python
from gemma_triton_flash_attn import patch_transformers_5_5_4_flash_attn_key
patch_transformers_5_5_4_flash_attn_key()     # call once, before any config load
```

## API

### Integration (most users start here)

```python
register_triton_attention(name: str = "triton_gqa") -> None
    Register the Triton kernel in transformers' ALL_ATTENTION_FUNCTIONS under
    the given name. Safe to call multiple times.

triton_gqa_attention(module, q, k, v, attention_mask, *, scaling, sliding_window, ...)
    The adapter function itself (exported in case you want to wrap / override).

patch_transformers_5_5_4_flash_attn_key() -> None
    Fixes the transformers 5.5.4 KeyError bug. No-op on other versions.
```

### Direct kernel (when you don't want transformers in the loop)

```python
attention_flash_gqa(q, k, v, causal=False, slide_size=0) -> out
    Inference forward. No autograd. Uses less memory than training.

flash_attn_gqa_train(q, k, v, causal=False, slide_size=0) -> out
    Training forward with autograd backward support.

# Input shapes (all tensors):
#   q: (B, N_Q_HEADS,  N, D)
#   k: (B, N_KV_HEADS, N, D)   — GQA: N_Q_HEADS must be multiple of N_KV_HEADS
#   v: (B, N_KV_HEADS, N, D)
```

Supported: `dtype={fp16, bf16}`, `HEAD_DIM ∈ [64, 512]` (tested 256 and 512),
`slide_size ∈ {0, positive int}`.

## What it does NOT support

- Variable-length sequences / padding mask
- ALiBi or positional bias injection
- `softcap` (raises `NotImplementedError` in the adapter)
- Attention dropout (same)
- `HEAD_DIM` outside 64-512 range
- Devices other than CUDA

## Repository layout

```
flash_attn/
  __init__.py            # public API exports
  attention.py           # all Triton kernels + wrappers (fwd + bwd + SWA)
  hf_integration.py      # the HF attention adapter + register_triton_attention()
  gemma4_e2e.py          # hand-built Gemma4-style stack benchmark (no HF)
  utils.py               # benchmark utilities

tests/gemma4_integration/
  test_adapter.py        # adapter unit test: GQA × SWA × D — no model download
  test_gemma4.py         # real google/gemma-4-E2B E2E test (correctness + perf)
  test_memory.py         # peak memory benchmark: SDPA vs Triton, max context
  README.md              # how to run the tests
  pyproject.toml         # uv workspace member

pyproject.toml           # package config (PyPI name: gemma-triton-flash-attn)
requirements.txt         # integration test deps
```
