# Transformers integration

This document explains how `gemma-triton-flash-attn` plugs into HuggingFace
transformers, what the adapter does, and the one-time patch needed for
transformers 5.5.4.

## Registry mechanism

transformers ≥ 5.5 exposes a pluggable dict, `ALL_ATTENTION_FUNCTIONS`, keyed
by implementation name (`"sdpa"`, `"eager"`, `"flash_attention_2"`, …). Every
attention layer looks up its kernel by
`ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`. We register one extra
entry, `"triton_gqa"`, pointing at the adapter.

```python
from gemma_triton_flash_attn import register_triton_attention
register_triton_attention()                # default name: "triton_gqa"
register_triton_attention(name="my_attn")  # or pick your own
```

## What the adapter does

The adapter (`triton_gqa_attention` in
[`../flash_attn/hf_integration.py`](../flash_attn/hf_integration.py)) is ~40
lines and has five responsibilities:

1. **Scaling reconciliation.** Gemma4 passes `scaling=1.0` with `1/√d` folded
   into `q_norm`; most other models pass `1/√d`. The adapter pre-multiplies
   `q` so the kernel's internal `1/√d` cancels out to the requested scale.

2. **Sliding-window mapping.** `sliding_window=None|0` → full causal;
   `sliding_window=S` → SWA with window size `S`.

3. **Mask handling.** The kernel builds its own causal + sliding-window mask
   internally. HuggingFace's additive `attention_mask` is ignored (it would
   apply a redundant mask, not an incompatible one).

4. **Shape transpose.** Kernel uses `(B, H, N, D)`; transformers expects
   `(B, N, H, D)` so the downstream `.reshape(B, N, H*D)` works.

5. **Loud failure on unsupported features.** `softcap ≠ 0` or non-zero
   `dropout` raise `NotImplementedError` immediately rather than silently
   producing wrong numerics.

## transformers 5.5.4 KeyError workaround

Loading any model config on transformers 5.5.4 raises:

```
KeyError: 'flash_attn'
```

This is a bug in `transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING`.
We ship a one-line patch:

```python
from gemma_triton_flash_attn import patch_transformers_5_5_4_flash_attn_key
patch_transformers_5_5_4_flash_attn_key()   # call once, before any config load
```

It's a no-op on other transformers versions.

## Full integration example

```python
from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
)
from transformers import AutoModelForCausalLM

patch_transformers_5_5_4_flash_attn_key()
register_triton_attention()

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B", dtype="bfloat16", device_map="cuda")
model.config._attn_implementation = "triton_gqa"
if hasattr(model.config, "text_config"):
    model.config.text_config._attn_implementation = "triton_gqa"

out = model(input_ids)          # every layer now uses Triton
```

## Verifying the adapter is actually hit

`test_gemma4.py` wraps the registered function with a call counter and asserts
it's invoked exactly once per attention layer per forward. On Gemma-4-E2B (35
layers: 7 full + 28 sliding), a single forward yields 35 adapter calls — if
SWA or some other code path were silently falling back to SDPA, the count
would drop.
