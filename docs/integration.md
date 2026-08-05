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
   apply a redundant mask, not an incompatible one) — **except** for
   Gemma-4-26B-A4B multimodal training, where the upstream OR-mask
   (image-bidirectional inside an image span) is plumbed in through a
   ContextVar instead. See [Multimodal OR-mask routing](#multimodal-or-mask-routing).

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
    patch_gemma4_image_group_ids_for_kernel,  # only needed for 26B-A4B multimodal
)
from transformers import AutoModelForCausalLM

patch_transformers_5_5_4_flash_attn_key()
register_triton_attention()
patch_gemma4_image_group_ids_for_kernel()       # safe no-op on E2B/E4B

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B", dtype="bfloat16", device_map="cuda")
model.config._attn_implementation = "triton_gqa"
if hasattr(model.config, "text_config"):
    model.config.text_config._attn_implementation = "triton_gqa"

out = model(input_ids)          # every layer now uses Triton
```

## Multimodal OR-mask routing

Gemma-4-26B-A4B (and only that variant — `text_config.use_bidirectional_attention == "vision"`) sets an `or_mask_function` on every sliding layer's mask kwargs:

```
mask = causal_swa | (q_group == kv_group & q_group >= 0)
```

Inside an image span, all tokens see each other bidirectionally; outside, the standard causal+SWA pattern holds. Upstream computes this by reading `mm_token_type_ids` (a `(B, N)` long tensor passed through `Gemma4Model.forward`) at mask-build time. By the time it reaches our adapter the mask is already a `BlockMask` or 4D bool — the group identity is baked in and not extractable.

Plumbing instead of mask reading:

```python
from gemma_triton_flash_attn import patch_gemma4_image_group_ids_for_kernel
patch_gemma4_image_group_ids_for_kernel()   # call once before model load
```

This wraps `Gemma4Model.forward` so that on every call:

1. If `text_config.use_bidirectional_attention == "vision"` AND `mm_token_type_ids` is in kwargs (training / prefill — incremental decode skips this), compute `group_ids`, `group_lo`, `group_hi_excl` from the token-type ids and stash them in a `ContextVar`.
2. The adapter pulls the state from the ContextVar inside the same forward and forwards it to `flash_attn_gqa_train`.
3. The kernel widens its KV/Q loop bounds to cover image spans and ORs the `(q_group == kv_group & ≥ 0)` mask into `valid` in the masked tile.
4. The wrapper resets the ContextVar on exit (even on exception) — no leakage across forwards.

The wrapper is a no-op on E2B/E4B and on text-only batches: state stays empty, adapter takes the standard causal/SWA path, kernel never compiles the group-aware branch.

**Loud failure path.** If a 4D bool mask or `BlockMask` reaches the adapter on a sliding+causal layer with no group state set, the adapter raises rather than silently dropping the OR-mask. Pre-fix, this combination produced wrong gradients in MoE multimodal training. The raise message points at `patch_gemma4_image_group_ids_for_kernel`.

Verified by `tests/test_image_group_mask.py` (kernel fwd+bwd vs eager OR-mask reference) and `tests/gemma4_integration/test_adapter_multimodal.py` (adapter + ContextVar + patch wiring + loud-failure raise).

## Verifying the adapter is actually hit

Registry selection telemetry can verify the selected implementation and config
without printing from the attention hot path or sending data anywhere:

```python
from gemma_triton_flash_attn import capture_attention_selection

with capture_attention_selection(
    "summary", labels={"model_profile": "gemma4_e2b"}
) as telemetry:
    out = model(input_ids)
    out.loss.backward()  # include this line to count backward roles

print(telemetry.format_summary())
snapshot = telemetry.snapshot()  # JSON-compatible dict
assert snapshot["total_fallbacks"] == 0
```

`summary` aggregates calls by role, config, hardware, and attention semantics.
`debug` additionally retains the full immutable spec/runtime plus accepted and
rejected registry candidates once per distinct selection. Debug mode is meant
for short diagnosis runs because dynamic shapes can create more distinct
records.

Telemetry is opt-in and process-local. Under FSDP, every rank owns an
independent recorder; callers should either compare the per-rank snapshots or
write them to rank-qualified paths. The varlen adapter explicitly records its
route to batched Triton when packing metadata is missing. A model configured to
use SDPA never enters this adapter, so release tests must also assert the
expected total attention-call count.

`test_gemma4.py` currently wraps the registered function with a call counter
and asserts it is invoked exactly once per attention layer per forward. On
Gemma-4-E2B (35 layers: 7 full + 28 sliding), a single forward yields 35 adapter
calls; telemetry is the durable replacement for that ad-hoc counter.
