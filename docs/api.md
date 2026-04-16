# API reference

All exports live in `gemma_triton_flash_attn` (package name: `gemma-triton-flash-attn`).

## HF integration (most users start here)

### `register_triton_attention(name: str = "triton_gqa") -> None`

Register the Triton kernel in transformers' `ALL_ATTENTION_FUNCTIONS` under
the given name. Safe to call multiple times (idempotent).

```python
from gemma_triton_flash_attn import register_triton_attention
register_triton_attention()                 # "triton_gqa" (default)
register_triton_attention(name="my_attn")   # or pick your own key
```

### `triton_gqa_attention(module, q, k, v, attention_mask, *, scaling, sliding_window, dropout=0.0, softcap=0.0, **kwargs) -> (out, None)`

The adapter function itself. Exported in case you want to wrap / override.
Follows the transformers attention_interface contract.

### `patch_transformers_5_5_4_flash_attn_key() -> None`

Monkey-patches `transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING`
to fix the `KeyError: 'flash_attn'` bug in transformers 5.5.4. No-op on
other versions. Call once, before any config load.

## Direct kernel API (skip transformers)

### `attention_flash_gqa(q, k, v, causal=False, slide_size=0) -> out`

Inference forward. No autograd. Uses less memory than training (no LSE
stored).

### `flash_attn_gqa_train(q, k, v, causal=False, slide_size=0) -> out`

Training forward. Returns a tensor with autograd graph connected to a custom
backward.

### Reference implementations (for testing / sanity checks)

```python
attention_gqa_ref(q, k, v, causal=False)      # eager SDPA-equivalent
attention_swa_ref(q, k, v, slide_size, causal=False)  # eager SWA reference
```

## Input shapes

All tensors in `(B, H, N, D)` layout:

```
q: (B, N_Q_HEADS,  N, D)
k: (B, N_KV_HEADS, N, D)   # GQA: N_Q_HEADS must be a multiple of N_KV_HEADS
v: (B, N_KV_HEADS, N, D)
```

## Supported configurations

| Dimension | Supported |
|-----------|-----------|
| dtype | fp16, bf16 |
| HEAD_DIM | 64–512 (tested heavily at 256 and 512) |
| GQA ratio | 1:1 (MHA), 2:1, 4:1, 8:1 |
| `slide_size` | 0 (full causal) or any positive int |
| `causal` | True / False |
| device | CUDA only |

## Not supported

The following raise `NotImplementedError` (adapter) or simply aren't
implemented (direct kernel):

- Variable-length sequences / padding mask
- ALiBi or positional bias injection
- `softcap ≠ 0`
- `dropout > 0`
- `HEAD_DIM` outside 64–512
- CPU / MPS / ROCm
