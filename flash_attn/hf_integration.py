"""HuggingFace transformers integration.

Registers the Triton GQA kernel as a pluggable attention implementation that
any transformers model using the `ALL_ATTENTION_FUNCTIONS` registry can opt
into via `model.config._attn_implementation = "triton_gqa"`.

Tested on transformers>=5.5 with Gemma4-style models (sliding + full causal
interleaved GQA).

Typical usage:

    from gemma_triton_flash_attn import register_triton_attention
    register_triton_attention()

    model = AutoModelForCausalLM.from_pretrained(...)
    model.config._attn_implementation = "triton_gqa"
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = "triton_gqa"

Every attention layer now routes through the Triton kernel. No model code
changes required.
"""
from __future__ import annotations

import torch

from .attention import flash_attn_gqa_train


def triton_gqa_attention(
    module,
    query: torch.Tensor,            # (B, H_Q, N, D)
    key: torch.Tensor,              # (B, H_KV, N, D)
    value: torch.Tensor,            # (B, H_KV, N, D)
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    sliding_window: int | None = None,
    **kwargs,
):
    """Adapter matching transformers' `attention_interface` contract.

    Ignores `attention_mask` — our kernel builds its own causal / sliding-window
    mask internally from `sliding_window` and `module.is_causal`. HuggingFace's
    mask would normally encode the same thing, so ignoring it is safe when the
    kernel supports the requested pattern.

    Raises NotImplementedError for features the kernel doesn't support
    (softcap, nonzero dropout) — fail loudly rather than produce wrong numerics.
    """
    # Reconcile scaling. Our kernel bakes in 1/sqrt(D) internally. If the module
    # passes a different `scaling` (e.g., Gemma4 passes 1.0 because scaling is
    # folded into q_norm), pre-multiply q to cancel the kernel's internal scale.
    scale = scaling if scaling is not None else module.head_dim ** -0.5
    default_scale = query.shape[-1] ** -0.5
    if scale != default_scale:
        query = query * (scale / default_scale)

    if softcap is not None:
        raise NotImplementedError("triton_gqa_attention does not support softcap")
    if dropout != 0.0:
        raise NotImplementedError("triton_gqa_attention does not support dropout")

    slide = int(sliding_window) if sliding_window else 0
    is_causal = getattr(module, "is_causal", True)

    # Multi-GPU (device_map="auto") safety: Triton launches kernels on
    # torch.cuda.current_device(), NOT on the tensor's device. accelerate's
    # layer dispatch moves tensors across GPUs but doesn't switch the current
    # device, so a layer on cuda:N would launch its Triton kernel on cuda:0's
    # stream — silently producing NaN output. Wrap the launch in a device ctx.
    with torch.cuda.device(query.device):
        out = flash_attn_gqa_train(query, key, value, causal=is_causal, slide_size=slide)
    out = out.transpose(1, 2).contiguous()
    return out, None  # (attn_output, attn_weights)


def register_triton_attention(name: str = "triton_gqa") -> None:
    """Register the Triton GQA kernel under the given name in transformers'
    `ALL_ATTENTION_FUNCTIONS` registry.

    After calling this, set `model.config._attn_implementation = name` (and
    `model.config.text_config._attn_implementation = name` for multimodal
    configs) to route every attention layer through the Triton kernel.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[name] = triton_gqa_attention


def patch_transformers_5_5_4_flash_attn_key() -> None:
    """Workaround for a transformers 5.5.4 bug where loading **any** model
    raises `KeyError: 'flash_attn'` from `PACKAGE_DISTRIBUTION_MAPPING`.

    Call this right after `import transformers` and before loading a config.
    Safe to call multiple times (uses `setdefault`).
    """
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash-attn"])
