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


class _SharedKVStatesHolder:
    """Opaque container for Gemma-4 cross-layer KV sharing under FSDP2.

    FSDP2's per-module `pre_forward` hook runs `tree_flatten` / `tree_unflatten`
    on kwargs to register a backward hook on grad-requiring tensors. `dict` is a
    registered pytree container, so unflatten rebuilds a *new* empty dict — the
    writes from earlier layers vanish, and later `is_kv_shared_layer` reads fail
    with `KeyError`. An object not registered with pytree is treated as a leaf,
    so its identity (and contents) survive the flatten/unflatten round-trip.
    Subscript ops `[idx]` are routed to an internal dict to match the original
    `shared_kv_states: dict[int, tuple[K, V]]` contract used by Gemma-4 code.
    """
    __slots__ = ("_d",)

    def __init__(self):
        self._d = {}

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v

    def __contains__(self, k):
        return k in self._d


def patch_gemma4_shared_kv_states_for_fsdp2() -> None:
    """Replace `Gemma4TextModel.forward`'s `shared_kv_states = {}` with an
    opaque holder that survives FSDP2's pytree flatten/unflatten.

    Required when wrapping each `Gemma4TextDecoderLayer` with `fully_shard()`
    for per-layer parameter sharding (which gives proper all-gather/compute
    overlap). Without this, the dict identity is lost across FSDP2 boundaries
    and layers past the KV-sharing point raise `KeyError` on lookup.

    No-op for single-boundary sharding (no per-layer wrap), but still safe to
    call. Idempotent (marks the patched method with an attribute).
    """
    from transformers.models.gemma4 import modeling_gemma4 as mg

    if getattr(mg.Gemma4TextModel.forward, "_patched_for_fsdp2", False):
        return

    orig_forward = mg.Gemma4TextModel.forward

    # Re-implemented body of Gemma4TextModel.forward with exactly one change:
    # `shared_kv_states = _SharedKVStatesHolder()` instead of `{}`. Done this
    # way (full body copy) because the dict is constructed inline with no hook
    # point we could intercept from outside. Drops the original's
    # `@merge_with_config_defaults @capture_outputs @auto_docstring` decorators
    # which only add ancillary features (config defaults, intermediate-output
    # capture, docstrings) — none are needed for training.
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    def _forward_impl(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        per_layer_inputs=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            import torch
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # THE FIX: use pytree-opaque holder so FSDP2's per-layer flatten/unflatten
        # preserves the dict-like state across decoder-layer boundaries.
        shared_kv_states = _SharedKVStatesHolder()

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    _forward_impl._patched_for_fsdp2 = True
    _forward_impl._original_forward = orig_forward
    mg.Gemma4TextModel.forward = _forward_impl
