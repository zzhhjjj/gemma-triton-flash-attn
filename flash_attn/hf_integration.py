"""HuggingFace transformers integration.

Registers the Triton GQA kernel as a pluggable attention implementation that
any Transformers model using `AttentionInterface` can opt into via
`model.config._attn_implementation = "triton_gqa"`.

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

import contextvars
from typing import Callable, NamedTuple

import torch

from .attention import flash_attn_gqa_train


# =====================================================================
# Image-bidirectional-mask plumbing for Gemma-4 multimodal training
#
# Gemma-4 (when `text_config.use_bidirectional_attention == "vision"`,
# currently only the 26B-A4B MoE) builds an OR-mask on top of the standard
# sliding causal mask: tokens within the same vision span attend to each
# other bidirectionally (`q_group == k_group & q_group >= 0`). See
# `transformers.models.gemma4.modeling_gemma4.create_causal_mask_mapping`
# (it sets `sliding_mask_kwargs["or_mask_function"]` only — full-attention
# layers stay pure-causal).
#
# The mask reaches our adapter as a `BlockMask` / 4D bool tensor with the
# group info baked in — there's no public API to extract it back. We solve
# this by patching `Gemma4Model.forward` to compute the group state once
# per forward and stash it in a ContextVar that the adapter reads.
#
# Adapter contract: when this ContextVar is set AND the layer is a sliding
# layer (sliding_window != None), pass the group tensors to the kernel.
# Full-attention layers ignore them (matches upstream).
# =====================================================================


class _ImageGroupState(NamedTuple):
    group_ids: torch.Tensor  # (B, N) int32; -1 = non-vision token
    group_lo: torch.Tensor  # (B, N) int32; first KV idx in same group (or self)
    group_hi_excl: torch.Tensor  # (B, N) int32; last KV idx+1 in same group (or self+1)


_image_group_state: contextvars.ContextVar[_ImageGroupState | None] = (
    contextvars.ContextVar("_triton_gqa_image_group_state", default=None)
)


def _compute_image_group_state(mm_token_type_ids: torch.Tensor) -> _ImageGroupState:
    """Replicate Gemma-4's `vision_group_ids` derivation, plus precompute the
    per-token bidirectional reach (`group_lo`, `group_hi_excl`) the kernel
    needs to extend its loop bounds.

    Source for the group derivation: `modeling_gemma4.py` lines 2052-2057
    (must stay in sync with upstream).
    """
    # mm_token_type_ids: (B, N) int. vision tokens have type_id 1 or 2.
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    vision_group_ids = torch.where(is_vision, vision_group_ids, -1)

    B, N = vision_group_ids.shape
    device = vision_group_ids.device
    idx = torch.arange(N, device=device, dtype=torch.int32)
    idx_b = idx.unsqueeze(0).expand(B, -1)

    # Bucket = group_id + 1 so non-vision (-1) maps to bucket 0 (sentinel,
    # never read back). Vision groups occupy buckets 1..G_max.
    buckets = (vision_group_ids + 1).long()
    G_buckets = int(buckets.max().item()) + 1

    group_min = torch.full((B, G_buckets), N, dtype=torch.int32, device=device)
    group_max = torch.full((B, G_buckets), -1, dtype=torch.int32, device=device)
    group_min.scatter_reduce_(1, buckets, idx_b, reduce="amin", include_self=True)
    group_max.scatter_reduce_(1, buckets, idx_b, reduce="amax", include_self=True)

    gathered_min = group_min.gather(1, buckets)
    gathered_max = group_max.gather(1, buckets)
    # Non-vision tokens contribute their own position only — they don't
    # expand any block's iteration range beyond the standard SWA window.
    group_lo = torch.where(is_vision, gathered_min, idx_b)
    group_hi_excl = torch.where(is_vision, gathered_max + 1, idx_b + 1)

    return _ImageGroupState(
        group_ids=vision_group_ids.to(torch.int32).contiguous(),
        group_lo=group_lo.to(torch.int32).contiguous(),
        group_hi_excl=group_hi_excl.to(torch.int32).contiguous(),
    )


def triton_gqa_attention(
    module,
    query: torch.Tensor,  # (B, H_Q, N, D)
    key: torch.Tensor,  # (B, H_KV, N, D)
    value: torch.Tensor,  # (B, H_KV, N, D)
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
    scale = scaling if scaling is not None else module.head_dim**-0.5
    default_scale = query.shape[-1] ** -0.5
    if scale != default_scale:
        query = query * (scale / default_scale)

    if softcap is not None:
        raise NotImplementedError("triton_gqa_attention does not support softcap")
    if dropout != 0.0:
        raise NotImplementedError("triton_gqa_attention does not support dropout")

    slide = int(sliding_window) if sliding_window else 0
    is_causal = getattr(module, "is_causal", True)

    # Image-bidirectional mask path (Gemma-4 MoE multimodal training):
    # Sliding layers get an OR-mask that grants bidirectional attention
    # within each vision span. Full layers don't (matches upstream:
    # `create_causal_mask_mapping` only sets `or_mask_function` on
    # `sliding_mask_kwargs`). The state is set by
    # `patch_gemma4_image_group_ids_for_kernel` before the model forward.
    group_state = _image_group_state.get()
    use_image_groups = (group_state is not None) and (slide > 0) and is_causal

    # Defensive guard: if the caller passed a 4D / BlockMask but no group
    # state is in context, image bidirectional info would be silently
    # dropped — fail loudly so the user installs the patch.
    if (
        (group_state is None)
        and (attention_mask is not None)
        and slide > 0
        and is_causal
    ):
        is_4d = isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4
        from torch.nn.attention.flex_attention import BlockMask

        is_blockmask = isinstance(attention_mask, BlockMask)
        if is_4d or is_blockmask:
            raise RuntimeError(
                "triton_gqa_attention received a 4D / BlockMask attention_mask on "
                "a sliding layer but no image-group state is set. If you are "
                "training Gemma-4 multimodal, call "
                "`patch_gemma4_image_group_ids_for_kernel()` before model load."
            )

    if use_image_groups:
        # Cross-device safety: device_map="auto" spreads decoder layers across
        # GPUs, but the ContextVar state was built on the input's device. Move
        # the (small: B × N int64) tensors to the current Q device per-call;
        # at ~16 KB per tensor × 3 tensors × 30 layers this is sub-ms overhead.
        dev = query.device
        group_args = (
            group_state.group_ids.to(dev, non_blocking=True),
            group_state.group_lo.to(dev, non_blocking=True),
            group_state.group_hi_excl.to(dev, non_blocking=True),
        )
    else:
        group_args = (None, None, None)

    # Multi-GPU (device_map="auto") safety: Triton launches kernels on
    # torch.cuda.current_device(), NOT on the tensor's device. accelerate's
    # layer dispatch moves tensors across GPUs but doesn't switch the current
    # device, so a layer on cuda:N would launch its Triton kernel on cuda:0's
    # stream — silently producing NaN output. Wrap the launch in a device ctx.
    with torch.cuda.device(query.device):
        out = flash_attn_gqa_train(
            query,
            key,
            value,
            causal=is_causal,
            slide_size=slide,
            group_ids=group_args[0],
            group_lo=group_args[1],
            group_hi_excl=group_args[2],
        )
    out = out.transpose(1, 2).contiguous()
    return out, None  # (attn_output, attn_weights)


def register_triton_attention(name: str = "triton_gqa") -> None:
    """Register the Triton GQA kernel through Transformers' public attention API.

    After calling this, set `model.config._attn_implementation = name` (and
    `model.config.text_config._attn_implementation = name` for multimodal
    configs) to route every attention layer through the Triton kernel.
    """
    _register_attention(name, triton_gqa_attention)


def _register_attention(name: str, attention_fn: Callable) -> None:
    """Register an attention callable through Transformers' public API."""
    if not name or name == "eager":
        raise ValueError(
            "A custom attention implementation needs a non-empty name other than 'eager'."
        )

    try:
        from transformers import AttentionInterface
    except ImportError as exc:
        raise ImportError(
            "gemma-triton-flash-attn's Hugging Face integration requires transformers>=5.5.0."
        ) from exc

    AttentionInterface.register(name, attention_fn)


def _build_ulysses_attention(
    cp_group, attention_fn: Callable = triton_gqa_attention
) -> Callable:
    """Build a Transformers attention adapter with Ulysses all-to-all communication.

    This factory is intentionally separate from registration so its distributed
    layout and autograd behavior can be tested with a small PyTorch attention
    reference on CPU-only CI.
    """
    import os

    import torch.distributed as dist

    from .cp_comm import SeqAllToAll4D

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Ulysses attention must be registered after torch.distributed is initialized."
        )

    cp_size = dist.get_world_size(cp_group)
    if cp_size < 2:
        raise ValueError(
            "Ulysses attention requires a context-parallel group with at least two ranks."
        )

    seen = 0

    def ulysses_attention(module, query, key, value, attention_mask=None, **kwargs):
        nonlocal seen
        seen += 1

        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "Ulysses attention expects Q, K, and V in (batch, heads, sequence, dim) layout."
            )
        if key.shape != value.shape:
            raise ValueError(
                f"K and V must have the same shape, got {key.shape} and {value.shape}."
            )
        if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
            raise ValueError(
                f"Q and K/V batch, sequence, and head dimensions must match: {query.shape}, {key.shape}."
            )
        if attention_mask is not None:
            raise NotImplementedError(
                "The dense Ulysses adapter currently supports kernel-owned causal/sliding masks only. "
                "Use the varlen adapter for packed samples."
            )

        num_query_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        # When a Gemma global-attention layer has fewer KV heads than CP ranks,
        # expand K/V to MHA before scattering. This avoids empty all-to-all chunks.
        if num_kv_heads < cp_size:
            kv_repeat = num_query_heads // num_kv_heads
            key = key.repeat_interleave(kv_repeat, dim=1)
            value = value.repeat_interleave(kv_repeat, dim=1)
            num_kv_heads = key.shape[1]

        if num_query_heads % cp_size != 0 or num_kv_heads % cp_size != 0:
            raise ValueError(
                "Ulysses requires the post-replication Q and KV head counts to be divisible by "
                f"cp_size={cp_size}, got Hq={num_query_heads} and Hkv={num_kv_heads}."
            )

        if seen <= 6 and os.environ.get("CP_DEBUG", "0") == "1":
            if dist.get_rank() == 0:
                print(
                    f"[triton_gqa][cp] call={seen} q={tuple(query.shape)} "
                    f"k={tuple(key.shape)} sliding_window={kwargs.get('sliding_window')}",
                    flush=True,
                )

        # (B, H, N_local, D) -> (B, H/cp, N_full, D)
        query = SeqAllToAll4D.apply(cp_group, query, 1, 2)
        key = SeqAllToAll4D.apply(cp_group, key, 1, 2)
        value = SeqAllToAll4D.apply(cp_group, value, 1, 2)

        result = attention_fn(module, query, key, value, attention_mask=None, **kwargs)
        output = result[0] if isinstance(result, tuple) else result
        if output.ndim != 4:
            raise ValueError(
                f"The wrapped attention must return a 4D output, got shape {output.shape}."
            )

        # Transformers attention functions return (B, N_full, H/cp, D).
        output = output.transpose(1, 2).contiguous()
        output = SeqAllToAll4D.apply(cp_group, output, 2, 1)
        output = output.transpose(1, 2).contiguous()
        return output, None

    return ulysses_attention


def register_triton_attention_ulysses(
    cp_group, name: str = "triton_gqa_ulysses"
) -> None:
    """Register a Ulysses-context-parallel variant of triton_gqa.

    Wraps each attention call with an all-to-all that scatters head dim and
    gathers seq dim across `cp_group`, runs local attention on the full seq
    with a head-shard, then inverts the all-to-all.
    """
    _register_attention(name, _build_ulysses_attention(cp_group))


# =====================================================================
# Varlen (packed-sequence) attention adapter
# =====================================================================

_varlen_cu_seqlens_state = {"value": None}


def set_varlen_cu_seqlens(cu_seqlens, max_seqlen):
    """Store cu_seqlens for the current forward pass. Called by the trainer
    before model() when neat_packing is active. Uses module global (not
    ContextVar) so gradient checkpointing recompute can access it."""
    old = _varlen_cu_seqlens_state["value"]
    _varlen_cu_seqlens_state["value"] = (cu_seqlens, max_seqlen)
    return old


def clear_varlen_cu_seqlens(token):
    """Reset after forward+backward pass."""
    _varlen_cu_seqlens_state["value"] = token


def cu_seqlens_from_2d_indices(indices_2d):
    """Convert LF's 2D attention_mask indices [1,1,2,2,2,0] to cu_seqlens.

    Args:
        indices_2d: (B, N) or (N,) int tensor. Non-zero values are sample IDs.

    Returns:
        cu_seqlens: int32 (num_samples+1,), total_valid: int
    """
    import torch

    indices = indices_2d.squeeze(0) if indices_2d.dim() == 2 else indices_2d
    mask = indices != 0
    total_valid = int(mask.sum().item())
    if total_valid == 0:
        return torch.zeros(1, dtype=torch.int32, device=indices.device), 0
    valid_indices = indices[mask]
    unique_ids = valid_indices.unique(sorted=True)
    boundaries = [0]
    for uid in unique_ids:
        boundaries.append(boundaries[-1] + int((valid_indices == uid).sum().item()))
    return torch.tensor(
        boundaries, dtype=torch.int32, device=indices.device
    ), total_valid


def triton_gqa_varlen_attention(
    module,
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    attention_mask: "torch.Tensor | None",
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    sliding_window: int | None = None,
    **kwargs,
):
    """Varlen attention adapter for packed sequences.

    Expects cu_seqlens to be set via `set_varlen_cu_seqlens` before the model
    forward, or extracts them from a 2D attention_mask (LF neat_packing indices).
    Falls back to batched triton_gqa_attention if no packing info is available.
    """
    import torch
    from .attention import flash_attn_gqa_varlen

    state = _varlen_cu_seqlens_state["value"]
    if state is None and attention_mask is not None and attention_mask.dim() == 2:
        cu, total = cu_seqlens_from_2d_indices(attention_mask)
        if total > 0:
            max_sl = max(int(cu[i + 1] - cu[i]) for i in range(cu.numel() - 1))
            state = (cu, max_sl)

    if state is None:
        return triton_gqa_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            softcap=softcap,
            sliding_window=sliding_window,
            **kwargs,
        )

    cu_seqlens, max_seqlen = state

    scale = scaling if scaling is not None else module.head_dim**-0.5
    default_scale = query.shape[-1] ** -0.5
    if scale != default_scale:
        query = query * (scale / default_scale)

    if softcap is not None:
        raise NotImplementedError("triton_gqa_varlen does not support softcap")

    slide = int(sliding_window) if sliding_window else 0

    B, H_Q, N, D = query.shape
    _, H_KV, _, _ = key.shape
    q_packed = query.squeeze(0).permute(1, 0, 2).contiguous()  # (N, H_Q, D)
    k_packed = key.squeeze(0).permute(1, 0, 2).contiguous()
    v_packed = value.squeeze(0).permute(1, 0, 2).contiguous()

    # cu_seqlens already includes dummy padding sample (set by trainer)
    with torch.cuda.device(query.device):
        out_packed = flash_attn_gqa_varlen(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=getattr(module, "is_causal", True),
            window_size=slide,
        )

    out = out_packed.permute(1, 0, 2).unsqueeze(0).contiguous()  # (1, H_Q, N, D)
    out = out.transpose(1, 2).contiguous()  # (1, N, H_Q, D)
    return out, None


def register_triton_attention_varlen(name: str = "triton_gqa_varlen") -> None:
    """Register the varlen adapter."""
    _register_attention(name, triton_gqa_varlen_attention)


def register_triton_attention_varlen_ulysses(
    cp_group, name: str = "triton_gqa_varlen_ulysses"
) -> None:
    """Register a Ulysses CP + varlen variant.

    Wraps each attention call with all-to-all (scatter heads, gather tokens),
    runs varlen attention on the reassembled full sequence, then inverses.
    """
    import torch
    import torch.distributed as _dist
    from .cp_comm import SeqAllToAll4D
    from .attention import flash_attn_gqa_varlen

    def _varlen_ulysses_wrapped(
        module, query, key, value, attention_mask=None, **kwargs
    ):
        cp_size = _dist.get_world_size(cp_group)

        if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
            raise ValueError(
                "The varlen Ulysses adapter requires batch_size=1 packed input."
            )
        if kwargs.get("softcap") is not None:
            raise NotImplementedError(
                "triton_gqa_varlen_ulysses does not support softcap"
            )
        if kwargs.get("dropout", 0.0) != 0.0:
            raise NotImplementedError(
                "triton_gqa_varlen_ulysses does not support attention dropout"
            )

        scale = kwargs.get("scaling", None)
        if scale is None:
            scale = module.head_dim**-0.5
        default_scale = query.shape[-1] ** -0.5
        if scale != default_scale:
            query = query * (scale / default_scale)
        kwargs.pop("scaling", None)

        slide = int(kwargs.get("sliding_window", 0) or 0)
        is_causal = getattr(module, "is_causal", True)

        B, H_Q, N_local, D = query.shape
        H_KV = key.shape[1]

        if H_Q % H_KV != 0:
            raise ValueError(f"H_Q ({H_Q}) must be divisible by H_KV ({H_KV}).")

        # KV replication for GQA when H_KV < cp_size
        if H_KV < cp_size:
            kv_rep = H_Q // H_KV
            key = key.repeat_interleave(kv_rep, dim=1)
            value = value.repeat_interleave(kv_rep, dim=1)
            H_KV = key.shape[1]

        if H_Q % cp_size != 0 or H_KV % cp_size != 0:
            raise ValueError(
                f"Post-replication head counts must be divisible by cp_size={cp_size}, "
                f"got H_Q={H_Q}, H_KV={H_KV}."
            )

        # Reshape to packed: (B=1, H, N_local, D) -> (N_local, H, D)
        q_local = query.squeeze(0).permute(1, 0, 2).contiguous()
        k_local = key.squeeze(0).permute(1, 0, 2).contiguous()
        v_local = value.squeeze(0).permute(1, 0, 2).contiguous()

        # All-to-all: scatter heads (dim=1), gather tokens (dim=0)
        q_full = SeqAllToAll4D.apply(cp_group, q_local, 1, 0)
        k_full = SeqAllToAll4D.apply(cp_group, k_local, 1, 0)
        v_full = SeqAllToAll4D.apply(cp_group, v_local, 1, 0)

        # cu_seqlens from ContextVar (set by trainer, includes dummy padding sample)
        N_full = q_full.shape[0]
        state = _varlen_cu_seqlens_state["value"]
        if state is not None:
            cu_seqlens, max_seqlen = state
        else:
            cu_seqlens = torch.tensor(
                [0, N_full], dtype=torch.int32, device=q_full.device
            )
            max_seqlen = N_full

        with torch.cuda.device(query.device):
            out_full = flash_attn_gqa_varlen(
                q_full.contiguous(),
                k_full.contiguous(),
                v_full.contiguous(),
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=is_causal,
                window_size=slide,
            )

        # Inverse all-to-all: scatter tokens (dim=0), gather heads (dim=1)
        out_local = SeqAllToAll4D.apply(cp_group, out_full, 0, 1)

        # Reshape back to (1, H_Q, N_local, D) then (1, N_local, H_Q, D)
        out = out_local.permute(1, 0, 2).unsqueeze(0).contiguous()
        out = out.transpose(1, 2).contiguous()
        return out, None

    _register_attention(name, _varlen_ulysses_wrapped)


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
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(
                inputs_embeds, per_layer_inputs
            )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            import torch

            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
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
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        # THE FIX: use pytree-opaque holder so FSDP2's per-layer flatten/unflatten
        # preserves the dict-like state across decoder-layer boundaries.
        shared_kv_states = _SharedKVStatesHolder()

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
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


def patch_gemma4_image_group_ids_for_kernel() -> None:
    """Patch `Gemma4Model.forward` to compute the image-group state and stash
    it in a ContextVar so the Triton adapter can fold the image-bidirectional
    OR-mask into the kernel.

    Required when training Gemma-4 multimodal **and** the text config has
    `use_bidirectional_attention == "vision"` (currently only Gemma-4-26B-A4B
    MoE). For E2B/E4B (`use_bidirectional_attention is None`) this is a no-op
    in effect — the wrapper still sets up the ContextVar machinery but the
    state stays empty so the adapter takes the standard causal/SWA path.

    Idempotent (marks the patched method with an attribute). Safe to call
    after `register_triton_attention()`.
    """
    from transformers.models.gemma4 import modeling_gemma4 as mg

    if getattr(mg.Gemma4Model.forward, "_patched_for_image_groups", False):
        return

    orig_forward = mg.Gemma4Model.forward

    def _wrapped(self, *args, **kwargs):
        # Trigger only when (a) text-config opts into vision-bidirectional and
        # (b) we actually have mm_token_type_ids (the prefill / training case
        # — incremental decode doesn't carry them).
        text_cfg = self.config.get_text_config()
        wants_groups = (
            getattr(text_cfg, "use_bidirectional_attention", None) == "vision"
        )
        mm_token_type_ids = kwargs.get("mm_token_type_ids", None)

        token = None
        if wants_groups and mm_token_type_ids is not None:
            state = _compute_image_group_state(mm_token_type_ids)
            token = _image_group_state.set(state)
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            if token is not None:
                _image_group_state.reset(token)

    _wrapped._patched_for_image_groups = True
    _wrapped._original_forward = orig_forward
    mg.Gemma4Model.forward = _wrapped
