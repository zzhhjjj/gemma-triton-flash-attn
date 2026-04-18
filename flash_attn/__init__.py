"""gemma-triton-flash-attn: Triton Flash Attention for GQA + large HEAD_DIM + SWA.

Public API:
    attention_flash_gqa(q, k, v, causal=False, slide_size=0) -> out
        Inference forward. No autograd support; uses less memory.

    flash_attn_gqa_train(q, k, v, causal=False, slide_size=0) -> out
        Training forward with autograd support for backward pass.

    attention_gqa_ref(q, k, v, causal=False) -> out
        PyTorch SDPA reference implementation (GQA with KV head expansion).

    attention_swa_ref(q, k, v, slide_size) -> out
        PyTorch reference for sliding window attention (GQA + causal + window).

Input shapes:
    q: (B, N_Q_HEADS,  N, D)
    k: (B, N_KV_HEADS, N, D)  — supports GQA (N_Q_HEADS must be multiple of N_KV_HEADS)
    v: (B, N_KV_HEADS, N, D)

Supported:
    - HEAD_DIM: typically 64-512 (tested: 256, 512)
    - dtype: fp16, bf16
    - causal: bool
    - slide_size: 0 (no window) or positive int (window covers last slide_size positions)

Example:
    >>> import torch
    >>> from gemma_triton_flash_attn import flash_attn_gqa_train
    >>> q = torch.randn(1, 32, 2048, 512, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    >>> k = torch.randn(1,  4, 2048, 512, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    >>> v = torch.randn(1,  4, 2048, 512, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    >>> out = flash_attn_gqa_train(q, k, v, causal=True)
    >>> out.sum().backward()
"""

from .attention import (
    # Primary Triton kernels (public API)
    attention_flash_gqa,
    flash_attn_gqa_train,
    FlashAttnGQAFunction,
    # Reference implementations
    attention_gqa_ref,
    attention_swa_ref,
    # Standard scaled dot-product attention (PyTorch fallback)
    attention,
)
# HuggingFace transformers integration
from .hf_integration import (
    register_triton_attention,
    triton_gqa_attention,
    patch_transformers_5_5_4_flash_attn_key,
    patch_gemma4_shared_kv_states_for_fsdp2,
)

__version__ = "0.1.0"

__all__ = [
    "attention_flash_gqa",
    "flash_attn_gqa_train",
    "FlashAttnGQAFunction",
    "attention_gqa_ref",
    "attention_swa_ref",
    "attention",
    # HF transformers integration
    "register_triton_attention",
    "triton_gqa_attention",
    "patch_transformers_5_5_4_flash_attn_key",
    "patch_gemma4_shared_kv_states_for_fsdp2",
    "__version__",
]
