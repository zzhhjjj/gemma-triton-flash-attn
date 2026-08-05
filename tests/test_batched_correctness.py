from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flash_attn.attention import (
    attention_flash_gqa,
    attention_gqa_ref,
    attention_swa_ref,
    flash_attn_gqa_train,
)
from tests.numerics import Tolerance, assert_close


pytestmark = pytest.mark.gpu


@dataclass(frozen=True)
class Profile:
    name: str
    q_heads: int
    kv_heads: int
    seq_len: int
    head_dim: int
    window_size: int


PROFILES = [
    # Exact E2B head ratio/head dim with a non-tile-aligned sequence length.
    Profile("gemma4_e2b_full_boundary", 8, 1, 129, 512, 0),
    # Exact Gemma-4 26B-A4B MoE full-attention shape.
    Profile("gemma4_moe_full_boundary", 16, 2, 129, 512, 0),
    # Exact E2B sliding head ratio/head dim with a real truncated window.
    Profile("gemma4_e2b_sliding_boundary", 8, 1, 257, 256, 128),
    # Exact Gemma-4 MoE sliding head ratio/head dim.
    Profile("gemma4_moe_sliding_boundary", 16, 8, 257, 256, 128),
]

OUTPUT_TOLERANCE = Tolerance(
    cosine_min=0.9999,
    max_abs=5e-2,
    mean_abs=7e-4,
    relative_l2=2e-2,
)
GRAD_TOLERANCE = Tolerance(
    cosine_min=0.9999,
    max_abs=2e-1,
    mean_abs=2e-3,
    relative_l2=3e-2,
)


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int):
    if window:
        return attention_swa_ref(q, k, v, slide_size=window)
    return attention_gqa_ref(q, k, v, causal=True)


def _inputs(profile: Profile, dtype: torch.dtype):
    torch.manual_seed(20260801)
    shape_q = (1, profile.q_heads, profile.seq_len, profile.head_dim)
    shape_kv = (1, profile.kv_heads, profile.seq_len, profile.head_dim)
    return (
        torch.randn(shape_q, dtype=dtype, device="cuda"),
        torch.randn(shape_kv, dtype=dtype, device="cuda"),
        torch.randn(shape_kv, dtype=dtype, device="cuda"),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_batched_inference_matches_independent_reference(
    profile: Profile, dtype: torch.dtype
) -> None:
    q, k, v = _inputs(profile, dtype)
    expected = _reference(q, k, v, profile.window_size)

    inference = attention_flash_gqa(
        q,
        k,
        v,
        causal=True,
        slide_size=profile.window_size,
    )
    assert_close(
        inference,
        expected,
        name=f"{profile.name}/{dtype}/inference",
        tolerance=OUTPUT_TOLERANCE,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_batched_autograd_matches_independent_reference(
    profile: Profile, dtype: torch.dtype
) -> None:
    q_data, k_data, v_data = _inputs(profile, dtype)
    q_ref = q_data.clone().requires_grad_(True)
    k_ref = k_data.clone().requires_grad_(True)
    v_ref = v_data.clone().requires_grad_(True)
    expected = _reference(q_ref, k_ref, v_ref, profile.window_size)
    grad_out = torch.randn_like(expected)
    expected.backward(grad_out)

    q = q_data.clone().requires_grad_(True)
    k = k_data.clone().requires_grad_(True)
    v = v_data.clone().requires_grad_(True)
    actual = flash_attn_gqa_train(
        q,
        k,
        v,
        causal=True,
        slide_size=profile.window_size,
    )
    actual.backward(grad_out)

    assert_close(
        actual,
        expected,
        name=f"{profile.name}/{dtype}/training_forward",
        tolerance=OUTPUT_TOLERANCE,
    )
    for name, actual_grad, expected_grad in (
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        assert actual_grad is not None and expected_grad is not None
        assert_close(
            actual_grad,
            expected_grad,
            name=f"{profile.name}/{dtype}/{name}",
            tolerance=GRAD_TOLERANCE,
        )
