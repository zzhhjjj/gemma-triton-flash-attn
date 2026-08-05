from __future__ import annotations

import pytest
import torch

from flash_attn.attention import attention_flash_gqa, attention_gqa_ref
from tests.numerics import Tolerance, assert_close


pytestmark = pytest.mark.gpu


TOLERANCE = Tolerance(
    cosine_min=0.9999,
    max_abs=5e-2,
    mean_abs=7e-4,
    relative_l2=2e-2,
)


def _inputs(*, batch: int = 1, length: int = 65, head_dim: int = 64):
    torch.manual_seed(20260801)
    return (
        torch.randn(batch, 8, length, head_dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch, 1, length, head_dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch, 1, length, head_dim, device="cuda", dtype=torch.bfloat16),
    )


def test_causal_prefix_is_independent_of_future_kv() -> None:
    q, k, v = _inputs()
    prefix = 32
    baseline = attention_flash_gqa(q, k, v, causal=True)
    changed_k = k.clone()
    changed_v = v.clone()
    changed_k[:, :, prefix:, :] = torch.randn_like(changed_k[:, :, prefix:, :]) * 10
    changed_v[:, :, prefix:, :] = torch.randn_like(changed_v[:, :, prefix:, :]) * 10
    changed = attention_flash_gqa(q, changed_k, changed_v, causal=True)
    torch.testing.assert_close(
        changed[:, :, :prefix, :], baseline[:, :, :prefix, :], rtol=0, atol=0
    )


def test_sliding_queries_are_independent_of_tokens_outside_window() -> None:
    q, k, v = _inputs()
    window = 16
    baseline = attention_flash_gqa(q, k, v, causal=True, slide_size=window)
    changed_k = k.clone()
    changed_v = v.clone()
    changed_k[:, :, 0, :] = 100
    changed_v[:, :, 0, :] = -100
    changed = attention_flash_gqa(
        q, changed_k, changed_v, causal=True, slide_size=window
    )
    # Token 0 is outside the window for queries q >= window.
    torch.testing.assert_close(
        changed[:, :, window:, :], baseline[:, :, window:, :], rtol=0, atol=0
    )


def test_batch_items_do_not_share_attention_state() -> None:
    q, k, v = _inputs(batch=2)
    baseline = attention_flash_gqa(q, k, v, causal=True)
    changed_k = k.clone()
    changed_v = v.clone()
    changed_k[1] = torch.randn_like(changed_k[1]) * 10
    changed_v[1] = torch.randn_like(changed_v[1]) * 10
    changed = attention_flash_gqa(q, changed_k, changed_v, causal=True)
    torch.testing.assert_close(changed[0], baseline[0], rtol=0, atol=0)


@pytest.mark.parametrize("window", [0, 16], ids=["full", "sliding"])
def test_constant_v_produces_constant_output(window: int) -> None:
    q, k, _ = _inputs()
    v = torch.full_like(k, 0.25)
    actual = attention_flash_gqa(
        q, k, v, causal=True, slide_size=window
    )
    expected = torch.full_like(actual, 0.25)
    assert_close(actual, expected, name=f"constant_v/window={window}", tolerance=TOLERANCE)


def test_noncontiguous_last_dimension_matches_reference() -> None:
    torch.manual_seed(20260801)
    q_storage = torch.randn(1, 8, 33, 128, device="cuda", dtype=torch.bfloat16)
    k_storage = torch.randn(1, 1, 33, 128, device="cuda", dtype=torch.bfloat16)
    v_storage = torch.randn(1, 1, 33, 128, device="cuda", dtype=torch.bfloat16)
    q = q_storage[..., ::2]
    k = k_storage[..., ::2]
    v = v_storage[..., ::2]
    assert not q.is_contiguous() and q.stride(-1) == 2
    expected = attention_gqa_ref(q, k, v, causal=True)
    actual = attention_flash_gqa(q, k, v, causal=True)
    assert_close(actual, expected, name="noncontiguous_stride_d", tolerance=TOLERANCE)


def test_invalid_gqa_ratio_fails_before_kernel_launch() -> None:
    q = torch.randn(1, 6, 17, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, 17, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="q_heads must be divisible by kv_heads"):
        attention_flash_gqa(q, k, v, causal=True)
