from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flash_attn.attention import attention_gqa_varlen_ref, flash_attn_gqa_varlen
from tests.numerics import Tolerance, assert_close


pytestmark = pytest.mark.gpu


@dataclass(frozen=True)
class VarlenProfile:
    name: str
    q_heads: int
    kv_heads: int
    head_dim: int
    lengths: tuple[int, ...]
    window_size: int
    padded_storage: bool = False


PROFILES = [
    VarlenProfile("full_skewed_d128", 8, 2, 128, (1, 17, 129), 0),
    VarlenProfile("gemma4_e2b_sliding_d256", 8, 1, 256, (33, 129), 64),
    VarlenProfile("gemma4_moe_sliding_d256", 16, 8, 256, (31, 65), 32),
    VarlenProfile("gemma4_e2b_full_d512", 8, 1, 512, (17, 65), 0, True),
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


def _random_packed(
    total: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    padded_storage: bool,
) -> torch.Tensor:
    if not padded_storage:
        return torch.randn(total, heads, head_dim, dtype=dtype, device="cuda")
    storage = torch.randn(total, heads, head_dim + 8, dtype=dtype, device="cuda")
    result = storage[..., :head_dim]
    assert not result.is_contiguous() and result.stride(-1) == 1
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_varlen_forward_backward_matches_independent_reference(
    profile: VarlenProfile, dtype: torch.dtype
) -> None:
    torch.manual_seed(20260801)
    total = sum(profile.lengths)
    max_seqlen = max(profile.lengths)
    q_data = _random_packed(
        total,
        profile.q_heads,
        profile.head_dim,
        dtype,
        padded_storage=profile.padded_storage,
    )
    k_data = _random_packed(
        total,
        profile.kv_heads,
        profile.head_dim,
        dtype,
        padded_storage=profile.padded_storage,
    )
    v_data = _random_packed(
        total,
        profile.kv_heads,
        profile.head_dim,
        dtype,
        padded_storage=profile.padded_storage,
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(profile.lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    grad_out = torch.randn(
        total,
        profile.q_heads,
        profile.head_dim,
        dtype=dtype,
        device="cuda",
    )

    q_ref = q_data.detach().clone().requires_grad_(True)
    k_ref = k_data.detach().clone().requires_grad_(True)
    v_ref = v_data.detach().clone().requires_grad_(True)
    expected = attention_gqa_varlen_ref(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=True,
        window_size=profile.window_size,
    )
    expected.backward(grad_out)

    q = q_data.detach().requires_grad_(True)
    k = k_data.detach().requires_grad_(True)
    v = v_data.detach().requires_grad_(True)
    actual = flash_attn_gqa_varlen(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=True,
        window_size=profile.window_size,
    )
    actual.backward(grad_out)

    assert_close(
        actual,
        expected,
        name=f"{profile.name}/{dtype}/forward",
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
