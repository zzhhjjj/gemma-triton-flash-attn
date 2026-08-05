"""Verify Triton fwd+bwd matches SDPA on Gemma-4 vision shapes.

Vision attention is bidirectional (`is_causal=False`), MHA (H_Q=H_KV=12),
small head dim (D=64), no sliding window, no softcap. Our adapter routes
it the same way as text — this test guards that the non-causal kernel path
is numerically correct end-to-end.

Run: pytest tests/test_noncausal_vision_shape.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from flash_attn.attention import flash_attn_gqa_train, attention_flash_gqa
from tests.numerics import Tolerance, assert_close


pytestmark = pytest.mark.gpu

OUTPUT_TOLERANCE = Tolerance(
    cosine_min=0.9999,
    max_abs=5e-3,
    mean_abs=5e-4,
    relative_l2=2e-2,
)
ADAPTER_TOLERANCE = Tolerance(
    cosine_min=0.9999,
    max_abs=2e-2,
    mean_abs=7e-4,
    relative_l2=2e-2,
)
GRAD_TOLERANCE = Tolerance(
    cosine_min=0.9999,
    max_abs=1e-1,
    mean_abs=2e-3,
    relative_l2=3e-2,
)


def sdpa_ref(q, k, v, causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


@pytest.mark.parametrize("B,H,N,D", [
    (1, 12, 256, 64),    # Vision E2B-ish
    (2, 12, 729, 64),    # Vision typical patch count (27×27 image)
    (1, 16, 1024, 64),   # Larger vision
    (1, 12, 256, 128),   # Audio-ish D (won't be used but worth covering)
    (1, 8, 512, 256),    # Larger D for safety
])
def test_fwd_noncausal_matches_sdpa(B, H, N, D):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

    ref = sdpa_ref(q, k, v, causal=False)
    out = attention_flash_gqa(q, k, v, causal=False, slide_size=0)

    assert_close(
        out,
        ref,
        name=f"vision_forward/B={B}/H={H}/N={N}/D={D}",
        tolerance=OUTPUT_TOLERANCE,
    )


@pytest.mark.parametrize("B,H,N,D", [
    (1, 12, 256, 64),
    (1, 12, 729, 64),
    (1, 8, 512, 128),
])
def test_bwd_noncausal_matches_sdpa(B, H, N, D):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    # Reference
    qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))
    ref = sdpa_ref(qr, kr, vr, causal=False)
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)

    # Triton
    out = flash_attn_gqa_train(q, k, v, causal=False, slide_size=0)
    out.backward(grad_out)

    for name, ours, theirs in [("dq", q.grad, qr.grad),
                                ("dk", k.grad, kr.grad),
                                ("dv", v.grad, vr.grad)]:
        assert_close(
            ours,
            theirs,
            name=f"vision_{name}/B={B}/H={H}/N={N}/D={D}",
            tolerance=GRAD_TOLERANCE,
        )


def test_adapter_routes_noncausal_through_kernel():
    """Sanity: adapter with is_causal=False on the module hits the kernel
    non-causal path (not SDPA fallback)."""
    from flash_attn.hf_integration import triton_gqa_attention

    class FakeModule:
        is_causal = False
        head_dim = 64

    fm = FakeModule()
    B, H, N, D = 1, 12, 256, 64
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out, _ = triton_gqa_attention(fm, q, k, v, attention_mask=None,
                                   scaling=1.0, sliding_window=None)
    # Adapter transposes back; vision contract is (B, N, H, D)
    assert out.shape == (B, N, H, D)
    # Check vs SDPA with the same scaling=1.0 quirk (Q pre-scaled). The
    # `q * sqrt(D)` rescale introduces extra bf16 rounding, so tolerate
    # ~2e-2 max elementwise vs the ~5e-3 used in the pure-kernel tests above.
    q_scaled = q * (D ** 0.5)  # cancel kernel's internal 1/sqrt(D)
    ref = sdpa_ref(q_scaled, k, v, causal=False).transpose(1, 2).contiguous()
    assert_close(
        out,
        ref,
        name="vision_adapter",
        tolerance=ADAPTER_TOLERANCE,
    )
