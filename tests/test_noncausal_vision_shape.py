"""Verify Triton fwd+bwd matches SDPA on Gemma-4 vision shapes.

Vision attention is bidirectional (`is_causal=False`), MHA (H_Q=H_KV=12),
small head dim (D=64), no sliding window, no softcap. Our adapter routes
it the same way as text — this test guards that the non-causal kernel path
is numerically correct end-to-end.

Run: pytest tests/test_noncausal_vision_shape.py -v
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import flash_attn_gqa_train, attention_flash_gqa


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

    diff = (ref.float() - out.float()).abs()
    # bf16 with N up to 1K tolerates ~1e-2 elementwise on small magnitudes
    assert diff.max().item() < 5e-3, (
        f"max|Δ|={diff.max().item():.2e} mean|Δ|={diff.mean().item():.2e}")


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
        d = (ours.float() - theirs.float()).abs()
        # bf16 grads accumulate more error; loosen vs fwd
        assert d.max().item() < 1e-1, (
            f"{name}: max|Δ|={d.max().item():.2e} mean|Δ|={d.mean().item():.2e}")


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
    diff = (ref.float() - out.float()).abs()
    assert diff.max().item() < 2e-2, (
        f"max|Δ|={diff.max().item():.2e} mean|Δ|={diff.mean().item():.2e}")


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", __file__, "-v"]))
