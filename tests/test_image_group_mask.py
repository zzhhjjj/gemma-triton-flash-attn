"""Verify Triton kernel matches an eager OR-mask reference for the Gemma-4
multimodal sliding-attention path.

Mask spec (from `transformers.models.gemma4.modeling_gemma4.create_causal_mask_mapping`):
    mask = causal_swa | (q_group == k_group & q_group >= 0)

Construct synthetic image groups inside a sequence and check fwd + bwd vs
the eager reference at MoE-realistic shapes.

Run: pytest tests/test_image_group_mask.py -v
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.attention import flash_attn_gqa_train, attention_flash_gqa
from flash_attn.hf_integration import _compute_image_group_state


def make_groups(B, N, image_spans):
    """Build mm_token_type_ids for `image_spans = [(start, length), ...]` per batch.
    Returns (mm_token_type_ids: (B, N) long).
    """
    mm = torch.zeros(B, N, dtype=torch.long, device="cuda")
    for b in range(B):
        for start, length in image_spans:
            mm[b, start:start + length] = 1  # type_id 1 = image
    return mm


def eager_or_mask_attn(q, k, v, slide_size, group_ids, scale=None):
    """Eager reference: GQA causal+SWA with image-bidirectional OR-mask.
    q: (B, H_Q, N, D), k/v: (B, H_KV, N, D), group_ids: (B, N) int.
    """
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    if H_Q != H_KV:
        r = H_Q // H_KV
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)

    if scale is None:
        scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale  # (B,H,N,N)

    idx = torch.arange(N, device=q.device)
    causal = idx[None, :] <= idx[:, None]            # (N,N)
    swa = (idx[:, None] - idx[None, :]) < slide_size  # (N,N), allowed if dist<W
    swa_causal = causal & swa                         # (N,N)

    # OR mask: same group + group >= 0 (per batch, broadcast across heads)
    g = group_ids.to(q.device)
    bidir = (g[:, :, None] == g[:, None, :]) & (g[:, :, None] >= 0)  # (B,N,N)

    valid = swa_causal[None, None, :, :] | bidir[:, None, :, :]  # (B,1,N,N) or (B,H?,N,N)

    scores = scores.masked_fill(~valid, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    # Rows fully masked (shouldn't happen in causal — q always sees self) are NaN'd to 0:
    p = torch.nan_to_num(p, nan=0.0)
    out = torch.einsum("bhqk,bhkd->bhqd", p, v.float())
    return out.to(q.dtype)


# Realistic MoE shapes: H_Q=16 H_KV=8 D=256 slide=1024, varying N and image layout.
SHAPES = [
    # (B, H_Q, H_KV, N, D, slide, image_spans)
    (1, 16, 8,  512, 256,  256, [(64, 128)]),                  # one image fits in window
    (1, 16, 8, 1024, 256,  256, [(100, 200), (600, 200)]),     # two non-overlapping images
    (2, 16, 8, 1024, 256,  256, [(50, 100), (400, 256)]),      # batch 2, image right at end
    (1, 16, 8, 2048, 256,  512, [(0, 256), (1200, 256)]),      # image at sequence start
    (1, 16, 8, 2048, 256,  256, [(800, 700)]),                 # IMAGE > SLIDE_SIZE
    (1,  8, 1, 1024, 256,  256, [(100, 200)]),                 # GQA 8:1 like E2B sliding
]


@pytest.mark.parametrize("B,H_Q,H_KV,N,D,slide,spans", SHAPES)
def test_fwd_image_group_or_mask(B, H_Q, H_KV, N, D, slide, spans):
    torch.manual_seed(0)
    q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")

    mm = make_groups(B, N, spans)
    state = _compute_image_group_state(mm)

    ref = eager_or_mask_attn(q, k, v, slide_size=slide, group_ids=state.group_ids)
    out = attention_flash_gqa(
        q, k, v, causal=True, slide_size=slide,
        group_ids=state.group_ids,
        group_lo=state.group_lo,
        group_hi_excl=state.group_hi_excl,
    )

    diff = (ref.float() - out.float()).abs()
    # bf16 ULP near 1.0 is 2^-6 = 1.56e-2; tolerate one rounding step.
    assert diff.max().item() < 2e-2 and diff.mean().item() < 5e-4, (
        f"max|Δ|={diff.max().item():.2e} mean|Δ|={diff.mean().item():.2e}"
    )


# Backward-tested subset (smaller for speed).
BWD_SHAPES = [
    (1, 16, 8,  512, 256,  256, [(64, 128)]),
    (1, 16, 8, 1024, 256,  256, [(100, 200), (600, 200)]),
    (1,  8, 1,  512, 256,  256, [(50, 200)]),
]


@pytest.mark.parametrize("B,H_Q,H_KV,N,D,slide,spans", BWD_SHAPES)
def test_bwd_image_group_or_mask(B, H_Q, H_KV, N, D, slide, spans):
    torch.manual_seed(0)
    q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    mm = make_groups(B, N, spans)
    state = _compute_image_group_state(mm)

    qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))
    ref = eager_or_mask_attn(qr, kr, vr, slide_size=slide, group_ids=state.group_ids)
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)

    out = flash_attn_gqa_train(
        q, k, v, causal=True, slide_size=slide,
        group_ids=state.group_ids,
        group_lo=state.group_lo,
        group_hi_excl=state.group_hi_excl,
    )
    out.backward(grad_out)

    for name, ours, theirs in [("dq", q.grad, qr.grad),
                                ("dk", k.grad, kr.grad),
                                ("dv", v.grad, vr.grad)]:
        d = (ours.float() - theirs.float()).abs()
        assert d.max().item() < 1.5e-1, (
            f"{name}: max|Δ|={d.max().item():.2e} mean|Δ|={d.mean().item():.2e}"
        )


def test_no_image_path_unchanged():
    """Pass all-text mm_token_type_ids: kernel must produce identical output to
    the no-group path (same numerics, no spurious extra KV iters)."""
    torch.manual_seed(0)
    B, H_Q, H_KV, N, D = 1, 16, 8, 1024, 256
    q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")

    mm = torch.zeros(B, N, dtype=torch.long, device="cuda")  # all text
    state = _compute_image_group_state(mm)

    out_no_groups = attention_flash_gqa(q, k, v, causal=True, slide_size=512)
    out_with_groups = attention_flash_gqa(
        q, k, v, causal=True, slide_size=512,
        group_ids=state.group_ids,
        group_lo=state.group_lo,
        group_hi_excl=state.group_hi_excl,
    )
    # Both paths take the same masked-loop branch when SLIDE_SIZE>0, just the
    # group path additionally OR's a constantly-False bidir mask. Result must
    # be bit-identical.
    assert torch.equal(out_no_groups, out_with_groups), \
        "all-text group_ids changed output"


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", __file__, "-v"]))
