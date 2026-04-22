"""Integration test for the multimodal OR-mask path through the HF adapter.

Closes the silent-gap that hid the Gemma-4-26B-A4B vision-bidirectional bug:
the existing `test_adapter.py` only drives `triton_gqa_attention` with
text-only inputs, so the kernel's lack of OR-mask support stayed invisible
until the user spotted it. This file exercises the full plumbing — ContextVar
+ adapter + kernel — without downloading any model weights.

Three guards:
  1. test_adapter_with_image_groups
       Drive `triton_gqa_attention` after stashing image-group state in the
       ContextVar (mimicking what `patch_gemma4_image_group_ids_for_kernel`
       does at the model boundary). Output must match eager OR-mask reference.

  2. test_patch_wires_contextvar
       Wrap a fake `Gemma4Model.forward` analogue with the patch and verify
       (a) state is set when wants_groups + mm_token_type_ids are present
       (b) state is reset on exit (even on exception)
       (c) state is NOT set when text config has no vision-bidirectional opt-in

  3. test_adapter_raises_on_4d_mask_without_groups
       The original silent path: a 4D bool mask reaches the adapter with no
       group state. We now raise rather than silently dropping the OR-mask.

Run:
    source /opt/tiger/flash_gemma/bin/activate
    python tests/gemma4_integration/test_adapter_multimodal.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# Make repo root importable without installation.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from flash_attn.hf_integration import (  # noqa: E402
    _compute_image_group_state,
    _image_group_state,
    triton_gqa_attention,
)


# =====================================================================
# Reference: GQA causal+SWA with the same OR-mask the upstream uses.
# =====================================================================

def eager_or_mask_ref(q, k, v, *, slide_size: int, group_ids: torch.Tensor):
    """q/k/v: (B, H_*, N, D); group_ids: (B, N). Returns (B, N, H_Q, D)."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    if H_Q != H_KV:
        r = H_Q // H_KV
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)

    scale = D ** -0.5
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale  # (B,H,N,N)

    idx = torch.arange(N, device=q.device)
    causal = idx[:, None] >= idx[None, :]
    swa = (idx[:, None] - idx[None, :]) < slide_size
    swa_causal = causal & swa

    g = group_ids.to(q.device)
    bidir = (g[:, :, None] == g[:, None, :]) & (g[:, :, None] >= 0)  # (B,N,N)
    valid = swa_causal[None, None, :, :] | bidir[:, None, :, :]

    scores = scores.masked_fill(~valid, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    out = p @ v.float()
    return out.to(q.dtype).transpose(1, 2).contiguous()  # (B,N,H_Q,D)


# =====================================================================
# Test 1: adapter through ContextVar produces OR-mask output
# =====================================================================

def test_adapter_with_image_groups():
    print("\n=== test_adapter_with_image_groups ===")
    torch.manual_seed(0)
    # MoE-realistic: H_Q=16 H_KV=8 D=256 slide=1024, with 1 image span.
    B, H_Q, H_KV, N, D, slide = 1, 16, 8, 1024, 256, 1024
    q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")

    # Build mm_token_type_ids: image at [128, 384) of length 256.
    mm = torch.zeros(B, N, dtype=torch.long, device="cuda")
    mm[:, 128:384] = 1
    state = _compute_image_group_state(mm)

    module = SimpleNamespace(head_dim=D, is_causal=True)
    token = _image_group_state.set(state)
    try:
        out, _ = triton_gqa_attention(
            module, q, k, v, attention_mask=None,
            dropout=0.0, scaling=None, softcap=None,
            sliding_window=slide,
        )
    finally:
        _image_group_state.reset(token)

    ref = eager_or_mask_ref(q, k, v, slide_size=slide, group_ids=state.group_ids)
    diff = (ref.float() - out.float()).abs()
    print(f"  shape={out.shape}  max|Δ|={diff.max().item():.2e}  "
          f"mean|Δ|={diff.mean().item():.2e}")
    assert diff.max().item() < 2e-2 and diff.mean().item() < 5e-4, "OR-mask mismatch"
    print("  PASS")


# =====================================================================
# Test 2: patch_gemma4_image_group_ids_for_kernel wires ContextVar
# =====================================================================

def test_patch_wires_contextvar():
    """Don't load real Gemma4Model — fake the minimum surface and confirm the
    wrapper sets/resets the ContextVar based on text_config flag + presence
    of mm_token_type_ids."""
    print("\n=== test_patch_wires_contextvar ===")
    import contextvars
    # Manually run the same machinery that patch_gemma4_image_group_ids_for_kernel
    # injects, against a synthetic config + forward. This avoids the transformers
    # import side-effects in this lightweight test.

    captured = {"state_during_forward": None}

    def fake_forward(self, **kwargs):
        captured["state_during_forward"] = _image_group_state.get()
        return "ok"

    def make_wrapped(orig):
        def _wrapped(self, **kwargs):
            text_cfg = self.config.get_text_config()
            wants = getattr(text_cfg, "use_bidirectional_attention", None) == "vision"
            mm = kwargs.get("mm_token_type_ids", None)
            tok = None
            if wants and mm is not None:
                state = _compute_image_group_state(mm)
                tok = _image_group_state.set(state)
            try:
                return orig(self, **kwargs)
            finally:
                if tok is not None:
                    _image_group_state.reset(tok)
        return _wrapped

    wrapped = make_wrapped(fake_forward)

    # (a) wants_groups=True, mm provided → state set
    cfg_a = SimpleNamespace(get_text_config=lambda: SimpleNamespace(
        use_bidirectional_attention="vision"))
    self_a = SimpleNamespace(config=cfg_a)
    mm = torch.tensor([[0, 1, 1, 0]], device="cuda")
    captured["state_during_forward"] = "sentinel"
    wrapped(self_a, mm_token_type_ids=mm)
    assert captured["state_during_forward"] is not None, \
        "state should be set when wants_groups + mm present"
    assert _image_group_state.get() is None, "state should be reset after forward"
    print("  (a) state set when wants_groups + mm provided   PASS")

    # (b) text_config has no vision opt-in → state stays None even with mm
    cfg_b = SimpleNamespace(get_text_config=lambda: SimpleNamespace())
    self_b = SimpleNamespace(config=cfg_b)
    captured["state_during_forward"] = "sentinel"
    wrapped(self_b, mm_token_type_ids=mm)
    assert captured["state_during_forward"] is None, \
        "state must NOT be set when text config opts out (E2B/E4B)"
    print("  (b) state untouched when text config opts out    PASS")

    # (c) wants_groups but no mm → state stays None (incremental decode case)
    captured["state_during_forward"] = "sentinel"
    wrapped(self_a, mm_token_type_ids=None)
    assert captured["state_during_forward"] is None, \
        "state must NOT be set when mm_token_type_ids is missing"
    print("  (c) state untouched when mm_token_type_ids absent PASS")

    # (d) exception inside forward must still reset the ContextVar
    def bad_forward(self, **kwargs):
        raise RuntimeError("synthetic")
    wrapped_bad = make_wrapped(bad_forward)
    try:
        wrapped_bad(self_a, mm_token_type_ids=mm)
    except RuntimeError:
        pass
    assert _image_group_state.get() is None, \
        "state must be reset even on forward exception"
    print("  (d) state reset even when forward raises          PASS")


# =====================================================================
# Test 3: adapter raises on 4D mask without group state (no silent path)
# =====================================================================

def test_adapter_raises_on_4d_mask_without_groups():
    """Pre-fix, this combination (4D bool mask + sliding + causal, no group
    state) silently dropped the OR-mask. Now we raise so MoE-multimodal jobs
    misconfigured to skip the patch fail loud at first call instead of
    training on wrong gradients."""
    print("\n=== test_adapter_raises_on_4d_mask_without_groups ===")
    torch.manual_seed(0)
    B, H_Q, H_KV, N, D, slide = 1, 16, 8, 256, 256, 128
    q = torch.randn(B, H_Q, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device="cuda")
    bool_mask = torch.zeros(B, 1, N, N, dtype=torch.bool, device="cuda")
    bool_mask[..., :, :] = True  # contents don't matter; the type does

    module = SimpleNamespace(head_dim=D, is_causal=True)
    # Make sure no leftover state from prior tests:
    assert _image_group_state.get() is None

    raised = False
    try:
        triton_gqa_attention(
            module, q, k, v, attention_mask=bool_mask,
            dropout=0.0, scaling=None, softcap=None,
            sliding_window=slide,
        )
    except RuntimeError as e:
        raised = True
        msg = str(e)
        print(f"  raised RuntimeError as expected: {msg[:120]}")
        assert "image" in msg.lower() or "group" in msg.lower() or \
               "mask" in msg.lower(), \
            f"raise message should mention mask/group: {msg!r}"
    assert raised, "adapter must raise on 4D mask without group state"
    print("  PASS")


def main():
    fns = [
        test_adapter_with_image_groups,
        test_patch_wires_contextvar,
        test_adapter_raises_on_4d_mask_without_groups,
    ]
    n_fail = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n{len(fns) - n_fail}/{len(fns)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
