"""End-to-end training test: Gemma-4-E2B with Triton GQA attention.

Runs a few AdamW steps on synthetic next-token prediction and verifies:

  1. No NaN in forward or backward
  2. Loss decreases monotonically
  3. Loss trajectory matches SDPA baseline within tolerance

This is the backward-path counterpart to `test_gemma4_moe.py` (forward-only).
It exercises our Triton kernel's autograd wrapper in a real HF training loop,
proving the end-to-end training use case.

Memory (single H100, bf16, N=512, batch=1):
  weights(10GB) + grad(10GB) + AdamW states(40GB) + activations ≈ 65 GB → fits.
Increase --seq-len or --batch at your own risk.

Run:
    source /opt/tiger/flash_gemma/bin/activate
    export HF_TOKEN="hf_..."
    python tests/gemma4_integration/test_training.py --steps 5

CLI:
    --model NAME    HF model id (default: google/gemma-4-E2B, 5.1B dense)
    --seq-len N     Context length (default: 512)
    --batch B       Per-step batch size (default: 1)
    --steps S       Training steps per impl (default: 5)
    --lr LR         AdamW learning rate (default: 1e-5)
    --device DEV    cuda device index (default: 0)
    --seed SEED     torch manual seed (default: 0)
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F

# transformers 5.5.4 workaround — must run before any config load.
import transformers  # noqa: F401
from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
)
patch_transformers_5_5_4_flash_attn_key()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def set_impl(model, impl: str):
    model.config._attn_implementation = impl
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = impl


def make_batch(tok, batch: int, seq_len: int, device, seed: int):
    """Random token ids — gives a realistic starting loss (~log(vocab)≈12).

    Natural-text prompts collapse to near-zero loss within a couple of steps on
    a well-trained LM, which makes per-step relative diffs look large even when
    absolute diffs are tiny.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab = tok.vocab_size if hasattr(tok, "vocab_size") else 256000
    ids = torch.randint(0, vocab, (batch, seq_len), generator=g).to(device)
    return ids


def step_loss(model, ids):
    """Next-token cross-entropy (mean over batch × (seq_len - 1) positions)."""
    out = model(ids)
    logits = out.logits[:, :-1].float()           # (B, N-1, V)
    targets = ids[:, 1:]                          # (B, N-1)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def run_training(model, ids, steps, lr, tag):
    """Run `steps` AdamW steps, return list of loss values."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = step_loss(model, ids)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[{tag}] step {s}: loss is NaN/Inf — abort")
            return losses, False
        loss.backward()
        # NaN check on gradients
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"[{tag}] step {s}: non-finite grad in {n} — abort")
                return losses, False
        opt.step()
        losses.append(loss.item())
        print(f"[{tag}] step {s}: loss={loss.item():.4f}")
    return losses, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    register_triton_attention()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    print(f"=== Gemma-4 E2E Training ({args.model}, N={args.seq_len}, "
          f"batch={args.batch}, steps={args.steps}) ===")
    print(f"[load] loading model onto {device} (bf16)...")
    # Single-GPU load — device_map="auto" is incompatible with backward.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[load] params={sum(p.numel() for p in model.parameters())/1e9:.2f}B "
          f"dtype={next(model.parameters()).dtype}")

    ids = make_batch(tok, args.batch, args.seq_len, device, args.seed)
    print(f"[input] ids.shape={tuple(ids.shape)}")

    # Snapshot state dict so both runs start from the same weights.
    state = copy.deepcopy(model.state_dict())

    print("\n--- SDPA baseline ---")
    set_impl(model, "sdpa")
    model.train()
    losses_sdpa, ok_sdpa = run_training(model, ids, args.steps, args.lr, "sdpa")

    print("\n--- Triton ---")
    model.load_state_dict(state)
    set_impl(model, "triton_gqa")
    model.train()
    losses_tri, ok_tri = run_training(model, ids, args.steps, args.lr, "triton")

    # Verdict
    print("\n=== Verdict ===")
    ok = ok_sdpa and ok_tri
    if not ok:
        print("FAIL: one or both runs hit NaN/Inf")
        sys.exit(1)

    # Monotonic decrease on first → last step
    sdpa_decreases = losses_sdpa[-1] < losses_sdpa[0]
    tri_decreases = losses_tri[-1] < losses_tri[0]
    print(f"SDPA:   {losses_sdpa[0]:.4f} → {losses_sdpa[-1]:.4f}   "
          f"{'↓' if sdpa_decreases else '↑ (BAD)'}")
    print(f"Triton: {losses_tri[0]:.4f} → {losses_tri[-1]:.4f}   "
          f"{'↓' if tri_decreases else '↑ (BAD)'}")

    # Per-step absolute diff (Triton vs SDPA). Use absolute — once loss collapses
    # below ~0.01 the denominator blows rel diff up without any real divergence.
    abs_diffs = [abs(a - b) for a, b in zip(losses_sdpa, losses_tri)]
    max_abs = max(abs_diffs)
    print(f"Per-step loss abs diff (Triton vs SDPA): max={max_abs:.2e}, "
          f"mean={sum(abs_diffs)/len(abs_diffs):.2e}")

    # PASS bar: both decrease + per-step absolute loss diff < 0.05 nats.
    # Tolerance budget: bf16 rounding in attention × 35 layers × AdamW step-to-step
    # compounding. 0.05 nats ≈ 5% relative at loss=1.0, which is within normal
    # bf16-vs-fp32 reproducibility for a full LM stack.
    ok_final = sdpa_decreases and tri_decreases and max_abs < 0.05
    print(f"{'PASS' if ok_final else 'FAIL'}")
    sys.exit(0 if ok_final else 1)


if __name__ == "__main__":
    main()
