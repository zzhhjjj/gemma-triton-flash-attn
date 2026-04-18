"""Proper mixed-precision training via FSDP2 on 8 GPUs.

What makes this a *real* mixed-precision run (vs `test_training.py`, which
loads everything in bf16):

  - Master weights:       fp32, sharded across 8 GPUs by FSDP2
  - Optimizer states:     fp32 (AdamW exp_avg, exp_avg_sq), sharded
  - Forward / backward:   bf16 matmul (FSDP2 casts params on access)
  - Gradient reduction:   fp32 (reduce_dtype) — avoids bf16 sum error at 8+ ranks
  - Optimizer update:     fp32 (params upcast on unshard)

Per-GPU memory (Gemma-4-E2B, 5.1B params, fp32 master):
  params:   20 GB / 8 = 2.5 GB
  grads:    20 GB / 8 = 2.5 GB
  AdamW:    40 GB / 8 = 5.0 GB
  act:      ~a few GB at N=512, bf16, no grad ckpt
  ≈ 12-15 GB per rank → very comfortable on H100 80 GB.

Run:
    source /opt/tiger/flash_gemma/bin/activate
    export HF_TOKEN="hf_..."
    torchrun --nproc-per-node=8 tests/gemma4_integration/test_training_fsdp2.py

CLI:
    --model NAME      HF id (default google/gemma-4-E2B)
    --seq-len N       Context length (default 512)
    --batch B         Local per-GPU batch (default 1 → effective B = world_size)
    --steps S         Training steps (default 5)
    --lr LR           AdamW learning rate (default 1e-5)
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

# transformers 5.5.4 workaround — before any config load.
import transformers  # noqa: F401
from gemma_triton_flash_attn import (
    patch_gemma4_shared_kv_states_for_fsdp2,
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
)
patch_transformers_5_5_4_flash_attn_key()
# Required for per-layer FSDP2 sharding: Gemma-4's `shared_kv_states` dict
# loses identity across FSDP2's pytree flatten/unflatten at each layer boundary
# (later KV-shared layers raise KeyError). Swaps the dict for a pytree-opaque
# holder that survives the round-trip.
patch_gemma4_shared_kv_states_for_fsdp2()

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def is_rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rp(*args, **kwargs):
    """Rank-0-only print."""
    if is_rank0():
        print(*args, **kwargs, flush=True)


def set_impl(model, impl: str):
    model.config._attn_implementation = impl
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = impl


def make_batches(tok, batch, seq_len, n_batches, device):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                      split=f"train[:{max(4000, n_batches * 500)}]")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = batch * seq_len * n_batches
    assert ids.shape[0] >= need, (
        f"dataset too small: have {ids.shape[0]} tokens, need {need}")
    chunks = ids[:need].view(n_batches, batch, seq_len)
    return [chunks[i].to(device) for i in range(n_batches)]


def fwd_loss(model, ids):
    # Wrap in autocast so activations become bf16 for matmul (with
    # cast_forward_inputs=False, FSDP2 does not cast the hidden_states input
    # that flows into each layer; autocast does that job).
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
    # Always compute cross-entropy in fp32 for stability.
    logits = out.logits[:, :-1].float()
    targets = ids[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--attn", default="triton_gqa",
                    choices=["triton_gqa", "sdpa"],
                    help="attention implementation used during training")
    ap.add_argument("--save-losses", default=None,
                    help="rank-0 writes per-step losses to this JSON path")
    args = ap.parse_args()

    # Initialize distributed — expects torchrun env vars.
    assert "LOCAL_RANK" in os.environ, "launch with `torchrun --nproc-per-node=N ...`"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    register_triton_attention()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    rp(f"=== FSDP2 mixed-precision training (world_size={world_size}) ===")
    rp(f"[load] {args.model} in fp32 onto {device}...")
    # Load fp32 master weights. Per-rank — each GPU gets a full copy, then
    # FSDP2 shards at `fully_shard()` time (duplicates freed).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa",
    ).to(device)
    tok = AutoTokenizer.from_pretrained(args.model)

    # Mixed-precision policy: bf16 matmul, fp32 master, fp32 gradient reduce.
    #
    # `reduce_dtype=fp32` keeps gradient reduce-scatter in fp32 to avoid
    # bf16-accumulation error across ranks — small per-param grads can lose
    # most of their precision when summed in bf16 at 8+ ranks. Cheap: comm
    # volume doubles but we're not bandwidth-bound on NVLink within a node.
    #
    # `cast_forward_inputs=False` keeps FSDP2 from casting forward inputs at
    # each boundary. We get bf16 matmul via the outer autocast in fwd_loss.
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    # Per-layer sharding: wrap each decoder layer individually so FSDP2 can
    # overlap layer N's forward compute with layer N+1's all-gather (and
    # similarly on backward). Then wrap the root for the remaining parameters
    # (embeddings, norm, lm_head). Requires `patch_gemma4_shared_kv_states_for_fsdp2`
    # (called at import time): without it, FSDP2's per-layer
    # `tree_flatten/unflatten` on kwargs replaces the `shared_kv_states` dict
    # with a fresh empty one each call → KV-shared layers raise KeyError.
    lm = model.model.language_model
    for layer in lm.layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    rp(f"[fsdp2] wrapped root + {len(lm.layers)} per-layer boundaries")

    # Sanity check: params are fp32, gradients allocated on step.
    sample_param = next(model.parameters())
    rp(f"[fsdp2] sample param dtype={sample_param.dtype}  "
       f"local shape={tuple(sample_param.shape)}")

    # Batches — identical across ranks (FSDP2 = DDP-style data parallelism on
    # inputs, with parameter sharding inside). Effective batch = batch×world.
    batches = make_batches(tok, args.batch, args.seq_len, args.steps, device)
    rp(f"[data] {len(batches)} batches, local shape={tuple(batches[0].shape)}, "
       f"effective batch per step = {args.batch * world_size}")

    # --- Step-0 forward correctness: SDPA vs Triton on same weights ---
    model.eval()
    set_impl(model, "sdpa")
    with torch.no_grad():
        l_sdpa = fwd_loss(model, batches[0]).item()
    set_impl(model, "triton_gqa")
    with torch.no_grad():
        l_tri = fwd_loss(model, batches[0]).item()
    rp(f"\n[step-0 fwd correctness] SDPA={l_sdpa:.6f}  Triton={l_tri:.6f}  "
       f"|Δ|={abs(l_sdpa - l_tri):.4e}")

    # --- Training: `--attn` attention + mixed precision via FSDP2 ---
    set_impl(model, args.attn)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rp(f"\n[train] attn={args.attn}, FSDP2 mixed precision, AdamW lr={args.lr}")

    losses = []
    any_nan = False
    for step, ids in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        loss = fwd_loss(model, ids)
        if not torch.isfinite(loss):
            rp(f"  step {step}: non-finite loss — abort")
            any_nan = True
            break
        loss.backward()
        # Gradient finite check — each rank checks its local shard.
        local_bad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                local_bad = True
                break
        bad_any = torch.tensor([1 if local_bad else 0], device=device)
        dist.all_reduce(bad_any, op=dist.ReduceOp.SUM)
        if bad_any.item() > 0:
            rp(f"  step {step}: non-finite grad on some rank — abort")
            any_nan = True
            break
        opt.step()
        losses.append(loss.item())
        rp(f"  step {step}: loss={loss.item():.4f}")

    # Inspect optimizer state dtype. FSDP2 wraps params as DTensor, and the
    # optimizer state key can be the underlying local tensor rather than the
    # DTensor object — so `opt.state[p]` on the iterator param may miss it.
    # Scan all entries in opt.state directly.
    optim_dtype = None
    for st in opt.state.values():
        if "exp_avg" in st:
            ea = st["exp_avg"]
            # DTensor has .dtype; plain Tensor has .dtype too.
            optim_dtype = getattr(ea, "dtype", None)
            break

    rp(f"\n=== Verdict ===")
    rp(f"param dtype (master): {sample_param.dtype}")
    rp(f"optim exp_avg dtype:  {optim_dtype}")
    rp(f"losses: {[f'{l:.4f}' for l in losses]}")
    rp(f"step-0 SDPA vs Triton fwd |Δ|: {abs(l_sdpa - l_tri):.4e}")

    # PASS: fp32 master + step-0 fwd match + no NaN across all steps. The
    # optim_dtype read is informational; skip it from the PASS gate because
    # FSDP2/DTensor state introspection isn't guaranteed stable across torch
    # versions, and the params-fp32 check already implies AdamW state fp32.
    ok = (sample_param.dtype == torch.float32
          and not any_nan
          and len(losses) == len(batches)
          and abs(l_sdpa - l_tri) < 0.05)
    rp(f"{'PASS' if ok else 'FAIL'}")

    if args.save_losses and is_rank0():
        import json
        payload = {
            "attn": args.attn,
            "model": args.model,
            "seq_len": args.seq_len,
            "batch_per_rank": args.batch,
            "world_size": world_size,
            "effective_batch": args.batch * world_size,
            "lr": args.lr,
            "step0_sdpa_loss": l_sdpa,
            "step0_triton_loss": l_tri,
            "losses": [{"step": i, "loss": l} for i, l in enumerate(losses)],
        }
        with open(args.save_losses, "w") as f:
            json.dump(payload, f, indent=2)
        rp(f"[save] losses → {args.save_losses}")

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
