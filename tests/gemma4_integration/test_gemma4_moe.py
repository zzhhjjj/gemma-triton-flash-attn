"""End-to-end integration test: Gemma-4-26B-A4B (MoE) with Triton GQA attention.

Loads the 26B MoE model sharded across multiple GPUs (device_map="auto"),
swaps attention → Triton kernel, and verifies that last-token logits match
SDPA within tolerance. Also measures forward throughput.

MoE attention shapes (from real config.json, per-layer):
  sliding: H_Q=16, H_KV=8, head_dim=256, slide=1024   (24 of 30 layers)
  full:    H_Q=16, H_KV=2, head_dim=512, slide=0      ( 6 of 30 layers)

Full-sequence logits are (1, N, 262K) ≈ 1 GB at N=1K and OOM on top of the
~52 GB MoE weights, so we only compare the last-token row (<1 MB).

Run (gated model — need HF_TOKEN):
    source /opt/tiger/flash_gemma/bin/activate
    export HF_TOKEN="hf_..."
    # Uses all visible GPUs by default; restrict with CUDA_VISIBLE_DEVICES if needed.
    python tests/gemma4_integration/test_gemma4_moe.py --seq-len 1024

CLI:
    --model NAME        HF model id (default: google/gemma-4-26B-A4B)
    --seq-len N         Correctness seq len (default: 1024)
    --bench-seqs N,...  Comma-separated throughput seq lens (default: 512,1024,2048)
    --skip-perf         Correctness only
    --reserve-gb X      Leave X GB per GPU for activations (default: 10)
"""
import argparse
import os
import sys

import torch

# transformers 5.5.4 `PACKAGE_DISTRIBUTION_MAPPING['flash_attn']` KeyError —
# must run before any config load.
import transformers  # noqa: F401
from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
)
patch_transformers_5_5_4_flash_attn_key()

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def set_impl(model, impl: str):
    model.config._attn_implementation = impl
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = impl


def load_model(model_id: str, reserve_gb: float, dtype=torch.bfloat16):
    """Shard the MoE model across all visible GPUs, leaving reserve_gb headroom."""
    print(f"[load] Loading config for {model_id}...")
    cfg = AutoConfig.from_pretrained(model_id)
    tcfg = getattr(cfg, "text_config", cfg)
    print(f"[load] type={cfg.model_type}  hidden={getattr(tcfg,'hidden_size','?')}  "
          f"layers={getattr(tcfg,'num_hidden_layers','?')}  "
          f"H_Q={getattr(tcfg,'num_attention_heads','?')}  "
          f"H_KV={getattr(tcfg,'num_key_value_heads','?')}  "
          f"H_KV_global={getattr(tcfg,'num_global_key_value_heads','?')}  "
          f"head_dim={getattr(tcfg,'head_dim','?')}  "
          f"global_head_dim={getattr(tcfg,'global_head_dim','?')}  "
          f"slide={getattr(tcfg,'sliding_window','?')}")
    if hasattr(tcfg, "num_experts"):
        print(f"[load] MoE: num_experts={tcfg.num_experts}  "
              f"top_k={getattr(tcfg,'top_k_experts',getattr(tcfg,'num_experts_per_tok','?'))}  "
              f"moe_intermediate={getattr(tcfg,'moe_intermediate_size','?')}")

    # Build per-GPU memory budget. We reserve `reserve_gb` on every visible
    # GPU for activations, KV cache, tensor parallel temporaries, etc.
    n_gpus = torch.cuda.device_count()
    max_memory = {}
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1e9
        budget_gb = max(1.0, total_gb - reserve_gb)
        max_memory[i] = f"{int(budget_gb)}GiB"
    print(f"[load] sharding across {n_gpus} GPUs (budget per GPU: "
          f"{list(max_memory.values())[0]} with {reserve_gb:.0f} GB reserved)")

    print(f"[load] Loading weights ({dtype}) — this may take a minute...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto",
        max_memory=max_memory, attn_implementation="sdpa",
    )
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tok = None

    # Where does the embedding / lm_head live? That's where we must place inputs.
    try:
        input_device = next(model.get_input_embeddings().parameters()).device
    except Exception:
        input_device = torch.device("cuda:0")
    print(f"[load] input embedding device: {input_device}")
    return model, tok, input_device


def make_ids(tok, N, device):
    if tok is not None:
        prompt = "Hello, world! " * max(1, N // 4)
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] >= N:
            ids = ids[:, :N]
        else:
            pad = torch.zeros(1, N - ids.shape[1], dtype=ids.dtype, device=device)
            ids = torch.cat([ids, pad], dim=1)
        return ids
    return torch.randint(0, 1000, (1, N), device=device)


def last_token_logits(model, ids):
    """Run forward, return last-token logits as fp32 on CPU (small, safe to keep)."""
    with torch.no_grad():
        # ids goes in on the embedding device; lm_head may be on another GPU.
        out = model(ids).logits[:, -1].float().cpu()
    return out


def test_correctness(model, ids):
    """Compare last-token logits under SDPA vs Triton — same weights, same input."""
    set_impl(model, "sdpa")
    o_sdpa = last_token_logits(model, ids)
    set_impl(model, "triton_gqa")
    o_tri = last_token_logits(model, ids)

    cos = torch.nn.functional.cosine_similarity(
        o_sdpa.flatten(), o_tri.flatten(), dim=0).item()
    max_abs = (o_sdpa - o_tri).abs().max().item()
    argmax_sdpa = o_sdpa.argmax(-1)
    argmax_tri = o_tri.argmax(-1)
    top1 = (argmax_sdpa == argmax_tri).float().mean().item()
    top5_sdpa = set(o_sdpa[0].topk(5).indices.tolist())
    top5_tri = set(o_tri[0].topk(5).indices.tolist())
    top5_overlap = len(top5_sdpa & top5_tri)

    print(f"[correct] last-token cos_sim = {cos:.6f}")
    print(f"[correct] last-token max|Δ|  = {max_abs:.4e}")
    print(f"[correct] top-1 match        = {top1*100:.1f}%")
    print(f"[correct] top-5 overlap      = {top5_overlap}/5")
    ok = cos > 0.999 and top1 >= 0.95
    print(f"[correct] {'PASS' if ok else 'FAIL'}")
    return ok


def bench_throughput(model, tok, input_device, seq_lens, reps=5, warmup=2):
    """Time forward under SDPA vs Triton at each seq_len."""
    print(f"\n=== Throughput ===")
    print(f"{'N':>6} {'sdpa (ms)':>12} {'triton (ms)':>13} {'speedup':>9}")
    print("-" * 44)
    for N in seq_lens:
        ids = make_ids(tok, N, input_device)
        times = {}
        for impl in ("sdpa", "triton_gqa"):
            set_impl(model, impl)
            with torch.no_grad():
                for _ in range(warmup):
                    model(ids)
            torch.cuda.synchronize()
            samples = []
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            for _ in range(reps):
                s.record()
                with torch.no_grad():
                    model(ids)
                e.record()
                torch.cuda.synchronize()
                samples.append(s.elapsed_time(e))
            samples.sort()
            times[impl] = samples[len(samples) // 2]
        sp = times["sdpa"] / times["triton_gqa"]
        print(f"{N:>6} {times['sdpa']:>12.2f} {times['triton_gqa']:>13.2f} {sp:>8.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-26B-A4B")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--bench-seqs", default="512,1024,2048")
    p.add_argument("--skip-perf", action="store_true")
    p.add_argument("--reserve-gb", type=float, default=10.0,
                   help="GB to reserve per GPU for activations / KV cache")
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    register_triton_attention()

    print(f"\n=== Gemma-4 MoE E2E Correctness ({args.model}, N={args.seq_len}) ===")
    model, tok, input_device = load_model(args.model, reserve_gb=args.reserve_gb)
    model.eval()

    ids = make_ids(tok, args.seq_len, input_device)
    print(f"[input] shape={ids.shape}  device={ids.device}")

    ok = test_correctness(model, ids)

    if ok and not args.skip_perf:
        seqs = [int(x) for x in args.bench_seqs.split(",") if x.strip()]
        bench_throughput(model, tok, input_device, seqs)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
