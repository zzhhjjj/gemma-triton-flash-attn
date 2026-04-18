"""End-to-end MoE bottleneck profiler.

Loads `google/gemma-4-26B-A4B` (or --model), swaps attention → Triton GQA,
runs a forward pass at configurable seq_len, and reports:

  1. Correctness vs SDPA (logits cos_sim, top-1 match)
  2. Overall speedup (Triton vs SDPA)
  3. **Layer-level breakdown** via `torch.profiler` — attention vs MoE router
     vs expert MLPs. This is what tells us whether attention optimization is
     worth pursuing or whether the bottleneck has moved.

Output: pretty-printed top-N CUDA-time consumers in each phase.

Run (gated model — need HF_TOKEN):
    source /opt/tiger/flash_gemma/bin/activate
    export HF_TOKEN="hf_..."
    python tests/gemma4_integration/test_gemma4_moe_profile.py --seq-len 1024
"""
import argparse
import os
import sys

import torch

# transformers 5.5.4 workaround
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


def load_model(model_id: str, dtype=torch.bfloat16):
    print(f"[load] Loading config for {model_id}...")
    cfg = AutoConfig.from_pretrained(model_id)
    tcfg = getattr(cfg, "text_config", cfg)
    print(f"[load] type={cfg.model_type}, hidden={getattr(tcfg,'hidden_size','?')}, "
          f"layers={getattr(tcfg,'num_hidden_layers','?')}, "
          f"heads={getattr(tcfg,'num_attention_heads','?')}/"
          f"kv={getattr(tcfg,'num_key_value_heads','?')}, "
          f"head_dim={getattr(tcfg,'head_dim','?')}, "
          f"slide={getattr(tcfg,'sliding_window','?')}")
    if hasattr(tcfg, 'num_experts'):
        print(f"[load] MoE: num_experts={tcfg.num_experts}, "
              f"top_k={getattr(tcfg,'top_k_experts',getattr(tcfg,'num_experts_per_tok','?'))}, "
              f"moe_intermediate={getattr(tcfg,'moe_intermediate_size','?')}")
    print(f"[load] Loading weights (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="cuda", attn_implementation="sdpa"
    )
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tok = None
    return model, tok


def make_ids(tok, N):
    if tok is not None:
        prompt = "Hello, world! " * max(1, N // 4)
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        if ids.shape[1] >= N: ids = ids[:, :N]
        else:
            pad = torch.zeros(1, N - ids.shape[1], dtype=ids.dtype, device="cuda")
            ids = torch.cat([ids, pad], dim=1)
        return ids
    return torch.randint(0, 1000, (1, N), device="cuda")


def correctness(model, ids):
    """Forward once with SDPA, once with Triton; compare last-token logits only.

    Full-sequence logits are (1, N, V≈262K) — at N=8K that's 8 GB per copy and
    OOMs alongside the 52 GB MoE weights. Last-token logits are <1 MB."""
    set_impl(model, "sdpa")
    with torch.no_grad():
        o_sdpa = model(ids).logits[:, -1].float().clone()
    set_impl(model, "triton_gqa")
    with torch.no_grad():
        o_tri = model(ids).logits[:, -1].float().clone()
    cos = torch.nn.functional.cosine_similarity(
        o_sdpa.flatten(), o_tri.flatten(), dim=0).item()
    top1 = (o_sdpa.argmax(-1) == o_tri.argmax(-1)).float().mean().item()
    print(f"[correct] last-token cos_sim={cos:.6f}, top-1={top1*100:.1f}%")
    return cos > 0.999 and top1 >= 0.95


def time_fwd(model, ids, impl, reps=5, warmup=2):
    set_impl(model, impl)
    with torch.no_grad():
        for _ in range(warmup):
            model(ids)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        s.record()
        with torch.no_grad():
            model(ids)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def profile_fwd(model, ids, impl, rank_by="device_time_total"):
    """Run torch.profiler once, return top-20 aten/cuda ops."""
    from torch.profiler import profile, ProfilerActivity
    set_impl(model, impl)
    # Warmup
    with torch.no_grad():
        model(ids); model(ids)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 record_shapes=False) as prof:
        with torch.no_grad():
            model(ids)
    torch.cuda.synchronize()

    key_avgs = prof.key_averages()
    sorted_events = sorted(key_avgs, key=lambda e: getattr(e, rank_by), reverse=True)
    return sorted_events


def bucket_events(events):
    """Classify CUDA ops into coarse buckets.

    Primary goal: separate **Attention** from **Everything else**. If attention
    is already a small fraction of total, further kernel optimisation has
    diminishing returns regardless of how much faster we make it.
    """
    buckets = {
        "Attention (flash_attn kernel / SDPA)": 0.0,
        "MoE dispatch (topk / scatter / gather)": 0.0,
        "Matmul (MLP / proj / experts / lm_head)": 0.0,
        "Norm / RoPE / activation / elementwise": 0.0,
        "Memcpy / layout / other": 0.0,
    }
    # Substring-based classifier. Order matters: attention is checked first so
    # nested ops inside the attention code path (e.g. "aten::softmax" inside
    # SDPA) don't leak into elementwise bucket.
    ATTN_NEEDLES = ("flash_attn", "gqa_kernel", "scaled_dot_product",
                    "attention", "sdpa_math", "triton_gqa")
    DISPATCH_NEEDLES = ("topk", "one_hot", "scatter", "gather",
                        "index_select", "index_add", "nonzero")
    MATMUL_NEEDLES = ("bmm", "matmul", "addmm", "linear", "gemm",
                      "_mm_out", "mm_batched", "::mm")
    ELEMENTWISE_NEEDLES = ("norm", "rms", "silu", "gelu", "softmax",
                           "rope", "sin", "cos", "mul", "add", "div",
                           "sub", "exp", "log", "pow", "sqrt", "rsqrt",
                           "cast", "to_copy", "sigmoid", "tanh")
    LAYOUT_NEEDLES = ("memcpy", "copy", "view", "permute", "transpose",
                      "contiguous", "reshape", "cat", "split", "stack",
                      "empty", "zeros", "ones", "fill")

    for e in events:
        t = e.device_time_total / 1000  # μs → ms
        if t <= 0: continue
        name = e.key.lower()
        if any(n in name for n in ATTN_NEEDLES):
            buckets["Attention (flash_attn kernel / SDPA)"] += t
        elif any(n in name for n in DISPATCH_NEEDLES):
            buckets["MoE dispatch (topk / scatter / gather)"] += t
        elif any(n in name for n in MATMUL_NEEDLES):
            buckets["Matmul (MLP / proj / experts / lm_head)"] += t
        elif any(n in name for n in ELEMENTWISE_NEEDLES):
            buckets["Norm / RoPE / activation / elementwise"] += t
        else:
            buckets["Memcpy / layout / other"] += t
    return buckets


def print_top_events(events, impl, n=25):
    print(f"\n  Top-{n} CUDA ops under impl={impl}:")
    print(f"  {'op':<70} {'CUDA ms':>10} {'% of total':>10}")
    total = sum(e.device_time_total for e in events)
    for e in events[:n]:
        pct = 100 * e.device_time_total / total if total else 0
        t = e.device_time_total / 1000
        print(f"  {e.key[:68]:<70} {t:>10.2f} {pct:>9.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-26B-A4B")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--skip-sdpa", action="store_true",
                   help="Skip SDPA baseline (saves one forward pass if you only want Triton profile)")
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    register_triton_attention()

    print(f"\n=== Gemma-4 MoE E2E Profile ({args.model}, N={args.seq_len}) ===")
    model, tok = load_model(args.model)
    model.eval()
    ids = make_ids(tok, args.seq_len)
    print(f"[input] shape={ids.shape}")

    ok = correctness(model, ids)
    print(f"[correct] {'PASS' if ok else 'FAIL'}")

    # End-to-end timing
    t_tri = time_fwd(model, ids, "triton_gqa")
    if not args.skip_sdpa:
        t_sdpa = time_fwd(model, ids, "sdpa")
        print(f"\n[e2e] SDPA      fwd = {t_sdpa:>7.2f} ms")
    else:
        t_sdpa = None
    print(f"[e2e] Triton    fwd = {t_tri:>7.2f} ms")
    if t_sdpa: print(f"[e2e] speedup       = {t_sdpa/t_tri:>7.2f}x")

    # Profile Triton
    print(f"\n--- Profile under triton_gqa ---")
    ev_tri = profile_fwd(model, ids, "triton_gqa")
    b_tri = bucket_events(ev_tri)
    total_tri = sum(b_tri.values())
    print(f"  Bucket breakdown (total CUDA={total_tri:.1f}ms):")
    for k, v in sorted(b_tri.items(), key=lambda x: -x[1]):
        pct = 100 * v / total_tri if total_tri else 0
        print(f"    {k:<42} {v:>8.2f} ms  ({pct:>5.1f}%)")
    print_top_events(ev_tri, "triton_gqa", n=15)

    if not args.skip_sdpa:
        print(f"\n--- Profile under sdpa ---")
        ev_sdpa = profile_fwd(model, ids, "sdpa")
        b_sdpa = bucket_events(ev_sdpa)
        total_sdpa = sum(b_sdpa.values())
        print(f"  Bucket breakdown (total CUDA={total_sdpa:.1f}ms):")
        for k, v in sorted(b_sdpa.items(), key=lambda x: -x[1]):
            pct = 100 * v / total_sdpa if total_sdpa else 0
            print(f"    {k:<42} {v:>8.2f} ms  ({pct:>5.1f}%)")
        print_top_events(ev_sdpa, "sdpa", n=15)

        print(f"\n--- Attention-specific delta (SDPA → Triton) ---")
        attn_key = "Attention (flash_attn kernel / SDPA)"
        d_attn = b_sdpa[attn_key] - b_tri[attn_key]
        print(f"  Attention CUDA time saved by Triton: {d_attn:+.2f} ms "
              f"(sdpa={b_sdpa[attn_key]:.2f}, tri={b_tri[attn_key]:.2f})")
        print(f"  Non-attention CUDA time (should be equal): "
              f"sdpa={total_sdpa - b_sdpa[attn_key]:.1f}ms, "
              f"tri={total_tri - b_tri[attn_key]:.1f}ms")
        attn_share_sdpa = 100 * b_sdpa[attn_key] / total_sdpa
        attn_share_tri = 100 * b_tri[attn_key] / total_tri
        print(f"  Attention share of total CUDA: sdpa={attn_share_sdpa:.1f}%  "
              f"tri={attn_share_tri:.1f}%")
        # E2E ceiling: even a 100x-faster attention kernel only reduces E2E by
        # b_tri[attn_key] at most. Print the ceiling for sanity.
        ceiling_ms = total_tri - b_tri[attn_key]
        print(f"  E2E Amdahl ceiling if attention → 0: ~{ceiling_ms:.1f} ms of non-attention work "
              f"remains (={100*ceiling_ms/total_tri:.1f}% of current)")


if __name__ == "__main__":
    sys.exit(main())
