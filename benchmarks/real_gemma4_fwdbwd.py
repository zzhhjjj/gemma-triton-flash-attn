"""Real Gemma-4-E2B fwd+bwd training-style throughput benchmark.

Run:
    source /opt/tiger/flash_gemma/bin/activate
    python benchmarks/real_gemma4_fwdbwd.py

Compared to test_gemma4.py (fwd-only), this adds a backward pass + AdamW step
so we can observe Triton bwd optimizations in real training.
"""
import argparse
import gc
import os
import sys
import time

import torch
import transformers  # noqa: F401

from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
)
patch_transformers_5_5_4_flash_attn_key()

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def build_model(model_id, impl, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="cuda", attn_implementation=impl,
    )
    return model


def bench_step(model, seq_len, n_warmup=3, n_rep=8):
    """One training-style step: forward + mean loss + backward + AdamW.step."""
    device = "cuda"
    vocab_size = model.config.text_config.vocab_size if hasattr(model.config, 'text_config') else model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(input_ids, use_cache=False)
        logits = out.logits
        # Simple LM-style loss on shifted targets
        loss = logits.float().mean()
        loss.backward()
        opt.step()

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()

    ts = []
    for _ in range(n_rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts)//2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    args = p.parse_args()

    register_triton_attention()

    print(f"=== Real Gemma-4-E2B fwd+bwd benchmark (model={args.model}) ===")
    print(f"{'seq_len':>7} | {'SDPA (ms)':>10} | {'Triton (ms)':>11} | {'Speedup':>8}")
    print('-' * 50)

    for N in args.seq_lens:
        # Load with SDPA
        print(f"\n[load SDPA] seq_len={N}")
        model_s = build_model(args.model, "sdpa")
        for p in model_s.parameters(): p.requires_grad = True
        try:
            t_s = bench_step(model_s, N)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_s = float('nan')
        del model_s
        gc.collect(); torch.cuda.empty_cache()

        # Load with Triton
        print(f"[load Triton] seq_len={N}")
        model_t = build_model(args.model, "triton_gqa")
        for p in model_t.parameters(): p.requires_grad = True
        try:
            t_t = bench_step(model_t, N)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_t = float('nan')
        del model_t
        gc.collect(); torch.cuda.empty_cache()

        sp = t_s / t_t if (t_s == t_s and t_t == t_t and t_t > 0) else float('nan')
        print(f"{N:>7} | {t_s:>10.2f} | {t_t:>11.2f} | {sp:>7.2f}x")


if __name__ == "__main__":
    main()
