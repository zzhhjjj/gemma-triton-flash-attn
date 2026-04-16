"""End-to-end integration test: swap Gemma4 attention → our Triton GQA kernel.

Uses transformers 5.5.4's ALL_ATTENTION_FUNCTIONS registry: register a custom
attention interface under the name "triton_gqa", then set the model's
`config._attn_implementation = "triton_gqa"` to enable it on every layer.

Run:
    source /opt/tiger/flash_gemma/bin/activate
    python tests/gemma4_integration/test_gemma4.py

Optional CLI:
    --model NAME     HF model id (default: google/gemma-4-E2B)
    --seq-len N      Test sequence length (default: 512)
    --skip-perf      Skip throughput benchmark (correctness only)
"""
import argparse
import os
import sys

import torch

# Workaround for transformers 5.5.4 bug: `KeyError: 'flash_attn'` when loading
# any config (happens before our registration can run).
import transformers  # noqa: F401 — force import before patching
from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
)
patch_transformers_5_5_4_flash_attn_key()

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# =====================================================================
# Integration test
# =====================================================================

def try_load_model(model_id: str, dtype=torch.bfloat16):
    """Load model with both attention implementations, sharing the same weights."""
    print(f"[load] Loading config for {model_id}...")
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as e:
        print(f"[load] FAILED to load config: {e}")
        return None, None, None

    print(f"[load] model type={config.model_type}")
    # Gemma4 may be a CausalLM or multimodal; grab the text config if nested
    text_cfg = getattr(config, "text_config", config)
    print(f"[load] hidden_size={getattr(text_cfg, 'hidden_size', '?')}, "
          f"num_layers={getattr(text_cfg, 'num_hidden_layers', '?')}, "
          f"num_heads={getattr(text_cfg, 'num_attention_heads', '?')}, "
          f"num_kv={getattr(text_cfg, 'num_key_value_heads', '?')}, "
          f"head_dim={getattr(text_cfg, 'head_dim', '?')}, "
          f"sliding_window={getattr(text_cfg, 'sliding_window', '?')}")

    print(f"[load] Downloading weights (bf16)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="cuda", attn_implementation="sdpa"
        )
    except Exception as e:
        print(f"[load] FAILED to load model: {e}")
        return None, None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tokenizer = None

    return model, tokenizer, config


def make_input(tokenizer, seq_len, batch_size=1, device="cuda"):
    """Build a test input — use tokenizer if available, else synthetic IDs."""
    if tokenizer is not None:
        prompt = "Hello, world! " * max(1, seq_len // 4)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # Truncate or pad to exactly seq_len
        if ids.shape[1] >= seq_len:
            ids = ids[:, :seq_len]
        else:
            pad = torch.zeros(1, seq_len - ids.shape[1], dtype=ids.dtype, device=device)
            ids = torch.cat([ids, pad], dim=1)
    else:
        ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    return ids


def test_correctness(model_id: str, seq_len: int):
    """Compare triton attn vs sdpa attn output on same input + weights."""
    print(f"\n=== Correctness Test ({model_id}, seq_len={seq_len}) ===")
    model, tokenizer, config = try_load_model(model_id)
    if model is None:
        return False

    register_triton_attention()
    ids = make_input(tokenizer, seq_len)
    print(f"[test] Input shape: {ids.shape}")

    # Forward with SDPA (reference)
    model.config._attn_implementation = "sdpa"
    # Propagate to text_config if it exists separately
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = "sdpa"
    with torch.no_grad():
        out_sdpa = model(ids).logits

    # Forward with Triton
    model.config._attn_implementation = "triton_gqa"
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = "triton_gqa"
    with torch.no_grad():
        out_tri = model(ids).logits

    abs_diff = (out_sdpa - out_tri).abs().max().item()
    rel_diff = (out_sdpa - out_tri).norm() / out_sdpa.norm()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_sdpa.float().flatten(), out_tri.float().flatten(), dim=0
    ).item()
    print(f"[test] logits shape: {out_sdpa.shape}")
    print(f"[test] max abs diff:     {abs_diff:.4e}")
    print(f"[test] rel frob diff:    {rel_diff:.4e}")
    print(f"[test] cos similarity:   {cos_sim:.6f}")

    # Check argmax match on last token (what really matters for generation)
    argmax_sdpa = out_sdpa[:, -1].argmax(-1)
    argmax_tri = out_tri[:, -1].argmax(-1)
    top_k_match = (argmax_sdpa == argmax_tri).float().mean().item()
    print(f"[test] top-1 token match (last pos): {top_k_match * 100:.1f}%")

    # Top-5 overlap
    top5_sdpa = out_sdpa[:, -1].topk(5, dim=-1).indices
    top5_tri = out_tri[:, -1].topk(5, dim=-1).indices
    overlap = len(set(top5_sdpa[0].tolist()) & set(top5_tri[0].tolist()))
    print(f"[test] top-5 overlap (last pos): {overlap}/5")

    passed = cos_sim > 0.999 and top_k_match >= 0.95
    print(f"[test] {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


def bench_throughput(model_id: str, seq_lens):
    """Measure forward throughput with sdpa vs triton attn."""
    print(f"\n=== Throughput Benchmark ({model_id}) ===")
    model, tokenizer, _ = try_load_model(model_id)
    if model is None:
        return
    register_triton_attention()
    model.eval()

    print(f"{'seq_len':>8} {'sdpa (ms)':>10} {'triton (ms)':>12} {'speedup':>8}")
    print("-" * 42)
    for N in seq_lens:
        ids = make_input(tokenizer, N)
        times = {}
        for impl in ("sdpa", "triton_gqa"):
            model.config._attn_implementation = impl
            if hasattr(model.config, "text_config"):
                model.config.text_config._attn_implementation = impl
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    model(ids)
            torch.cuda.synchronize()
            # Timed
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(5):
                start.record()
                with torch.no_grad():
                    model(ids)
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            samples.sort()
            times[impl] = samples[len(samples) // 2]  # median
        sp = times["sdpa"] / times["triton_gqa"]
        print(f"{N:>8} {times['sdpa']:>10.2f} {times['triton_gqa']:>12.2f} {sp:>7.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--skip-perf", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    ok = test_correctness(args.model, args.seq_len)

    if ok and not args.skip_perf:
        bench_throughput(args.model, [512, 1024, 2048])

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
