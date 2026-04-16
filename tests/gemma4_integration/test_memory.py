"""Memory benchmark: SDPA vs Triton on real Gemma-4-E2B forward.

Measures peak GPU allocation (above model weights) for a single forward pass
at increasing sequence lengths. Shows:
  - equal memory at short N (SDPA uses flash-attention internally)
  - Triton savings appear around N=16K (SDPA starts materialising attn scratch)
  - Triton runs at N=32K where SDPA OOMs — 2× context length on same hardware

Run:
    source /opt/tiger/flash_gemma/bin/activate
    export HF_TOKEN="hf_..."
    python tests/gemma4_integration/test_memory.py
"""
from __future__ import annotations

import gc
import os
import sys

import torch

# Bug workaround for transformers 5.5.4
import transformers  # noqa: F401
from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
)
patch_transformers_5_5_4_flash_attn_key()

from transformers import AutoModelForCausalLM  # noqa: E402


MB = 1024 ** 2
SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
MODEL_ID = os.environ.get("GEMMA4_MODEL", "google/gemma-4-E2B")


def set_impl(model, impl: str) -> None:
    model.config._attn_implementation = impl
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = impl


def measure_peak(model, impl: str, seq_len: int) -> float | None:
    """Return peak MB allocated during one forward pass, or None on OOM."""
    set_impl(model, impl)
    try:
        # Warmup (trigger Triton compile on first invocation)
        warm_ids = torch.randint(0, 1000, (1, 128), device="cuda")
        with torch.no_grad():
            _ = model(warm_ids)
        del warm_ids
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        ids = torch.randint(0, 1000, (1, seq_len), device="cuda")
        base_alloc = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model(ids)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base_alloc) / MB
        del ids
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None


def main():
    register_triton_attention()
    print(f"[load] {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    weight_mem = torch.cuda.memory_allocated() / MB
    print(f"[load] model weights: {weight_mem:.0f} MB")

    print()
    print(f"{'seq_len':>8} {'SDPA peak (MB)':>16} {'Triton peak (MB)':>18} "
          f"{'Reduction':>10} {'Saved':>10}")
    print("-" * 68)

    for N in SEQ_LENS:
        sdpa_peak = measure_peak(model, "sdpa", N)
        tri_peak = measure_peak(model, "triton_gqa", N)

        def fmt(v):
            return f"{v:>16.1f}" if v is not None else f"{'OOM':>16}"

        if sdpa_peak is None and tri_peak is None:
            print(f"{N:>8} {'OOM':>16} {'OOM':>18}")
            break
        if sdpa_peak is None:
            print(f"{N:>8} {'OOM':>16} {tri_peak:>18.1f} "
                  f"{'∞':>10} {'Triton only':>10}")
            continue
        if tri_peak is None:
            print(f"{N:>8} {sdpa_peak:>16.1f} {'OOM':>18}")
            continue

        ratio = sdpa_peak / tri_peak if tri_peak > 0 else 0
        saved = sdpa_peak - tri_peak
        print(f"{N:>8} {sdpa_peak:>16.1f} {tri_peak:>18.1f} "
              f"{ratio:>9.2f}x {saved:>9.1f}MB")

    print()
    print("Interpretation:")
    print("  - Short N: SDPA's flash backend keeps peak ~equal to Triton.")
    print("  - N≈16K:   SDPA starts materialising attn scratch; Triton's online")
    print("             softmax stays bounded → significant savings.")
    print("  - N≥32K:   SDPA OOMs on 80GB H100; Triton continues working.")


if __name__ == "__main__":
    sys.exit(main())
