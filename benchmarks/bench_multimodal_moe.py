"""Gemma-4-26B-A4B multimodal (vision-bidirectional) correctness + speed bench.

Exercises the OR-mask path end-to-end through real model weights:
  - Real 26B-A4B MoE weights, loaded with device_map='auto'.
  - Real tokenizer output — BOI + 280 image soft-tokens + EOI, surrounded by
    prose tokens. This produces authentic `mm_token_type_ids` of the shape
    `create_causal_mask_mapping` consumes to build the upstream OR-mask.
  - Vision encoder is bypassed (`pixel_values=None`); the image-slot
    embeddings are just the learned embedding for token id `<|image|>`.
    The attention mask + kernel path don't care about the embed values —
    only the group layout — so this is a fair test for attention correctness.

Run:
    source /opt/tiger/flash_gemma/bin/activate
    python benchmarks/bench_multimodal_moe.py
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch
import transformers  # noqa: F401 — force early import

# Apply the 5.5.4 import workaround BEFORE any config load.
from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
    patch_gemma4_image_group_ids_for_kernel,
    patch_gemma4_shared_kv_states_for_fsdp2,
)
patch_transformers_5_5_4_flash_attn_key()

from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer  # noqa: E402


MODEL_ID_DEFAULT = "google/gemma-4-26B-A4B"


# =====================================================================
# Input builder: real tokens, real mm_token_type_ids, no pixel_values
# =====================================================================

def build_multimodal_input(tokenizer, cfg, *, total_len: int, n_images: int = 1,
                           device: str = "cuda:0"):
    """Construct (input_ids, mm_token_type_ids) that look like a real
    multimodal prompt: prose text containing `n_images` image spans, padded
    with more prose to reach exactly `total_len` tokens.

    mm_token_type_ids is 1 at image-soft-token positions (between BOI/EOI),
    0 elsewhere — same convention `create_causal_mask_mapping` consumes.
    """
    soft_per_image = cfg.vision_soft_tokens_per_image            # 280 for 26B-A4B
    image_token_id = cfg.image_token_id
    boi_token_id = cfg.boi_token_id
    eoi_token_id = cfg.eoi_token_id

    # Start with a prose prefix.
    prefix = tokenizer(
        "Describe the images below and then answer a question about them. ",
        add_special_tokens=False,
    ).input_ids

    # Each image span = [BOI] + 280 × image_token + [EOI]
    image_span = [boi_token_id] + [image_token_id] * soft_per_image + [eoi_token_id]

    # Filler prose between / after images.
    filler_one = tokenizer(
        "The image above is interesting. Based on it, continue the description "
        "with concrete detail: colors, shapes, lighting, composition, mood, and "
        "anything else that stands out clearly.",
        add_special_tokens=False,
    ).input_ids

    ids: list[int] = list(prefix)
    for _ in range(n_images):
        ids.extend(image_span)
        ids.extend(filler_one)

    # Pad (or truncate) with prose filler tokens to reach total_len.
    pad_token = tokenizer.encode(" and", add_special_tokens=False)[0]
    if len(ids) > total_len:
        ids = ids[:total_len]
    while len(ids) < total_len:
        ids.append(pad_token)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # mm_token_type_ids = 1 exactly where input_ids == image_token_id.
    mm = (input_ids == image_token_id).long()

    return input_ids, mm


# =====================================================================
# Correctness
# =====================================================================

def run_forward(model, input_ids, mm, *, hidden=False):
    """Single no-grad forward; returns last-hidden logits (and optionally
    per-layer hidden states for drift analysis)."""
    kwargs = dict(input_ids=input_ids, pixel_values=None, use_cache=False)
    if mm is not None:
        kwargs["mm_token_type_ids"] = mm
    if hidden:
        kwargs["output_hidden_states"] = True
    with torch.no_grad():
        out = model(**kwargs)
    return (out.logits, out.hidden_states) if hidden else out.logits


def compare(ref, triton_out):
    diff = (ref.float() - triton_out.float()).abs()
    rel = diff.norm() / ref.float().norm()
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), triton_out.float().flatten(), dim=0).item()
    am_ref = ref[:, -1].argmax(-1)
    am_tri = triton_out[:, -1].argmax(-1)
    top1 = (am_ref == am_tri).float().mean().item()
    # top-5 overlap at last position
    k = 5
    top_ref = ref[:, -1].topk(k, dim=-1).indices.tolist()[0]
    top_tri = triton_out[:, -1].topk(k, dim=-1).indices.tolist()[0]
    overlap = len(set(top_ref) & set(top_tri))
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_frob": rel.item(),
        "cos_sim": cos,
        "top1_match": top1,
        f"top{k}_overlap_last": f"{overlap}/{k}",
    }


# =====================================================================
# Speed
# =====================================================================

def bench_fwd(model, input_ids, mm, *, n_warmup=3, n_rep=5):
    for _ in range(n_warmup):
        run_forward(model, input_ids, mm)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_forward(model, input_ids, mm)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def set_impl(model, impl: str):
    """Flip _attn_implementation on both top-level and text configs."""
    model.config._attn_implementation = impl
    tc = model.config.get_text_config()
    if tc is not model.config:
        tc._attn_implementation = impl


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID_DEFAULT)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--images", type=int, nargs="+", default=[1, 2, 3],
                   help="Number of image spans to place (each = 282 tokens with BOI/EOI)")
    p.add_argument("--skip-perf", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--sanity-no-images", action="store_true",
                   help="Set mm_token_type_ids all-zero (text-only) — if this "
                        "also fails, the problem is model-wide, not OR-mask.")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    # Register Triton + image-group patch BEFORE model instantiation so the
    # wrapper is in place on the very first forward.
    register_triton_attention()
    patch_gemma4_image_group_ids_for_kernel()

    print(f"[load] tokenizer + config: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    tc = cfg.get_text_config()
    print(f"[load] text: layers={tc.num_hidden_layers} H_Q={tc.num_attention_heads} "
          f"H_KV={tc.num_key_value_heads} D_slide={tc.head_dim} D_full={tc.global_head_dim} "
          f"slide={tc.sliding_window}")
    print(f"[load] vision_soft_tokens_per_image={cfg.vision_soft_tokens_per_image} "
          f"use_bidirectional_attention={tc.use_bidirectional_attention}")

    print(f"[load] weights (bf16, device_map='auto') — may take a minute...")
    t_load_start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    t_load = time.perf_counter() - t_load_start
    print(f"[load] done in {t_load:.1f}s")
    print(f"[load] first param device: {next(model.parameters()).device}")

    # =====================================================================
    # Correctness
    # =====================================================================
    print(f"\n=== Correctness (SDPA vs Triton, N=2048, n_images=2) ===")
    # Use a shape that has an image span straddling block boundaries.
    input_ids, mm = build_multimodal_input(
        tokenizer, cfg, total_len=2048, n_images=2,
        device=next(model.parameters()).device.type + ":" + str(next(model.parameters()).device.index),
    )
    if args.sanity_no_images:
        mm = None  # skip passing mm_token_type_ids entirely (pure text path)
        print("[input] SANITY: omitting mm_token_type_ids (pure text path, no OR-mask)")
    n_img_tokens = int(mm.sum().item()) if mm is not None else 0
    print(f"[input] ids shape={tuple(input_ids.shape)}, image tokens={n_img_tokens} "
          f"(expected {2 * cfg.vision_soft_tokens_per_image if mm is not None else 0})")

    set_impl(model, "sdpa")
    ref, h_ref = run_forward(model, input_ids, mm, hidden=True)
    ref, h_ref = ref.clone(), [h.clone() for h in h_ref]

    set_impl(model, "triton_gqa")
    tri, h_tri = run_forward(model, input_ids, mm, hidden=True)
    tri, h_tri = tri.clone(), [h.clone() for h in h_tri]

    stats = compare(ref, tri)
    print(f"[compare] max|Δ|={stats['max_abs']:.3e}  mean|Δ|={stats['mean_abs']:.3e}  "
          f"rel_frob={stats['rel_frob']:.3e}  cos={stats['cos_sim']:.6f}")
    print(f"[compare] last-pos top-1 match={stats['top1_match'] * 100:.1f}%  "
          f"top-5 overlap={stats['top5_overlap_last']}")

    # Per-layer drift — tells us whether deviation is accumulated model noise
    # or a per-layer kernel bug. We expect small at layer 0 and growing slowly.
    print(f"\n[layer drift] {'layer':>5} {'cos':>10} {'rel_frob':>10}")
    for i in (0, 1, 2, 5, 10, 15, 20, 25, len(h_ref) - 1):
        if i >= len(h_ref):
            continue
        a, b = h_ref[i].float(), h_tri[i].float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rel = ((a - b).norm() / a.norm()).item()
        print(f"[layer drift] {i:>5} {cos:>10.6f} {rel:>10.3e}")

    # Threshold reflects what's achievable on a 30-layer MoE at bf16 —
    # per-layer cos is always >0.999 (kernel is numerically correct within
    # one bf16 ULP, verified by tests/test_image_group_mask.py). Accumulated
    # divergence through MoE routers inflates final-logit distance beyond
    # what's meaningful for a correctness assertion; top-1 match at the last
    # position is the informative signal.
    layer1_cos = torch.nn.functional.cosine_similarity(
        h_ref[1].float().flatten(), h_tri[1].float().flatten(), dim=0).item()
    passed = layer1_cos > 0.9999 and stats["top1_match"] >= 0.95
    print(f"[compare] {'PASS' if passed else 'FAIL'} "
          f"(gate: layer-1 cos>0.9999 [got {layer1_cos:.6f}], "
          f"top1>=0.95 [got {stats['top1_match'] * 100:.1f}%])")
    del ref, tri, h_ref, h_tri
    torch.cuda.empty_cache()

    if args.skip_perf:
        sys.exit(0 if passed else 1)

    # =====================================================================
    # Speed: one sweep per image count, across seq_lens
    # =====================================================================
    print(f"\n=== Speed (forward, no autograd) ===")
    print(f"{'N':>6} {'images':>7} {'SDPA (ms)':>11} {'Triton (ms)':>12} {'speedup':>8}")
    print("-" * 50)

    for n_img in args.images:
        for N in args.seq_lens:
            needed = n_img * (2 + cfg.vision_soft_tokens_per_image) + 32
            if N < needed:
                continue
            ids, m = build_multimodal_input(
                tokenizer, cfg, total_len=N, n_images=n_img,
                device=next(model.parameters()).device.type + ":" +
                       str(next(model.parameters()).device.index),
            )

            set_impl(model, "sdpa")
            t_sdpa = bench_fwd(model, ids, m)
            set_impl(model, "triton_gqa")
            t_tri = bench_fwd(model, ids, m)
            sp = t_sdpa / t_tri
            print(f"{N:>6} {n_img:>7} {t_sdpa:>11.2f} {t_tri:>12.2f} {sp:>7.2f}x")

            del ids, m
            gc.collect()
            torch.cuda.empty_cache()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
