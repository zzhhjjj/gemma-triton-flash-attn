#!/usr/bin/env python3
"""Minimal registry-backed varlen target for NCU/NSYS capture."""

from __future__ import annotations

import argparse
import json

import torch

from flash_attn import (
    DEFAULT_MODEL_PROFILES,
    capture_attention_selection,
    flash_attn_gqa_varlen,
)


PROFILE_IDS = (
    "gemma4_e2b_text_full",
    "gemma4_e2b_text_sliding",
    "gemma4_moe_text_full",
    "gemma4_moe_text_sliding",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_IDS, required=True)
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--phase", choices=("forward", "forward_backward"), required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--cuda-profiler-api", action="store_true")
    args = parser.parse_args()
    args.lengths = tuple(int(value) for value in args.lengths.split(","))
    if not args.lengths or any(value <= 0 for value in args.lengths):
        parser.error("lengths must be positive")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    profile = DEFAULT_MODEL_PROFILES.get(args.profile)
    total = sum(args.lengths)
    maximum = max(args.lengths)
    window = 0 if profile.window_size >= maximum else profile.window_size
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(args.lengths, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    torch.manual_seed(20260801)
    q = torch.randn(total, profile.q_heads, profile.head_dim, dtype=dtype, device=device)
    k = torch.randn(total, profile.kv_heads, profile.head_dim, dtype=dtype, device=device)
    v = torch.randn(total, profile.kv_heads, profile.head_dim, dtype=dtype, device=device)
    grad_out = torch.randn_like(q)
    if args.phase == "forward_backward":
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

    def target() -> None:
        if args.phase == "forward_backward":
            q.grad = k.grad = v.grad = None
        output = flash_attn_gqa_varlen(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            maximum,
            maximum,
            causal=profile.causal,
            window_size=window,
        )
        if args.phase == "forward_backward":
            output.backward(grad_out)

    with capture_attention_selection(
        "debug", labels={"profile": args.profile, "phase": args.phase, "tool": "profiler"}
    ) as recorder:
        target()
    for _ in range(args.warmup - 1):
        target()
    torch.cuda.synchronize()
    print(json.dumps(recorder.snapshot(), sort_keys=True))

    if args.cuda_profiler_api:
        torch.cuda.profiler.start()
    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push("registry_varlen_target")
        target()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if args.cuda_profiler_api:
        torch.cuda.profiler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
