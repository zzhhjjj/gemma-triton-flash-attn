#!/usr/bin/env python3
"""Run one ephemeral B200 D512 varlen registry candidate.

This is an experiment-only wrapper around the canonical varlen benchmark. It
registers one higher-priority, process-local candidate, then calls the same
public API correctness and timing path as benchmark_varlen_registry.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import sys

import torch

from benchmarks.benchmark_varlen_registry import (
    SCHEMA_VERSION,
    benchmark_cell,
    slug,
)
from flash_attn.performance import (
    collect_git_metadata,
    collect_runtime_metadata,
    resolve_hardware_peak,
    write_json_exclusive,
)
from flash_attn.registry import (
    ConfigRegistration,
    DEFAULT_REGISTRY,
    KernelConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(part) for part in value.split(",") if part)
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--role",
        choices=("forward", "backward_dq", "backward_dkv"),
        default="backward_dkv",
    )
    parser.add_argument(
        "--profile",
        choices=("gemma4_e2b_text_full", "gemma4_moe_text_full"),
        required=True,
    )
    parser.add_argument("--lengths", type=parse_lengths, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--grad-output-scale", type=float, default=1.0)
    parser.add_argument("--block-q", type=int, required=True)
    parser.add_argument("--block-kv", type=int, required=True)
    parser.add_argument("--num-warps", type=int, required=True)
    parser.add_argument("--num-stages", type=int, required=True)
    parser.add_argument("--q-splits", type=int, default=1)
    parser.add_argument(
        "--separate-dkv-scratch",
        action="store_true",
        help="use separate dK/dV scratch, matching the production q-split path",
    )
    parser.add_argument(
        "--relaxed-dkv-atomics",
        action="store_true",
        help="use relaxed dK/dV atomics for the B200 q-split candidate",
    )
    parser.add_argument(
        "--split-gqa-heads",
        action="store_true",
        help="map GQA heads into the dKV grid instead of a static inner loop",
    )
    parser.add_argument(
        "--bf16x2-dkv-atomics",
        action="store_true",
        help="use relaxed BF16x2 atomics and BF16 dK/dV buffers",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("warmup must be non-negative and repetitions positive")
    if not math.isfinite(args.grad_output_scale) or args.grad_output_scale <= 0:
        parser.error("grad-output-scale must be finite and positive")
    if args.separate_dkv_scratch and (
        args.role != "backward_dkv"
        or (args.q_splits <= 1 and not args.split_gqa_heads)
    ):
        parser.error(
            "--separate-dkv-scratch requires --role backward_dkv and --q-splits > 1"
        )
    if args.relaxed_dkv_atomics and (
        args.role != "backward_dkv"
        or (args.q_splits <= 1 and not args.split_gqa_heads)
    ):
        parser.error(
            "--relaxed-dkv-atomics requires --role backward_dkv and --q-splits > 1"
        )
    if args.split_gqa_heads and args.role != "backward_dkv":
        parser.error("--split-gqa-heads requires --role backward_dkv")
    if args.bf16x2_dkv_atomics and (
        args.role != "backward_dkv"
        or not args.separate_dkv_scratch
        or args.dtype != "bfloat16"
    ):
        parser.error(
            "--bf16x2-dkv-atomics requires BF16 backward_dkv and "
            "--separate-dkv-scratch"
        )
    return args


def register_candidate(args: argparse.Namespace) -> str:
    candidate_id = (
        f"candidate.{args.experiment_id}.triton_gqa_varlen_v1."
        f"{args.role}.sm100.d512.bq{args.block_q}.bkv{args.block_kv}."
        f"w{args.num_warps}.s{args.num_stages}.qs{args.q_splits}."
        f"separate{int(args.separate_dkv_scratch)}."
        f"relaxed{int(args.relaxed_dkv_atomics)}."
        f"splitgqa{int(args.split_gqa_heads)}."
        f"bf16x2{int(args.bf16x2_dkv_atomics)}"
    )
    DEFAULT_REGISTRY.register_config(
        ConfigRegistration(
            id=candidate_id,
            implementation_id="triton_gqa_varlen_v1",
            gpu_arches=frozenset({"sm100"}),
            head_dims=frozenset({512}),
            config=KernelConfig(
                args.block_q,
                args.block_kv,
                512,
                args.num_warps,
                args.num_stages,
                q_splits=args.q_splits,
            ),
            evidence_status="baseline",
            evidence=f"ephemeral candidate from {args.experiment_id}",
            config_kind="tuned_override",
            role=args.role,
            priority=1000,
            dtypes=frozenset({args.dtype}),
            training_modes=frozenset({True}),
            gpu_name_patterns=frozenset({"B200"}),
            torch_version_prefixes=frozenset({"2.11"}),
            triton_version_prefixes=frozenset({"3.6"}),
            separate_dkv_scratch=args.separate_dkv_scratch,
            relaxed_dkv_atomics=args.relaxed_dkv_atomics,
            split_gqa_heads=args.split_gqa_heads,
            bf16x2_dkv_atomics=args.bf16x2_dkv_atomics,
        )
    )
    return candidate_id


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("candidate benchmark requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"candidate benchmark requires CUDA: {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    candidate_id = register_candidate(args)
    started = datetime.now(timezone.utc)
    runtime = collect_runtime_metadata(device)
    source = collect_git_metadata(REPO_ROOT)
    hardware_peak = resolve_hardware_peak(str(runtime["gpu_name"])).to_dict(args.dtype)
    cell = benchmark_cell(
        profile_id=args.profile,
        lengths=args.lengths,
        dtype_name=args.dtype,
        phase="forward_backward",
        device=device,
        warmup=args.warmup,
        repetitions=args.repetitions,
        seed=args.seed,
        dense_peak_tflops=float(hardware_peak["dense_tensor_tflops"]),
        grad_output_scale=args.grad_output_scale,
    )
    completed = datetime.now(timezone.utc)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "experiment_id": args.experiment_id,
        "candidate_role": args.role,
        "candidate_config_id": candidate_id,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "policy": {
            "seed": args.seed,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "grad_output_scale": args.grad_output_scale,
            "correctness_before_timing": True,
            "reference_timed": True,
            "memory_peak_measured_separately_from_timing": True,
            "memory_metric": "CUDA allocator peak growth above resident baseline",
            "output_is_exclusive": True,
            "candidate_is_ephemeral": True,
            "separate_dkv_scratch": args.separate_dkv_scratch,
            "relaxed_dkv_atomics": args.relaxed_dkv_atomics,
            "split_gqa_heads": args.split_gqa_heads,
            "bf16x2_dkv_atomics": args.bf16x2_dkv_atomics,
        },
        "source": source,
        "runtime": runtime,
        "hardware_peak": hardware_peak,
        "cells": [cell],
    }
    run_id = "_".join(
        (
            completed.strftime("%Y%m%dT%H%M%S%fZ"),
            slug(args.profile),
            args.role,
            f"bq{args.block_q}",
            slug(runtime["gpu_name"]),
            str(runtime["gpu_arch"]),
            str(source["commit"])[:8],
        )
    )
    result_path = args.output_root / run_id / "result.json"
    write_json_exclusive(result_path, payload)
    measurements = cell["measurements"]
    triton_record = measurements["triton_registry_public_api"]
    sdpa_record = measurements["torch_sdpa_per_sample"]
    print(f"result: {result_path}")
    print(
        json.dumps(
            {
                "candidate_config_id": candidate_id,
                "triton_median_ms": triton_record["latency"]["median"],
                "triton_incremental_peak_allocated_bytes": triton_record["memory"][
                    "incremental_peak_allocated_bytes"
                ],
                "sdpa_median_ms": sdpa_record["latency"]["median"],
                "sdpa_incremental_peak_allocated_bytes": sdpa_record["memory"][
                    "incremental_peak_allocated_bytes"
                ],
                "speedup": measurements["speedup_vs_torch_sdpa"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
