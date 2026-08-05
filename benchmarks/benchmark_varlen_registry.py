#!/usr/bin/env python3
"""Canonical registry-backed benchmark for packed varlen Gemma4 attention."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Callable

import torch
import torch.nn.functional as F

from flash_attn import (
    DEFAULT_MODEL_PROFILES,
    attention_gqa_varlen_ref,
    capture_attention_selection,
    flash_attn_gqa_varlen,
)
from flash_attn.performance import (
    UnknownHardwarePeak,
    achieved_tflops,
    collect_git_metadata,
    collect_runtime_metadata,
    compare_tensors,
    latency_summary,
    mfu_percent,
    resolve_hardware_peak,
    varlen_attention_flops,
    varlen_rectangular_grid_stats,
    write_json_exclusive,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "attention-benchmark/v1"
VARLEN_PROFILE_IDS = (
    "gemma4_e2b_text_full",
    "gemma4_e2b_text_sliding",
    "gemma4_moe_text_full",
    "gemma4_moe_text_sliding",
)
OUTPUT_TOLERANCE = {
    "cosine_min": 0.9999,
    "max_abs": 5e-2,
    "mean_abs": 7e-4,
    "relative_l2": 2e-2,
}
GRAD_TOLERANCE = {
    "cosine_min": 0.9999,
    "max_abs": 2e-1,
    "mean_abs": 2e-3,
    "relative_l2": 3e-2,
}


def parse_lengths(values: list[str]) -> tuple[int, ...]:
    parts = [part for value in values for part in value.split(",") if part]
    try:
        lengths = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("lengths must be integers") from error
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive and non-empty")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=VARLEN_PROFILE_IDS, required=True)
    parser.add_argument(
        "--lengths",
        nargs="+",
        required=True,
        help="Packed sample lengths, space- or comma-separated.",
    )
    parser.add_argument(
        "--phase", choices=("forward", "forward_backward"), default="forward"
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "exp" / "b200_speedup" / "runs"
    )
    parser.add_argument("--dense-peak-tflops", type=float)
    parser.add_argument("--hbm-bandwidth-gbps", type=float)
    args = parser.parse_args()
    try:
        args.lengths = parse_lengths(args.lengths)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("warmup must be non-negative and repetitions positive")
    if (args.dense_peak_tflops is None) != (args.hbm_bandwidth_gbps is None):
        parser.error("explicit peak override requires peak and HBM bandwidth")
    if args.dense_peak_tflops is not None and (
        args.dense_peak_tflops <= 0 or args.hbm_bandwidth_gbps <= 0
    ):
        parser.error("explicit peak and HBM bandwidth must be positive")
    return args


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def metric_verdict(
    actual: torch.Tensor,
    expected: torch.Tensor,
    tolerance: dict[str, float],
) -> dict[str, object]:
    metrics = compare_tensors(actual, expected).to_dict()
    failures: list[str] = []
    if metrics["cosine"] < tolerance["cosine_min"]:
        failures.append("cosine")
    for field in ("max_abs", "mean_abs", "relative_l2"):
        if metrics[field] > tolerance[field]:
            failures.append(field)
    return {
        "passed": not failures,
        "failed_metrics": failures,
        "tolerance": dict(tolerance),
        "metrics": metrics,
    }


def assert_correctness(checks: dict[str, dict[str, object]]) -> None:
    failed = [name for name, check in checks.items() if not check["passed"]]
    if failed:
        detail = json.dumps({name: checks[name] for name in failed}, sort_keys=True)
        raise AssertionError(f"correctness failed for {', '.join(failed)}: {detail}")


def measure_cuda_ms(
    function: Callable[[], object], *, warmup: int, repetitions: int
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return latency_summary(
        [start.elapsed_time(end) for start, end in zip(starts, ends)]
    )


def fresh_training_tensors(
    q_data: torch.Tensor, k_data: torch.Tensor, v_data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        q_data.detach().clone().requires_grad_(True),
        k_data.detach().clone().requires_grad_(True),
        v_data.detach().clone().requires_grad_(True),
    )


def cumulative_lengths(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )


def prepare_sliding_masks(
    lengths: tuple[int, ...],
    *,
    causal: bool,
    window_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor | None, ...]:
    """Build exact-semantics SDPA masks once, outside timed regions."""
    masks: list[torch.Tensor | None] = []
    for length in lengths:
        if not causal or window_size == 0 or window_size >= length:
            masks.append(None)
            continue
        positions = torch.arange(length, device=device)
        allowed = (positions[None, :] <= positions[:, None]) & (
            positions[:, None] - positions[None, :] < window_size
        )
        mask = torch.where(allowed, 0.0, -float("inf")).to(dtype)
        masks.append(mask[None, None, :, :])
    return tuple(masks)


def packed_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: tuple[int, ...],
    masks: tuple[torch.Tensor | None, ...],
    *,
    causal: bool,
) -> torch.Tensor:
    """Exact varlen semantics via one PyTorch SDPA call per real sample."""
    outputs: list[torch.Tensor] = []
    offset = 0
    ratio = q.shape[1] // k.shape[1]
    for length, mask in zip(lengths, masks):
        end = offset + length
        q_sample = q[offset:end].transpose(0, 1).unsqueeze(0).contiguous()
        k_sample = k[offset:end].transpose(0, 1).unsqueeze(0).contiguous()
        v_sample = v[offset:end].transpose(0, 1).unsqueeze(0).contiguous()
        if ratio != 1:
            k_sample = k_sample.repeat_interleave(ratio, dim=1)
            v_sample = v_sample.repeat_interleave(ratio, dim=1)
        output = F.scaled_dot_product_attention(
            q_sample,
            k_sample,
            v_sample,
            attn_mask=mask,
            is_causal=causal and mask is None,
        )
        outputs.append(output.squeeze(0).transpose(0, 1).contiguous())
        offset = end
    return torch.cat(outputs, dim=0)


def validate_varlen_telemetry(
    snapshot: dict[str, object], *, phase: str
) -> None:
    selections = snapshot.get("selections")
    if not isinstance(selections, list) or not selections:
        raise AssertionError("varlen benchmark captured no registry selections")
    implementations = {row.get("implementation") for row in selections}
    if implementations != {"triton_gqa_varlen_v1"}:
        raise AssertionError(f"unexpected implementation selection: {implementations}")
    roles = {row.get("role") for row in selections}
    expected_roles = (
        {"forward"}
        if phase == "forward"
        else {"forward", "backward_dq", "backward_dkv"}
    )
    if roles != expected_roles:
        raise AssertionError(f"registry roles {roles} != expected {expected_roles}")
    if snapshot.get("total_fallbacks") != 0:
        raise AssertionError(f"unexpected fallback: {snapshot.get('fallbacks')}")


def grid_analysis(
    telemetry: dict[str, object], lengths: tuple[int, ...]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in telemetry["selections"]:
        role = str(raw["role"])
        if role in {"forward", "backward_dq"}:
            block_size = int(raw["block_q"])
            heads = int(raw["q_heads"])
            q_splits = 1
        else:
            block_size = int(raw["block_kv"])
            heads = int(raw["kv_heads"])
            q_splits = int(raw["q_splits"])
        rows.append(
            {
                "role": role,
                "config_id": raw["config_id"],
                "block_axis": "q" if role != "backward_dkv" else "kv",
                "block_size": block_size,
                "heads": heads,
                "q_splits": q_splits,
                **varlen_rectangular_grid_stats(
                    lengths,
                    block_size=block_size,
                    heads=heads,
                    q_splits=q_splits,
                ),
            }
        )
    return rows


def benchmark_cell(
    *,
    profile_id: str,
    lengths: tuple[int, ...],
    dtype_name: str,
    phase: str,
    device: torch.device,
    warmup: int,
    repetitions: int,
    seed: int,
    dense_peak_tflops: float,
) -> dict[str, object]:
    profile = DEFAULT_MODEL_PROFILES.get(profile_id)
    dtype = dtype_from_name(dtype_name)
    total_tokens = sum(lengths)
    max_seqlen = max(lengths)
    effective_window = 0 if profile.window_size >= max_seqlen else profile.window_size
    cu_seqlens = cumulative_lengths(lengths, device)
    sdpa_masks = prepare_sliding_masks(
        lengths,
        causal=profile.causal,
        window_size=effective_window,
        dtype=dtype,
        device=device,
    )

    torch.manual_seed(seed)
    q_data = torch.randn(
        total_tokens, profile.q_heads, profile.head_dim, dtype=dtype, device=device
    )
    k_data = torch.randn(
        total_tokens, profile.kv_heads, profile.head_dim, dtype=dtype, device=device
    )
    v_data = torch.randn(
        total_tokens, profile.kv_heads, profile.head_dim, dtype=dtype, device=device
    )

    checks: dict[str, dict[str, object]] = {}
    if phase == "forward":
        expected = attention_gqa_varlen_ref(
            q_data,
            k_data,
            v_data,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=profile.causal,
            window_size=effective_window,
        )
        with capture_attention_selection(
            "debug", labels={"profile": profile_id, "phase": phase, "layout": "thd"}
        ) as recorder:
            actual = flash_attn_gqa_varlen(
                q_data,
                k_data,
                v_data,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=profile.causal,
                window_size=effective_window,
            )
        checks["output"] = metric_verdict(actual, expected, OUTPUT_TOLERANCE)
        assert_correctness(checks)

        def run_triton() -> torch.Tensor:
            return flash_attn_gqa_varlen(
                q_data,
                k_data,
                v_data,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=profile.causal,
                window_size=effective_window,
            )

        def run_sdpa() -> torch.Tensor:
            return packed_sdpa(
                q_data,
                k_data,
                v_data,
                lengths,
                sdpa_masks,
                causal=profile.causal,
            )

    else:
        q_ref, k_ref, v_ref = fresh_training_tensors(q_data, k_data, v_data)
        expected = attention_gqa_varlen_ref(
            q_ref,
            k_ref,
            v_ref,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=profile.causal,
            window_size=effective_window,
        )
        torch.manual_seed(seed + 1)
        grad_out = torch.randn_like(expected)
        expected.backward(grad_out)

        q_tri, k_tri, v_tri = fresh_training_tensors(q_data, k_data, v_data)
        with capture_attention_selection(
            "debug", labels={"profile": profile_id, "phase": phase, "layout": "thd"}
        ) as recorder:
            actual = flash_attn_gqa_varlen(
                q_tri,
                k_tri,
                v_tri,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=profile.causal,
                window_size=effective_window,
            )
            actual.backward(grad_out)
        checks["output"] = metric_verdict(actual, expected, OUTPUT_TOLERANCE)
        for name, actual_grad, expected_grad in (
            ("dq", q_tri.grad, q_ref.grad),
            ("dk", k_tri.grad, k_ref.grad),
            ("dv", v_tri.grad, v_ref.grad),
        ):
            if actual_grad is None or expected_grad is None:
                raise AssertionError(f"missing {name} gradient")
            checks[name] = metric_verdict(actual_grad, expected_grad, GRAD_TOLERANCE)
        assert_correctness(checks)

        q_bench, k_bench, v_bench = fresh_training_tensors(q_data, k_data, v_data)
        q_sdpa, k_sdpa, v_sdpa = fresh_training_tensors(q_data, k_data, v_data)

        def run_triton() -> torch.Tensor:
            q_bench.grad = k_bench.grad = v_bench.grad = None
            output = flash_attn_gqa_varlen(
                q_bench,
                k_bench,
                v_bench,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=profile.causal,
                window_size=effective_window,
            )
            output.backward(grad_out)
            return output

        def run_sdpa() -> torch.Tensor:
            q_sdpa.grad = k_sdpa.grad = v_sdpa.grad = None
            output = packed_sdpa(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                lengths,
                sdpa_masks,
                causal=profile.causal,
            )
            output.backward(grad_out)
            return output

    telemetry = recorder.snapshot()
    validate_varlen_telemetry(telemetry, phase=phase)
    triton_latency = measure_cuda_ms(
        run_triton, warmup=warmup, repetitions=repetitions
    )
    sdpa_latency = measure_cuda_ms(run_sdpa, warmup=warmup, repetitions=repetitions)
    semantic_flops = varlen_attention_flops(
        lengths,
        q_heads=profile.q_heads,
        head_dim=profile.head_dim,
        causal=profile.causal,
        window_size=effective_window,
        phase=phase,
    )
    triton_median_ms = float(triton_latency["median"])
    sdpa_median_ms = float(sdpa_latency["median"])

    def performance_record(latency: dict[str, object]) -> dict[str, object]:
        median_ms = float(latency["median"])
        return {
            "latency": latency,
            "useful_tokens_per_second": total_tokens / (median_ms * 1e-3),
            "semantic_tflops": achieved_tflops(semantic_flops, median_ms),
            "attention_kernel_mfu_percent": mfu_percent(
                semantic_flops, median_ms, dense_peak_tflops
            ),
        }

    return {
        "profile_id": profile_id,
        "spec": {
            "layout": "thd",
            "q_heads": profile.q_heads,
            "kv_heads": profile.kv_heads,
            "head_dim": profile.head_dim,
            "dtype": dtype_name,
            "causal": profile.causal,
            "requested_window_size": profile.window_size,
            "effective_window_size": effective_window,
            "lengths": list(lengths),
            "batch_size": len(lengths),
            "total_tokens": total_tokens,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen,
            "phase": phase,
        },
        "flops": {
            "semantic_algorithmic": semantic_flops,
            "convention": (
                "sum over real sample lengths; dominant attention matmuls; "
                "multiply-add=2 FLOPs; softmax and recomputation excluded"
            ),
        },
        "correctness": checks,
        "registry_telemetry": telemetry,
        "grid_analysis": grid_analysis(telemetry, lengths),
        "measurements": {
            "triton_registry_public_api": performance_record(triton_latency),
            "torch_sdpa_per_sample": performance_record(sdpa_latency),
            "speedup_vs_torch_sdpa": sdpa_median_ms / triton_median_ms,
        },
        "reference": {
            "id": "exact_semantics_per_sample_torch_sdpa",
            "timed": True,
            "timing_scope": (
                "one SDPA call per real sample plus layout materialization and "
                "packed output concatenation; sliding masks prebuilt outside timing"
            ),
        },
    }


def peak_record(args: argparse.Namespace, runtime: dict[str, object]) -> dict[str, object]:
    if args.dense_peak_tflops is not None:
        return {
            "id": "explicit_override",
            "dtype": args.dtype,
            "dense_tensor_tflops": args.dense_peak_tflops,
            "hbm_bandwidth_gbps": args.hbm_bandwidth_gbps,
            "peak_convention": "user-supplied dense Tensor Core throughput",
            "source_url": None,
            "source_note": "Explicit CLI override; not from the built-in catalog.",
        }
    return resolve_hardware_peak(str(runtime["gpu_name"])).to_dict(args.dtype)


def slug(value: object) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-" for character in str(value)
    ).strip("-")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("canonical varlen benchmark requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"canonical varlen benchmark requires CUDA: {device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    started = datetime.now(timezone.utc)
    runtime = collect_runtime_metadata(device)
    git = collect_git_metadata(REPO_ROOT)
    hardware_peak = peak_record(args, runtime)
    cell = benchmark_cell(
        profile_id=args.profile,
        lengths=args.lengths,
        dtype_name=args.dtype,
        phase=args.phase,
        device=device,
        warmup=args.warmup,
        repetitions=args.repetitions,
        seed=args.seed,
        dense_peak_tflops=float(hardware_peak["dense_tensor_tflops"]),
    )
    completed = datetime.now(timezone.utc)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "policy": {
            "seed": args.seed,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "correctness_before_timing": True,
            "reference_timed": True,
            "sliding_masks_prebuilt_outside_timing": True,
            "output_is_exclusive": True,
        },
        "source": git,
        "runtime": runtime,
        "hardware_peak": hardware_peak,
        "cells": [cell],
    }
    run_id = "_".join(
        (
            completed.strftime("%Y%m%dT%H%M%S%fZ"),
            slug(args.profile),
            args.phase,
            slug(runtime["gpu_name"]),
            str(runtime["gpu_arch"]),
            str(git["commit"])[:8],
        )
    )
    result_path = args.output_root / run_id / "result.json"
    write_json_exclusive(result_path, payload)
    measurement = cell["measurements"]["triton_registry_public_api"]
    sdpa = cell["measurements"]["torch_sdpa_per_sample"]
    print(f"result: {result_path}")
    print(
        f"{args.profile} lengths={list(args.lengths)} {args.phase}: "
        f"{measurement['latency']['median']:.4f} ms, "
        f"{measurement['useful_tokens_per_second'] / 1e6:.3f} Mtok/s, "
        f"MFU={measurement['attention_kernel_mfu_percent']:.2f}%, "
        f"SDPA={sdpa['latency']['median']:.4f} ms, "
        f"speedup={cell['measurements']['speedup_vs_torch_sdpa']:.2f}x"
    )
    for row in cell["grid_analysis"]:
        print(
            f"  {row['role']}: {row['config_id']} active_fraction="
            f"{row['active_fraction']:.3f} "
            f"({row['active_programs_upper_bound']}/{row['launched_programs']})"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UnknownHardwarePeak as error:
        raise SystemExit(f"hardware peak lookup failed: {error}") from error
