#!/usr/bin/env python3
"""Canonical profile-driven benchmark for the registry-backed public API.

Every measured cell first compares identical attention semantics against the
independent PyTorch reference.  Successful runs create a new immutable result
directory containing correctness, latency distribution, registry telemetry,
MFU, and complete hardware/software/source provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Callable

import torch

from flash_attn import (
    DEFAULT_MODEL_PROFILES,
    attention_flash_gqa,
    attention_gqa_ref,
    attention_swa_ref,
    capture_attention_selection,
    flash_attn_gqa_train,
)
from flash_attn.performance import (
    UnknownHardwarePeak,
    achieved_tflops,
    attention_flops,
    collect_git_metadata,
    collect_runtime_metadata,
    compare_tensors,
    latency_summary,
    mfu_percent,
    resolve_hardware_peak,
    write_json_exclusive,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "attention-benchmark/v1"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=DEFAULT_MODEL_PROFILES.ids(),
        default="gemma4_e2b_text_full",
    )
    parser.add_argument("--seq-len", type=int, nargs="+", default=[129])
    parser.add_argument(
        "--phase",
        choices=("forward", "forward_backward"),
        default="forward",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "exp" / "refactor" / "runs",
    )
    parser.add_argument(
        "--dense-peak-tflops",
        type=float,
        help="Explicit dense peak for an uncatalogued GPU (requires --hbm-bandwidth-gbps).",
    )
    parser.add_argument(
        "--hbm-bandwidth-gbps",
        type=float,
        help="Explicit HBM ceiling for an uncatalogued GPU (requires --dense-peak-tflops).",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.repetitions <= 0:
        parser.error("batch-size/repetitions must be positive and warmup non-negative")
    if any(length <= 0 for length in args.seq_len):
        parser.error("all sequence lengths must be positive")
    if (args.dense_peak_tflops is None) != (args.hbm_bandwidth_gbps is None):
        parser.error("explicit peak override requires both peak and HBM bandwidth")
    if args.dense_peak_tflops is not None and (
        args.dense_peak_tflops <= 0 or args.hbm_bandwidth_gbps <= 0
    ):
        parser.error("explicit peak and HBM bandwidth must be positive")
    return args


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size: int,
) -> torch.Tensor:
    if window_size:
        return attention_swa_ref(q, k, v, slide_size=window_size)
    return attention_gqa_ref(q, k, v, causal=causal)


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


def _fresh_training_tensors(
    q_data: torch.Tensor, k_data: torch.Tensor, v_data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        q_data.detach().clone().requires_grad_(True),
        k_data.detach().clone().requires_grad_(True),
        v_data.detach().clone().requires_grad_(True),
    )


def benchmark_cell(
    *,
    profile_id: str,
    sequence_length: int,
    batch_size: int,
    dtype_name: str,
    phase: str,
    device: torch.device,
    warmup: int,
    repetitions: int,
    seed: int,
    dense_peak_tflops: float,
) -> dict[str, object]:
    profile = DEFAULT_MODEL_PROFILES.get(profile_id)
    if profile.image_groups:
        raise NotImplementedError(
            "canonical benchmark v1 does not synthesize Gemma image-group metadata"
        )
    dtype = dtype_from_name(dtype_name)
    effective_window = (
        0 if profile.window_size >= sequence_length else profile.window_size
    )
    spec = profile.make_spec(
        dtype=dtype_name,
        training=phase == "forward_backward",
        batch_size=batch_size,
        query_length=sequence_length,
        key_length=sequence_length,
        window_size=effective_window,
    )

    torch.manual_seed(seed)
    q_shape = (batch_size, profile.q_heads, sequence_length, profile.head_dim)
    kv_shape = (batch_size, profile.kv_heads, sequence_length, profile.head_dim)
    q_data = torch.randn(q_shape, dtype=dtype, device=device)
    k_data = torch.randn(kv_shape, dtype=dtype, device=device)
    v_data = torch.randn(kv_shape, dtype=dtype, device=device)

    checks: dict[str, dict[str, object]] = {}
    if phase == "forward":
        expected = reference_attention(
            q_data,
            k_data,
            v_data,
            causal=profile.causal,
            window_size=effective_window,
        )
        with capture_attention_selection(
            "debug",
            labels={"profile": profile_id, "seq_len": sequence_length, "phase": phase},
        ) as telemetry:
            actual = attention_flash_gqa(
                q_data,
                k_data,
                v_data,
                causal=profile.causal,
                slide_size=effective_window,
            )
        checks["output"] = metric_verdict(actual, expected, OUTPUT_TOLERANCE)
        assert_correctness(checks)

        def run_triton() -> torch.Tensor:
            return attention_flash_gqa(
                q_data,
                k_data,
                v_data,
                causal=profile.causal,
                slide_size=effective_window,
            )

        def run_reference() -> torch.Tensor:
            return reference_attention(
                q_data,
                k_data,
                v_data,
                causal=profile.causal,
                window_size=effective_window,
            )

        baseline_id = (
            "torch_sdpa_expanded_gqa_explicit_sliding_mask"
            if effective_window
            else "torch_sdpa_expanded_gqa"
        )
    else:
        q_ref, k_ref, v_ref = _fresh_training_tensors(q_data, k_data, v_data)
        expected = reference_attention(
            q_ref,
            k_ref,
            v_ref,
            causal=profile.causal,
            window_size=effective_window,
        )
        torch.manual_seed(seed + 1)
        grad_out = torch.randn_like(expected)
        expected.backward(grad_out)

        q_tri, k_tri, v_tri = _fresh_training_tensors(q_data, k_data, v_data)
        with capture_attention_selection(
            "debug",
            labels={"profile": profile_id, "seq_len": sequence_length, "phase": phase},
        ) as telemetry:
            actual = flash_attn_gqa_train(
                q_tri,
                k_tri,
                v_tri,
                causal=profile.causal,
                slide_size=effective_window,
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

        q_bench, k_bench, v_bench = _fresh_training_tensors(q_data, k_data, v_data)
        q_ref_bench, k_ref_bench, v_ref_bench = _fresh_training_tensors(
            q_data, k_data, v_data
        )

        def run_triton() -> torch.Tensor:
            q_bench.grad = k_bench.grad = v_bench.grad = None
            output = flash_attn_gqa_train(
                q_bench,
                k_bench,
                v_bench,
                causal=profile.causal,
                slide_size=effective_window,
            )
            output.backward(grad_out)
            return output

        def run_reference() -> torch.Tensor:
            q_ref_bench.grad = k_ref_bench.grad = v_ref_bench.grad = None
            output = reference_attention(
                q_ref_bench,
                k_ref_bench,
                v_ref_bench,
                causal=profile.causal,
                window_size=effective_window,
            )
            output.backward(grad_out)
            return output

        baseline_id = (
            "torch_sdpa_autograd_expanded_gqa_explicit_sliding_mask"
            if effective_window
            else "torch_sdpa_autograd_expanded_gqa"
        )

    triton_latency = measure_cuda_ms(
        run_triton, warmup=warmup, repetitions=repetitions
    )
    reference_latency = measure_cuda_ms(
        run_reference, warmup=warmup, repetitions=repetitions
    )
    semantic_flops = attention_flops(
        batch_size=batch_size,
        q_heads=profile.q_heads,
        head_dim=profile.head_dim,
        query_length=sequence_length,
        key_length=sequence_length,
        causal=profile.causal,
        window_size=effective_window,
        phase=phase,
    )

    def measurement(latency: dict[str, object]) -> dict[str, object]:
        median_ms = float(latency["median"])
        return {
            "latency": latency,
            "semantic_tflops": achieved_tflops(semantic_flops, median_ms),
            "attention_kernel_mfu_percent": mfu_percent(
                semantic_flops, median_ms, dense_peak_tflops
            ),
        }

    triton_measurement = measurement(triton_latency)
    reference_measurement = measurement(reference_latency)
    return {
        "profile_id": profile_id,
        "spec": {
            "q_heads": spec.q_heads,
            "kv_heads": spec.kv_heads,
            "head_dim": spec.head_dim,
            "dtype": spec.dtype,
            "causal": spec.causal,
            "requested_window_size": profile.window_size,
            "effective_window_size": spec.window_size,
            "batch_size": spec.batch_size,
            "query_length": spec.query_length,
            "key_length": spec.key_length,
            "phase": phase,
        },
        "flops": {
            "semantic_algorithmic": semantic_flops,
            "convention": (
                "dominant attention matmuls; multiply-add=2 FLOPs; "
                "softmax and implementation-specific recomputation excluded"
            ),
        },
        "correctness": checks,
        "registry_telemetry": telemetry.snapshot(),
        "measurements": {
            "triton_registry_public_api": triton_measurement,
            baseline_id: reference_measurement,
        },
        "speedup_vs_reference": (
            float(reference_latency["median"]) / float(triton_latency["median"])
        ),
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
    peak = resolve_hardware_peak(str(runtime["gpu_name"]))
    return peak.to_dict(args.dtype)


def slug(value: object) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in str(value)).strip("-")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("canonical attention benchmark requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"canonical attention benchmark requires a CUDA device: {device}")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    started = datetime.now(timezone.utc)
    runtime = collect_runtime_metadata(device)
    git = collect_git_metadata(REPO_ROOT)
    hardware_peak = peak_record(args, runtime)
    cells = [
        benchmark_cell(
            profile_id=args.profile,
            sequence_length=sequence_length,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            phase=args.phase,
            device=device,
            warmup=args.warmup,
            repetitions=args.repetitions,
            seed=args.seed,
            dense_peak_tflops=float(hardware_peak["dense_tensor_tflops"]),
        )
        for sequence_length in args.seq_len
    ]
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
            "output_is_exclusive": True,
        },
        "source": git,
        "runtime": runtime,
        "hardware_peak": hardware_peak,
        "cells": cells,
    }
    run_id = "_".join(
        (
            completed.strftime("%Y%m%dT%H%M%S%fZ"),
            slug(runtime["gpu_name"]),
            str(runtime["gpu_arch"]),
            str(git["commit"])[:8],
        )
    )
    result_path = args.output_root / run_id / "result.json"
    write_json_exclusive(result_path, payload)
    print(f"result: {result_path}")
    for cell in cells:
        measurements = cell["measurements"]
        triton_result = measurements["triton_registry_public_api"]
        print(
            f"{cell['profile_id']} N={cell['spec']['query_length']} {args.phase}: "
            f"{triton_result['latency']['median']:.4f} ms, "
            f"{triton_result['semantic_tflops']:.2f} TFLOP/s, "
            f"attention-kernel MFU={triton_result['attention_kernel_mfu_percent']:.2f}%, "
            f"speedup={cell['speedup_vs_reference']:.3f}x"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UnknownHardwarePeak as error:
        raise SystemExit(f"hardware peak lookup failed: {error}") from error
