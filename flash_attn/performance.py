"""Shared, auditable performance metrics and run provenance.

This module is intentionally independent from kernel selection.  The attention
registry decides which implementation/config is valid; this module only counts
semantic work and converts measured latency into throughput/MFU using a
product-qualified hardware catalog.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import platform
from pathlib import Path
import subprocess
from typing import Literal


BenchmarkPhase = Literal["forward", "forward_backward"]


class UnknownHardwarePeak(ValueError):
    """The detected product/dtype has no unambiguous dense peak record."""


@dataclass(frozen=True)
class HardwarePeak:
    """Dense Tensor Core and HBM ceilings for one concrete GPU product."""

    id: str
    product_patterns: tuple[str, ...]
    dense_fp16_tflops: float
    dense_bf16_tflops: float
    hbm_bandwidth_gbps: float
    source_url: str
    source_note: str

    def dense_tflops(self, dtype: str) -> float:
        normalized = normalize_dtype(dtype)
        if normalized == "float16":
            return self.dense_fp16_tflops
        if normalized == "bfloat16":
            return self.dense_bf16_tflops
        raise UnknownHardwarePeak(
            f"{self.id} has no dense Tensor Core peak for dtype={normalized}"
        )

    def to_dict(self, dtype: str) -> dict[str, object]:
        return {
            "id": self.id,
            "dtype": normalize_dtype(dtype),
            "dense_tensor_tflops": self.dense_tflops(dtype),
            "hbm_bandwidth_gbps": self.hbm_bandwidth_gbps,
            "peak_convention": "dense (non-sparse) Tensor Core throughput",
            "source_url": self.source_url,
            "source_note": self.source_note,
        }


# Product-name matches are ordered from most specific to least specific.  The
# NVIDIA tables quote sparse throughput; dense values below are exactly half.
# B200 per-GPU values are derived from the official 8-GPU HGX totals.
HARDWARE_PEAKS: tuple[HardwarePeak, ...] = (
    HardwarePeak(
        id="nvidia_b200_hgx",
        product_patterns=("nvidia b200",),
        dense_fp16_tflops=2250.0,
        dense_bf16_tflops=2250.0,
        hbm_bandwidth_gbps=7750.0,
        source_url="https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf",
        source_note=(
            "Derived per GPU from the official 8-GPU HGX B200 totals: "
            "36 PFLOPS sparse FP16/BF16 and up to 62 TB/s HBM bandwidth; "
            "dense throughput is half the sparse figure."
        ),
    ),
    HardwarePeak(
        id="nvidia_h200_nvl",
        product_patterns=("h200 nvl",),
        dense_fp16_tflops=835.5,
        dense_bf16_tflops=835.5,
        hbm_bandwidth_gbps=4800.0,
        source_url="https://www.nvidia.com/en-gb/data-center/h200/",
        source_note="H200 NVL product-table values; dense throughput is half sparse.",
    ),
    HardwarePeak(
        id="nvidia_h200_sxm",
        product_patterns=("nvidia h200",),
        dense_fp16_tflops=989.5,
        dense_bf16_tflops=989.5,
        hbm_bandwidth_gbps=4800.0,
        source_url="https://www.nvidia.com/en-gb/data-center/h200/",
        source_note="H200 SXM product-table values; dense throughput is half sparse.",
    ),
    HardwarePeak(
        id="nvidia_h100_nvl",
        product_patterns=("h100 nvl",),
        dense_fp16_tflops=835.5,
        dense_bf16_tflops=835.5,
        hbm_bandwidth_gbps=3900.0,
        source_url="https://www.nvidia.com/en-us/data-center/h100/",
        source_note="H100 NVL product-table values; dense throughput is half sparse.",
    ),
    HardwarePeak(
        id="nvidia_h100_pcie",
        product_patterns=("h100 pcie",),
        dense_fp16_tflops=756.5,
        dense_bf16_tflops=756.5,
        hbm_bandwidth_gbps=2000.0,
        source_url="https://www.nvidia.com/en-us/data-center/h100/",
        source_note="H100 PCIe product-table values; dense throughput is half sparse.",
    ),
    HardwarePeak(
        id="nvidia_h100_sxm",
        product_patterns=("h100 80gb hbm3", "h100 sxm"),
        dense_fp16_tflops=989.5,
        dense_bf16_tflops=989.5,
        hbm_bandwidth_gbps=3350.0,
        source_url="https://www.nvidia.com/en-us/data-center/h100/",
        source_note="H100 SXM product-table values; dense throughput is half sparse.",
    ),
)


def normalize_dtype(dtype: object) -> str:
    value = str(dtype).removeprefix("torch.").lower()
    aliases = {
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }
    return aliases.get(value, value)


def resolve_hardware_peak(gpu_name: str) -> HardwarePeak:
    """Return a product-qualified peak or fail rather than guess a ceiling."""
    normalized_name = str(gpu_name).lower()
    for peak in HARDWARE_PEAKS:
        if any(pattern in normalized_name for pattern in peak.product_patterns):
            return peak
    raise UnknownHardwarePeak(
        f"no product-qualified hardware peak for gpu_name={gpu_name!r}; "
        "supply an explicit peak override to the benchmark"
    )


def attention_pair_count(
    query_length: int,
    key_length: int,
    *,
    causal: bool,
    window_size: int = 0,
) -> int:
    """Count query/key pairs admitted by the kernel's attention semantics."""
    if query_length <= 0 or key_length <= 0:
        raise ValueError("query_length and key_length must be positive")
    if window_size < 0:
        raise ValueError("window_size must be non-negative")
    if window_size and not causal:
        raise ValueError("sliding-window pair counts require causal=True")
    if not causal:
        return query_length * key_length
    if query_length != key_length:
        raise ValueError(
            "causal pair counts currently require self-attention with equal lengths"
        )

    length = query_length
    effective_window = length if window_size == 0 else min(window_size, length)
    return (
        effective_window * (effective_window + 1) // 2
        + (length - effective_window) * effective_window
    )


def attention_flops(
    *,
    batch_size: int,
    q_heads: int,
    head_dim: int,
    query_length: int,
    key_length: int,
    causal: bool,
    window_size: int = 0,
    phase: BenchmarkPhase = "forward",
) -> int:
    """Return dominant algorithmic matmul FLOPs for attention.

    One multiply-add counts as two FLOPs.  Forward counts QK^T and PV (four
    FLOPs per attended pair and head dimension).  Forward+backward additionally
    counts dV, dP, dQ, and dK (eight FLOPs), for twelve total.  Softmax scalar
    operations and implementation-specific recomputation are intentionally not
    included; this keeps MFU comparable across implementations.
    """
    if batch_size <= 0 or q_heads <= 0 or head_dim <= 0:
        raise ValueError("batch_size, q_heads, and head_dim must be positive")
    if phase not in {"forward", "forward_backward"}:
        raise ValueError(f"unsupported benchmark phase: {phase}")
    pairs = attention_pair_count(
        query_length,
        key_length,
        causal=causal,
        window_size=window_size,
    )
    flops_per_pair_dim = 4 if phase == "forward" else 12
    return batch_size * q_heads * head_dim * pairs * flops_per_pair_dim


def varlen_attention_flops(
    lengths: list[int] | tuple[int, ...],
    *,
    q_heads: int,
    head_dim: int,
    causal: bool,
    window_size: int = 0,
    phase: BenchmarkPhase = "forward",
) -> int:
    """Sum semantic attention FLOPs over real packed sample lengths."""
    normalized = tuple(int(length) for length in lengths)
    if not normalized or any(length <= 0 for length in normalized):
        raise ValueError("varlen lengths must be a non-empty sequence of positive integers")
    return sum(
        attention_flops(
            batch_size=1,
            q_heads=q_heads,
            head_dim=head_dim,
            query_length=length,
            key_length=length,
            causal=causal,
            window_size=window_size,
            phase=phase,
        )
        for length in normalized
    )


def varlen_rectangular_grid_stats(
    lengths: list[int] | tuple[int, ...],
    *,
    block_size: int,
    heads: int,
    q_splits: int = 1,
) -> dict[str, int | float]:
    """Compare launched rectangular programs with non-tail varlen programs."""
    normalized = tuple(int(length) for length in lengths)
    if not normalized or any(length <= 0 for length in normalized):
        raise ValueError("varlen lengths must be a non-empty sequence of positive integers")
    if block_size <= 0 or heads <= 0 or q_splits <= 0:
        raise ValueError("block_size, heads, and q_splits must be positive")
    batch_size = len(normalized)
    launched = math.ceil(max(normalized) / block_size) * batch_size * heads * q_splits
    active = sum(math.ceil(length / block_size) for length in normalized) * heads * q_splits
    return {
        "launched_programs": launched,
        "active_programs_upper_bound": active,
        "tail_early_return_programs": launched - active,
        "active_fraction": active / launched,
    }


def achieved_tflops(flops: int, latency_ms: float) -> float:
    if flops < 0:
        raise ValueError("flops must be non-negative")
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("latency_ms must be finite and positive")
    return flops / (latency_ms * 1e9)


def mfu_percent(flops: int, latency_ms: float, dense_peak_tflops: float) -> float:
    if not math.isfinite(dense_peak_tflops) or dense_peak_tflops <= 0:
        raise ValueError("dense_peak_tflops must be finite and positive")
    return 100.0 * achieved_tflops(flops, latency_ms) / dense_peak_tflops


def percentile(values: list[float] | tuple[float, ...], quantile: float) -> float:
    """Deterministic linearly interpolated percentile (quantile in [0, 1])."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(latencies_ms: list[float]) -> dict[str, object]:
    if not latencies_ms:
        raise ValueError("at least one latency sample is required")
    return {
        "unit": "ms",
        "samples": len(latencies_ms),
        "min": min(latencies_ms),
        "p20": percentile(latencies_ms, 0.20),
        "median": percentile(latencies_ms, 0.50),
        "p80": percentile(latencies_ms, 0.80),
        "max": max(latencies_ms),
        "raw": list(latencies_ms),
    }


@dataclass(frozen=True)
class TensorErrorMetrics:
    cosine: float
    max_abs: float
    mean_abs: float
    relative_l2: float
    actual_norm: float
    expected_norm: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


_DOT_CHUNK_ELEMENTS = 1 << 30


def _int32_safe_dot(actual: object, expected: object) -> object:
    """Accumulate dot products without exceeding cuBLAS' int32 length limit."""
    import torch

    if actual.numel() <= _DOT_CHUNK_ELEMENTS:
        return torch.dot(actual, expected)
    partials = [
        torch.dot(
            actual[start : start + _DOT_CHUNK_ELEMENTS],
            expected[start : start + _DOT_CHUNK_ELEMENTS],
        ).double()
        for start in range(0, actual.numel(), _DOT_CHUNK_ELEMENTS)
    ]
    return torch.stack(partials).sum()


def compare_tensors(actual: object, expected: object) -> TensorErrorMetrics:
    """Compare Torch tensors with all reductions accumulated in FP32."""
    import torch

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("actual and expected must be torch.Tensor instances")
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )
    if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
        raise ValueError("comparison tensors must contain only finite values")
    actual_f = actual.detach().float().reshape(-1)
    expected_f = expected.detach().float().reshape(-1)
    difference = actual_f - expected_f
    actual_norm_t = torch.linalg.vector_norm(actual_f)
    expected_norm_t = torch.linalg.vector_norm(expected_f)
    denominator = actual_norm_t * expected_norm_t
    if denominator.item() <= 1e-24:
        cosine = 1.0 if torch.equal(actual_f, expected_f) else 0.0
    else:
        cosine = _int32_safe_dot(actual_f, expected_f).div(denominator).item()
    expected_scale = max(expected_norm_t.item(), 1e-12)
    return TensorErrorMetrics(
        cosine=cosine,
        max_abs=difference.abs().max().item() if difference.numel() else 0.0,
        mean_abs=difference.abs().mean().item() if difference.numel() else 0.0,
        relative_l2=torch.linalg.vector_norm(difference).item() / expected_scale,
        actual_norm=actual_norm_t.item(),
        expected_norm=expected_norm_t.item(),
    )


def collect_runtime_metadata(device: object = "cuda") -> dict[str, object]:
    """Collect JSON-safe software and CUDA device provenance lazily."""
    import torch
    import triton

    resolved = torch.device(device)
    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)
    metadata: dict[str, object] = {
        "gpu_index": index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": properties.name,
        "gpu_arch": f"sm{major}{minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["nvidia_driver_version"] = query.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        metadata["nvidia_driver_version"] = "unknown"
    return metadata


def collect_git_metadata(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root)

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--short")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def write_json_exclusive(path: str | Path, payload: object) -> None:
    """Create a JSON record without ever replacing an existing result."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
