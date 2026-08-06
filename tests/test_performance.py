from __future__ import annotations

import math

import pytest
import torch

import flash_attn.performance as performance
from flash_attn.performance import (
    UnknownHardwarePeak,
    achieved_tflops,
    attention_flops,
    attention_pair_count,
    compare_tensors,
    latency_summary,
    mfu_percent,
    resolve_hardware_peak,
    varlen_attention_flops,
    varlen_rectangular_grid_stats,
    write_json_exclusive,
)


@pytest.mark.parametrize(
    ("length", "window", "expected"),
    [
        (1, 0, 1),
        (4, 0, 10),
        (4, 1, 4),
        (4, 2, 7),
        (4, 4, 10),
        (4, 8, 10),
    ],
)
def test_causal_attention_pair_count(length: int, window: int, expected: int) -> None:
    assert attention_pair_count(
        length, length, causal=True, window_size=window
    ) == expected


def test_noncausal_attention_pair_count() -> None:
    assert attention_pair_count(3, 5, causal=False) == 15


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_length": 3, "key_length": 4, "causal": True},
        {"query_length": 3, "key_length": 3, "causal": False, "window_size": 2},
    ],
)
def test_unsupported_pair_count_conventions_fail_loudly(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        attention_pair_count(**kwargs)


def test_forward_and_training_flop_conventions() -> None:
    common = dict(
        batch_size=2,
        q_heads=8,
        head_dim=64,
        query_length=4,
        key_length=4,
        causal=True,
        window_size=0,
    )
    forward = attention_flops(**common, phase="forward")
    training = attention_flops(**common, phase="forward_backward")
    assert forward == 2 * 8 * 64 * 10 * 4
    assert training == 3 * forward


def test_varlen_flops_sum_real_lengths_not_padded_rectangle() -> None:
    lengths = (1, 4)
    expected = sum(
        attention_flops(
            batch_size=1,
            q_heads=8,
            head_dim=64,
            query_length=length,
            key_length=length,
            causal=True,
            phase="forward_backward",
        )
        for length in lengths
    )
    assert varlen_attention_flops(
        lengths,
        q_heads=8,
        head_dim=64,
        causal=True,
        phase="forward_backward",
    ) == expected


def test_varlen_rectangular_grid_reports_tail_waste() -> None:
    stats = varlen_rectangular_grid_stats(
        (64, 128), block_size=64, heads=2, q_splits=1
    )
    assert stats == {
        "launched_programs": 8,
        "active_programs_upper_bound": 6,
        "tail_early_return_programs": 2,
        "active_fraction": 0.75,
    }


def test_throughput_and_mfu_units() -> None:
    # 1 TFLOP completed in 1 ms is 1000 TFLOP/s.
    assert achieved_tflops(10**12, 1.0) == pytest.approx(1000.0)
    assert mfu_percent(10**12, 1.0, 2000.0) == pytest.approx(50.0)


def test_latency_summary_keeps_distribution() -> None:
    summary = latency_summary([5.0, 1.0, 3.0, 2.0, 4.0])
    assert summary["median"] == 3.0
    assert summary["p20"] == pytest.approx(1.8)
    assert summary["p80"] == pytest.approx(4.2)
    assert summary["raw"] == [5.0, 1.0, 3.0, 2.0, 4.0]


@pytest.mark.parametrize(
    ("name", "peak_id", "dense_tflops", "bandwidth"),
    [
        ("NVIDIA B200", "nvidia_b200_hgx", 2250.0, 7750.0),
        ("NVIDIA H200", "nvidia_h200_sxm", 989.5, 4800.0),
        ("NVIDIA H200 NVL", "nvidia_h200_nvl", 835.5, 4800.0),
        ("NVIDIA H100 80GB HBM3", "nvidia_h100_sxm", 989.5, 3350.0),
        ("NVIDIA H100 NVL", "nvidia_h100_nvl", 835.5, 3900.0),
        ("NVIDIA H100 PCIe", "nvidia_h100_pcie", 756.5, 2000.0),
    ],
)
def test_product_qualified_peak_lookup(
    name: str, peak_id: str, dense_tflops: float, bandwidth: float
) -> None:
    peak = resolve_hardware_peak(name)
    assert peak.id == peak_id
    assert peak.dense_tflops("bf16") == dense_tflops
    assert peak.hbm_bandwidth_gbps == bandwidth


def test_generic_or_unknown_h100_peak_is_not_guessed() -> None:
    with pytest.raises(UnknownHardwarePeak):
        resolve_hardware_peak("NVIDIA H100")
    with pytest.raises(UnknownHardwarePeak):
        resolve_hardware_peak("Future GPU")


def test_tensor_comparison_accumulates_in_fp32() -> None:
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    actual = expected.clone()
    metrics = compare_tensors(actual, expected)
    assert metrics.cosine == pytest.approx(1.0)
    assert metrics.max_abs == 0.0
    assert math.isfinite(metrics.relative_l2)


def test_tensor_comparison_chunks_dot_above_int32_safe_limit(monkeypatch) -> None:
    monkeypatch.setattr(performance, "_DOT_CHUNK_ELEMENTS", 2)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    actual = expected.clone()
    metrics = compare_tensors(actual, expected)
    assert metrics.cosine == pytest.approx(1.0)
    assert metrics.relative_l2 == 0.0


def test_json_results_are_never_overwritten(tmp_path) -> None:
    destination = tmp_path / "run" / "result.json"
    write_json_exclusive(destination, {"status": "first"})
    with pytest.raises(FileExistsError):
        write_json_exclusive(destination, {"status": "replacement"})
    assert '"first"' in destination.read_text()
