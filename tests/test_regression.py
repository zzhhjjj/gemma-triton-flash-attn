from __future__ import annotations

from copy import deepcopy

import pytest

from flash_attn.regression import (
    IncompatibleBenchmarkResults,
    compare_benchmark_results,
)


def _selection(config_id: str = "config.v1", evidence: str = "run-a") -> dict[str, object]:
    row = {
        "role": "forward",
        "implementation": "triton",
        "config_id": config_id,
        "config_kind": "base",
        "evidence_status": "verified",
        "block_q": 32,
        "block_kv": 32,
        "block_d": 256,
        "num_warps": 4,
        "num_stages": 2,
        "q_splits": 1,
    }
    return {
        "selections": [row],
        "debug_events": [
            {
                "resolution": {
                    "config_id": config_id,
                    "evidence_status": "verified",
                    "evidence": evidence,
                }
            }
        ],
    }


def _result(median: float = 1.0) -> dict[str, object]:
    return {
        "schema_version": "attention-benchmark/v1",
        "status": "passed",
        "source": {"commit": "abc"},
        "runtime": {
            "gpu_name": "NVIDIA B200",
            "gpu_arch": "sm100",
            "sm_count": 148,
            "torch_version": "2.11",
            "triton_version": "3.6",
            "cuda_runtime_version": "13.0",
            "nvidia_driver_version": "580",
        },
        "hardware_peak": {
            "id": "nvidia_b200_hgx",
            "dtype": "bfloat16",
            "dense_tensor_tflops": 2250.0,
        },
        "policy": {"warmup": 5, "repetitions": 20},
        "cells": [
            {
                "profile_id": "gemma4_e2b_text_full",
                "spec": {
                    "query_length": 1024,
                    "key_length": 1024,
                    "phase": "forward",
                },
                "correctness": {"output": {"passed": True}},
                "registry_telemetry": _selection(),
                "measurements": {
                    "triton_registry_public_api": {
                        "latency": {
                            "median": median,
                            "p20": median * 0.9,
                            "p80": median * 1.1,
                        }
                    }
                },
            }
        ],
    }


def test_matched_faster_candidate_passes() -> None:
    verdict = compare_benchmark_results(_result(1.0), _result(0.9))
    assert verdict["passed"] is True
    assert verdict["cells"][0]["latency"]["relative_change"] == pytest.approx(-0.1)


def test_latency_regression_fails() -> None:
    verdict = compare_benchmark_results(
        _result(1.0), _result(1.06), max_latency_regression=0.05
    )
    assert verdict["passed"] is False
    assert "latency_regression" in verdict["cells"][0]["failures"]


def test_correctness_regression_fails() -> None:
    candidate = _result()
    candidate["cells"][0]["correctness"]["output"]["passed"] = False
    verdict = compare_benchmark_results(_result(), candidate)
    assert verdict["passed"] is False
    assert "correctness" in verdict["cells"][0]["failures"]


def test_stale_config_drift_fails() -> None:
    candidate = _result()
    candidate["cells"][0]["registry_telemetry"]["selections"][0]["block_q"] = 64
    verdict = compare_benchmark_results(_result(), candidate)
    assert verdict["passed"] is False
    assert "config_drift_without_new_verified_evidence" in verdict["cells"][0]["failures"]


def test_new_config_with_new_verified_evidence_passes() -> None:
    candidate = _result(0.9)
    candidate["cells"][0]["registry_telemetry"] = _selection(
        "config.v2", "b200-ncu-run-20260801"
    )
    verdict = compare_benchmark_results(_result(), candidate)
    assert verdict["passed"] is True
    assert verdict["cells"][0]["selection"]["config_drift"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result["runtime"].update(gpu_name="NVIDIA H100 80GB HBM3"),
        lambda result: result["runtime"].update(triton_version="3.7"),
        lambda result: result["policy"].update(repetitions=10),
        lambda result: result["cells"][0]["spec"].update(query_length=2048),
    ],
)
def test_unmatched_runs_are_rejected(mutation) -> None:
    candidate = deepcopy(_result())
    mutation(candidate)
    with pytest.raises(IncompatibleBenchmarkResults):
        compare_benchmark_results(_result(), candidate)
