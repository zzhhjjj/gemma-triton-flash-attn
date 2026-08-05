"""Strict comparison of canonical attention benchmark result records."""

from __future__ import annotations

import json
import math
from typing import Mapping


class IncompatibleBenchmarkResults(ValueError):
    """Baseline and candidate do not describe comparable measurements."""


_RUNTIME_COMPATIBILITY_FIELDS = (
    "gpu_name",
    "gpu_arch",
    "sm_count",
    "torch_version",
    "triton_version",
    "cuda_runtime_version",
    "nvidia_driver_version",
)
_POLICY_COMPATIBILITY_FIELDS = ("warmup", "repetitions")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IncompatibleBenchmarkResults(f"{name} must be a JSON object")
    return value


def _check_schema(result: Mapping[str, object], name: str) -> None:
    if result.get("schema_version") != "attention-benchmark/v1":
        raise IncompatibleBenchmarkResults(
            f"{name} has unsupported schema_version={result.get('schema_version')!r}"
        )
    if result.get("status") != "passed":
        raise IncompatibleBenchmarkResults(
            f"{name} benchmark status is not passed: {result.get('status')!r}"
        )


def _compatibility_snapshot(result: Mapping[str, object]) -> dict[str, object]:
    runtime = _require_mapping(result.get("runtime"), "runtime")
    peak = _require_mapping(result.get("hardware_peak"), "hardware_peak")
    policy = _require_mapping(result.get("policy"), "policy")
    return {
        "runtime": {field: runtime.get(field) for field in _RUNTIME_COMPATIBILITY_FIELDS},
        "hardware_peak": {
            field: peak.get(field)
            for field in ("id", "dtype", "dense_tensor_tflops")
        },
        "measurement_policy": {
            field: policy.get(field) for field in _POLICY_COMPATIBILITY_FIELDS
        },
    }


def _cell_key(cell: Mapping[str, object]) -> str:
    spec = _require_mapping(cell.get("spec"), "cell.spec")
    identity = {
        "profile_id": cell.get("profile_id"),
        "spec": dict(spec),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _index_cells(result: Mapping[str, object], name: str) -> dict[str, Mapping[str, object]]:
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise IncompatibleBenchmarkResults(f"{name}.cells must be a JSON array")
    indexed: dict[str, Mapping[str, object]] = {}
    for raw_cell in cells:
        cell = _require_mapping(raw_cell, f"{name}.cell")
        key = _cell_key(cell)
        if key in indexed:
            raise IncompatibleBenchmarkResults(f"{name} contains duplicate cell {key}")
        indexed[key] = cell
    return indexed


def _selection_signature(cell: Mapping[str, object]) -> tuple[tuple[object, ...], ...]:
    telemetry = _require_mapping(cell.get("registry_telemetry"), "registry_telemetry")
    selections = telemetry.get("selections")
    if not isinstance(selections, list) or not selections:
        raise IncompatibleBenchmarkResults("cell has no registry selection telemetry")
    fields = (
        "role",
        "implementation",
        "config_id",
        "config_kind",
        "evidence_status",
        "block_q",
        "block_kv",
        "block_d",
        "num_warps",
        "num_stages",
        "q_splits",
    )
    return tuple(
        sorted(
            tuple(_require_mapping(row, "selection").get(field) for field in fields)
            for row in selections
        )
    )


def _config_evidence(cell: Mapping[str, object]) -> dict[str, tuple[object, object]]:
    telemetry = _require_mapping(cell.get("registry_telemetry"), "registry_telemetry")
    events = telemetry.get("debug_events")
    if not isinstance(events, list):
        return {}
    evidence: dict[str, tuple[object, object]] = {}
    for raw_event in events:
        event = _require_mapping(raw_event, "debug_event")
        resolution = _require_mapping(event.get("resolution"), "debug_event.resolution")
        config_id = resolution.get("config_id")
        if isinstance(config_id, str):
            evidence[config_id] = (
                resolution.get("evidence_status"),
                resolution.get("evidence"),
            )
    return evidence


def _config_drift_has_new_verified_evidence(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> bool:
    baseline_evidence = _config_evidence(baseline)
    candidate_evidence = _config_evidence(candidate)
    old_evidence_strings = {record[1] for record in baseline_evidence.values()}
    baseline_ids = set(baseline_evidence)
    candidate_ids = set(candidate_evidence)
    changed_ids = candidate_ids - baseline_ids
    if not changed_ids:
        # A changed tile under an unchanged config ID needs an evidence update,
        # not merely a code edit that kept stale identity/provenance.
        changed_ids = {
            config_id
            for config_id in candidate_ids & baseline_ids
            if candidate_evidence[config_id] != baseline_evidence[config_id]
        }
    return bool(changed_ids) and all(
        candidate_evidence[config_id][0] == "verified"
        and bool(candidate_evidence[config_id][1])
        and candidate_evidence[config_id][1] not in old_evidence_strings
        for config_id in changed_ids
    )


def _triton_measurement(cell: Mapping[str, object]) -> Mapping[str, object]:
    measurements = _require_mapping(cell.get("measurements"), "measurements")
    return _require_mapping(
        measurements.get("triton_registry_public_api"),
        "measurements.triton_registry_public_api",
    )


def _median_and_dispersion(cell: Mapping[str, object]) -> tuple[float, float]:
    measurement = _triton_measurement(cell)
    latency = _require_mapping(measurement.get("latency"), "measurement.latency")
    median = float(latency["median"])
    p20 = float(latency["p20"])
    p80 = float(latency["p80"])
    if not all(math.isfinite(value) for value in (median, p20, p80)) or median <= 0:
        raise IncompatibleBenchmarkResults("latency distribution is not finite/positive")
    return median, (p80 - p20) / median


def _correctness_passed(cell: Mapping[str, object]) -> bool:
    correctness = _require_mapping(cell.get("correctness"), "correctness")
    if not correctness:
        return False
    return all(
        bool(_require_mapping(check, "correctness check").get("passed"))
        for check in correctness.values()
    )


def compare_benchmark_results(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    max_latency_regression: float = 0.05,
) -> dict[str, object]:
    """Compare matched cells and return a JSON-safe pass/fail verdict."""
    if max_latency_regression < 0:
        raise ValueError("max_latency_regression must be non-negative")
    _check_schema(baseline, "baseline")
    _check_schema(candidate, "candidate")
    baseline_compatibility = _compatibility_snapshot(baseline)
    candidate_compatibility = _compatibility_snapshot(candidate)
    if baseline_compatibility != candidate_compatibility:
        raise IncompatibleBenchmarkResults(
            "hardware/software/measurement policy mismatch: "
            + json.dumps(
                {
                    "baseline": baseline_compatibility,
                    "candidate": candidate_compatibility,
                },
                sort_keys=True,
            )
        )

    baseline_cells = _index_cells(baseline, "baseline")
    candidate_cells = _index_cells(candidate, "candidate")
    missing = sorted(set(baseline_cells) - set(candidate_cells))
    unexpected = sorted(set(candidate_cells) - set(baseline_cells))
    if missing or unexpected:
        raise IncompatibleBenchmarkResults(
            f"cell matrix mismatch: missing={missing}, unexpected={unexpected}"
        )

    cell_verdicts: list[dict[str, object]] = []
    for key in sorted(baseline_cells):
        baseline_cell = baseline_cells[key]
        candidate_cell = candidate_cells[key]
        baseline_median, baseline_dispersion = _median_and_dispersion(baseline_cell)
        candidate_median, candidate_dispersion = _median_and_dispersion(candidate_cell)
        latency_change = candidate_median / baseline_median - 1.0
        correctness_passed = _correctness_passed(candidate_cell)
        baseline_selection = _selection_signature(baseline_cell)
        candidate_selection = _selection_signature(candidate_cell)
        config_drift = candidate_selection != baseline_selection
        drift_has_new_verified_evidence = (
            _config_drift_has_new_verified_evidence(baseline_cell, candidate_cell)
            if config_drift
            else False
        )
        config_drift_allowed = not config_drift or drift_has_new_verified_evidence
        failures: list[str] = []
        if not correctness_passed:
            failures.append("correctness")
        if latency_change > max_latency_regression:
            failures.append("latency_regression")
        if not config_drift_allowed:
            failures.append("config_drift_without_new_verified_evidence")
        cell_verdicts.append(
            {
                "cell": json.loads(key),
                "passed": not failures,
                "failures": failures,
                "correctness_passed": correctness_passed,
                "latency": {
                    "baseline_median_ms": baseline_median,
                    "candidate_median_ms": candidate_median,
                    "relative_change": latency_change,
                    "max_allowed_regression": max_latency_regression,
                    "baseline_relative_p20_p80_span": baseline_dispersion,
                    "candidate_relative_p20_p80_span": candidate_dispersion,
                },
                "selection": {
                    "config_drift": config_drift,
                    "drift_has_new_verified_evidence": drift_has_new_verified_evidence,
                    "baseline": baseline_selection,
                    "candidate": candidate_selection,
                },
            }
        )

    return {
        "schema_version": "attention-benchmark-comparison/v1",
        "passed": all(cell["passed"] for cell in cell_verdicts),
        "max_latency_regression": max_latency_regression,
        "compatibility": baseline_compatibility,
        "baseline_source": baseline.get("source"),
        "candidate_source": candidate.get("source"),
        "cells": cell_verdicts,
    }
