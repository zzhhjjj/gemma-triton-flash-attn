from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

import flash_attn.hf_integration as hf_integration
from flash_attn.profiles import DEFAULT_MODEL_PROFILES
from flash_attn.registry import DEFAULT_REGISTRY, RuntimeSpec
from flash_attn.telemetry import (
    AttentionSelectionTelemetry,
    capture_attention_selection,
    record_attention_selection,
)


SM100_RUNTIME = RuntimeSpec(
    "sm100",
    gpu_name="NVIDIA B200",
    torch_version="2.11.0+cu130",
    triton_version="3.6.0",
)
attention_module = importlib.import_module("flash_attn.attention")


def _full_resolution():
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
        dtype="bf16", training=True, query_length=129
    )
    return spec, DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME)


def test_summary_aggregates_selection_counts_and_is_json_serializable() -> None:
    spec, resolution = _full_resolution()
    with capture_attention_selection(
        labels={"model_profile": "gemma4_e2b"}
    ) as telemetry:
        record_attention_selection(spec, SM100_RUNTIME, resolution)
        record_attention_selection(spec, SM100_RUNTIME, resolution)

    snapshot = telemetry.snapshot()
    assert snapshot["labels"] == {"model_profile": "gemma4_e2b"}
    assert snapshot["total_selections"] == 2
    assert snapshot["total_fallbacks"] == 0
    assert len(snapshot["selections"]) == 1
    assert snapshot["selections"][0]["count"] == 2
    assert snapshot["selections"][0]["config_kind"] == "base"
    assert snapshot["selections"][0]["separate_dkv_scratch"] is False
    assert snapshot["selections"][0]["relaxed_dkv_atomics"] is False
    assert snapshot["selections"][0]["split_gqa_heads"] is False
    assert snapshot["selections"][0]["bf16x2_dkv_atomics"] is False
    assert "debug_events" not in snapshot
    assert json.loads(telemetry.to_json()) == snapshot
    assert "forward.sm100.d512" in telemetry.format_summary()


def test_debug_retains_one_full_explanation_per_distinct_selection() -> None:
    spec, resolution = _full_resolution()
    with capture_attention_selection("debug") as telemetry:
        record_attention_selection(spec, SM100_RUNTIME, resolution)
        record_attention_selection(spec, SM100_RUNTIME, resolution)

    snapshot = telemetry.snapshot()
    assert snapshot["total_selections"] == 2
    assert len(snapshot["debug_events"]) == 1
    event = snapshot["debug_events"][0]
    assert event["spec"]["head_dim"] == 512
    assert event["runtime"]["gpu_name"] == "NVIDIA B200"
    assert event["resolution"]["config_kind"] == "base"
    assert event["resolution"]["separate_dkv_scratch"] is False
    assert event["resolution"]["relaxed_dkv_atomics"] is False
    assert event["resolution"]["split_gqa_heads"] is False
    assert event["resolution"]["bf16x2_dkv_atomics"] is False
    assert event["resolution"]["implementation_candidates"]


def test_summary_distinguishes_materialized_q_split_configs() -> None:
    specs = [
        DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
            dtype="bf16", training=True, query_length=query_length
        )
        for query_length in (129, 8192)
    ]
    resolutions = [
        DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME, role="backward_dkv")
        for spec in specs
    ]
    assert {resolution.config.q_splits for resolution in resolutions} == {1, 8}

    with capture_attention_selection() as telemetry:
        for spec, resolution in zip(specs, resolutions):
            record_attention_selection(spec, SM100_RUNTIME, resolution)

    rows = telemetry.snapshot()["selections"]
    assert len(rows) == 2
    assert {row["q_splits"] for row in rows} == {1, 8}


def test_summary_records_b200_qsplit_scratch_policy() -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
        dtype="bf16", training=True, query_length=8192, layout="thd"
    )
    resolution = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME, role="backward_dkv")
    assert resolution.config_registration.separate_dkv_scratch

    with capture_attention_selection() as telemetry:
        record_attention_selection(spec, SM100_RUNTIME, resolution)

    row = telemetry.snapshot()["selections"][0]
    assert row["q_splits"] == 1
    assert row["separate_dkv_scratch"] is True
    assert row["relaxed_dkv_atomics"] is True
    assert row["split_gqa_heads"] is True
    assert row["bf16x2_dkv_atomics"] is True


def test_debug_distinguishes_exact_shapes_with_the_same_launch_config() -> None:
    specs = [
        DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
            dtype="bf16", training=True, query_length=query_length
        )
        for query_length in (129, 130)
    ]
    resolutions = [DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME) for spec in specs]
    assert resolutions[0].config == resolutions[1].config

    with capture_attention_selection("debug") as telemetry:
        for spec, resolution in zip(specs, resolutions):
            record_attention_selection(spec, SM100_RUNTIME, resolution)

    events = telemetry.snapshot()["debug_events"]
    assert {event["spec"]["query_length"] for event in events} == {129, 130}


def test_nested_capture_contexts_are_isolated() -> None:
    spec, resolution = _full_resolution()
    with capture_attention_selection() as outer:
        record_attention_selection(spec, SM100_RUNTIME, resolution)
        with capture_attention_selection() as inner:
            record_attention_selection(spec, SM100_RUNTIME, resolution)
            record_attention_selection(spec, SM100_RUNTIME, resolution)
        record_attention_selection(spec, SM100_RUNTIME, resolution)

    assert outer.snapshot()["total_selections"] == 2
    assert inner.snapshot()["total_selections"] == 2


def test_resolution_hook_records_without_launching_a_kernel(monkeypatch) -> None:
    monkeypatch.setattr(
        attention_module.RuntimeSpec,
        "from_torch_device",
        classmethod(lambda cls, device: SM100_RUNTIME),
    )
    q = torch.empty(1, 8, 129, 512, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 129, 512, dtype=torch.bfloat16)

    with capture_attention_selection() as telemetry:
        resolution = attention_module._resolve_attention_config(
            q,
            k,
            causal=True,
            window_size=0,
            layout="bhsd",
            training=True,
            role="forward",
            batch_size=1,
            query_length=129,
            key_length=129,
        )

    assert resolution.config_registration.id.endswith("forward.sm100.d512")
    assert telemetry.snapshot()["total_selections"] == 1


def test_varlen_to_batched_adapter_route_is_an_explicit_fallback(
    monkeypatch,
) -> None:
    expected = (object(), None)

    def fake_batched(*args, **kwargs):
        return expected

    monkeypatch.setattr(hf_integration, "triton_gqa_attention", fake_batched)
    monkeypatch.setitem(hf_integration._varlen_cu_seqlens_state, "value", None)
    tensor = torch.empty(1, 2, 3, 4)

    with capture_attention_selection() as telemetry:
        actual = hf_integration.triton_gqa_varlen_attention(
            SimpleNamespace(head_dim=4, is_causal=True),
            tensor,
            tensor,
            tensor,
            attention_mask=None,
        )

    assert actual is expected
    snapshot = telemetry.snapshot()
    assert snapshot["total_fallbacks"] == 1
    assert snapshot["fallbacks"] == [
        {
            "count": 1,
            "source": "triton_gqa_varlen",
            "target": "triton_gqa",
            "reason": "no_varlen_metadata",
        }
    ]


def test_unsupported_telemetry_mode_fails_early() -> None:
    with pytest.raises(ValueError, match="unsupported telemetry mode"):
        AttentionSelectionTelemetry("verbose")
