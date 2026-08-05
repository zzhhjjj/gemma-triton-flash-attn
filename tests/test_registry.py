from __future__ import annotations

import pytest

from flash_attn.profiles import DEFAULT_MODEL_PROFILES
from flash_attn.registry import (
    AmbiguousKernel,
    AttentionKernelRegistry,
    AttentionSpec,
    ConfigRegistration,
    DEFAULT_REGISTRY,
    ImplementationRegistration,
    KernelConfig,
    NoKernelFound,
    RuntimeSpec,
)


SM100_RUNTIME = RuntimeSpec(
    "sm100",
    gpu_name="NVIDIA B200",
    torch_version="2.11.0+cu130",
    triton_version="3.6.0",
)
SM90_RUNTIME = RuntimeSpec("sm90", gpu_name="NVIDIA H100 80GB HBM3")
H200_RUNTIME = RuntimeSpec("sm90", gpu_name="NVIDIA H200")


@pytest.mark.parametrize(
    "profile_id,head_dim,window_size,q_heads,kv_heads",
    [
        ("gemma4_e2b_text_full", 512, 0, 8, 1),
        ("gemma4_e2b_text_sliding", 256, 512, 8, 1),
        ("gemma4_moe_text_full", 512, 0, 16, 2),
        ("gemma4_moe_text_sliding", 256, 1024, 16, 8),
    ],
)
def test_model_profiles_expand_to_complete_semantics(
    profile_id: str,
    head_dim: int,
    window_size: int,
    q_heads: int,
    kv_heads: int,
) -> None:
    spec = DEFAULT_MODEL_PROFILES.get(profile_id).make_spec(
        dtype="torch.bfloat16", training=True
    )
    assert (spec.head_dim, spec.window_size) == (head_dim, window_size)
    assert (spec.q_heads, spec.kv_heads) == (q_heads, kv_heads)
    assert spec.dtype == "bfloat16"
    assert spec.causal


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
@pytest.mark.parametrize(
    "profile_id,expected_tile",
    [
        ("gemma4_e2b_text_full", (32, 32, 512, 4, 2)),
        ("gemma4_e2b_text_sliding", (64, 64, 256, 4, 2)),
        ("gemma4_moe_text_full", (32, 32, 512, 4, 2)),
        ("gemma4_moe_text_sliding", (64, 64, 256, 4, 2)),
    ],
)
def test_resolution_is_gpu_and_semantics_aware(
    arch: str, profile_id: str, expected_tile: tuple[int, ...]
) -> None:
    spec = DEFAULT_MODEL_PROFILES.get(profile_id).make_spec(
        dtype="bf16", training=True
    )
    runtime = RuntimeSpec(
        arch,
        gpu_name="NVIDIA B200" if arch == "sm100" else "NVIDIA H100 80GB HBM3",
        torch_version="2.11.0+cu130" if arch == "sm100" else "unknown",
        triton_version="3.6.0" if arch == "sm100" else "unknown",
    )
    result = DEFAULT_REGISTRY.resolve(spec, runtime)
    assert result.implementation.id == "triton_gqa_batched_v1"
    assert result.config_registration.id.endswith(f".{arch}.d{spec.head_dim}")
    assert (
        result.config.block_q,
        result.config.block_kv,
        result.config.block_d,
        result.config.num_warps,
        result.config.num_stages,
    ) == expected_tile
    assert result.config_registration.evidence_status == (
        "verified" if arch == "sm100" else "baseline"
    )


def test_layout_selects_implementation_not_model_name() -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
        dtype="fp16", training=True, layout="thd"
    )
    result = DEFAULT_REGISTRY.resolve(
        spec,
        RuntimeSpec(
            "100",
            gpu_name="NVIDIA B200",
            torch_version="2.11.0",
            triton_version="3.6.0",
        ),
    )
    assert result.implementation.id == "triton_gqa_varlen_v1"


@pytest.mark.parametrize("spelling", ["100", "sm100", "sm_100", "10.0", "compute_100"])
def test_runtime_arch_spellings_normalize_to_sm100(spelling: str) -> None:
    assert RuntimeSpec(spelling).gpu_arch == "sm100"


def test_verified_config_requires_its_recorded_software_family() -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
        dtype="bf16", training=True
    )
    with pytest.raises(NoKernelFound, match="triton_version=unknown"):
        DEFAULT_REGISTRY.resolve(spec, RuntimeSpec("sm100"))


@pytest.mark.parametrize(
    "profile_id,sm90_tile,sm100_tile",
    [
        ("gemma4_e2b_text_full", (64, 32, 8), (32, 32, 4)),
        ("gemma4_e2b_text_sliding", (128, 64, 8), (64, 64, 4)),
    ],
)
def test_sm90_inference_config_is_preserved_separately_from_sm100(
    profile_id: str,
    sm90_tile: tuple[int, int, int],
    sm100_tile: tuple[int, int, int],
) -> None:
    spec = DEFAULT_MODEL_PROFILES.get(profile_id).make_spec(
        dtype="bf16", training=False, query_length=129
    )
    sm90 = DEFAULT_REGISTRY.resolve(spec, SM90_RUNTIME)
    sm100 = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME)
    assert sm90.config_registration.id.endswith(
        f"forward_h100_tuned.sm90.d{spec.head_dim}"
    )
    assert sm90.config_registration.config_kind == "tuned_override"
    assert (sm90.config.block_q, sm90.config.block_kv, sm90.config.num_warps) == sm90_tile
    assert sm100.config_registration.id.endswith(f"forward.sm100.d{spec.head_dim}")
    assert sm100.config_registration.config_kind == "base"
    assert (sm100.config.block_q, sm100.config.block_kv, sm100.config.num_warps) == sm100_tile


@pytest.mark.parametrize("gpu_name", ["NVIDIA H200", "unknown"])
def test_other_sm90_gpu_uses_compile_safe_base(gpu_name: str) -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
        dtype="bf16", training=False, query_length=129
    )
    result = DEFAULT_REGISTRY.resolve(spec, RuntimeSpec("sm90", gpu_name=gpu_name))
    assert result.config_registration.id.endswith("forward.sm90.d512")
    assert result.config_registration.config_kind == "base"
    assert (result.config.block_q, result.config.block_kv, result.config.num_warps) == (
        32,
        32,
        4,
    )


@pytest.mark.parametrize("layout", ["bhsd", "thd"])
@pytest.mark.parametrize(
    "profile_id,role,expected_tile",
    [
        ("gemma4_e2b_text_full", "forward", (32, 32, 4, 2)),
        ("gemma4_e2b_text_full", "backward_dq", (32, 64, 8, 2)),
        ("gemma4_e2b_text_full", "backward_dkv", (64, 16, 4, 2)),
        ("gemma4_e2b_text_sliding", "forward", (64, 64, 4, 2)),
        ("gemma4_e2b_text_sliding", "backward_dq", (64, 64, 4, 2)),
        ("gemma4_e2b_text_sliding", "backward_dkv", (128, 64, 8, 1)),
    ],
)
def test_h200_training_roles_keep_sm90_compile_safe_base(
    layout: str,
    profile_id: str,
    role: str,
    expected_tile: tuple[int, int, int, int],
) -> None:
    spec = DEFAULT_MODEL_PROFILES.get(profile_id).make_spec(
        dtype="bf16", training=True, query_length=129, layout=layout
    )
    result = DEFAULT_REGISTRY.resolve(spec, H200_RUNTIME, role=role)
    assert result.config_registration.id.endswith(
        f".{role}.sm90.d{spec.head_dim}"
    ) or f".{role}_" in result.config_registration.id
    assert result.config_registration.config_kind == "base"
    assert result.config_registration.evidence_status == "baseline"
    assert "h100_tuned" not in result.config_registration.id
    assert "b200_tuned" not in result.config_registration.id
    assert (
        result.config.block_q,
        result.config.block_kv,
        result.config.num_warps,
        result.config.num_stages,
    ) == expected_tile


@pytest.mark.parametrize(
    "profile_ids",
    [
        ("gemma4_e2b_text_full", "gemma4_moe_text_full"),
        ("gemma4_e2b_text_sliding", "gemma4_moe_text_sliding"),
    ],
)
def test_model_name_is_not_a_hardware_dispatch_key(
    profile_ids: tuple[str, str]
) -> None:
    config_ids = {
        DEFAULT_REGISTRY.resolve(
            DEFAULT_MODEL_PROFILES.get(profile_id).make_spec(
                dtype="bf16", training=True, query_length=129
            ),
            SM100_RUNTIME,
        ).config_registration.id
        for profile_id in profile_ids
    }
    assert len(config_ids) == 1


@pytest.mark.parametrize("profile_id", ["gemma4_e2b_text_full", "gemma4_e2b_text_sliding"])
@pytest.mark.parametrize("role", ["forward", "backward_dq", "backward_dkv"])
def test_training_roles_have_independent_h100_and_b200_records(
    profile_id: str, role: str
) -> None:
    spec = DEFAULT_MODEL_PROFILES.get(profile_id).make_spec(
        dtype="bf16", training=True, query_length=129, layout="thd"
    )
    sm90 = DEFAULT_REGISTRY.resolve(spec, SM90_RUNTIME, role=role)
    sm100 = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME, role=role)
    assert ".sm90." in sm90.config_registration.id
    assert ".sm100." in sm100.config_registration.id
    assert sm90.config_registration.id != sm100.config_registration.id
    if profile_id == "gemma4_e2b_text_full" and role == "backward_dkv":
        assert "backward_dkv_b200_tuned" in sm100.config_registration.id
        assert (sm100.config.block_q, sm100.config.block_kv) == (16, 16)
        assert sm100.config.q_splits == 1
        assert sm90.config != sm100.config
    else:
        assert sm90.config == sm100.config


def test_backward_roles_resolve_independently() -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
        dtype="bf16", training=True, query_length=129, layout="thd"
    )
    dq = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME, role="backward_dq")
    dkv = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME, role="backward_dkv")
    assert (dq.config.block_q, dq.config.block_kv, dq.config.num_warps) == (32, 64, 8)
    assert (dkv.config.block_q, dkv.config.block_kv, dkv.config.num_warps) == (
        16,
        16,
        4,
    )
    assert dkv.config.q_splits == 1


def test_d512_b200_tuning_keeps_unknown_sm100_safe_base() -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
        dtype="bf16", training=True, query_length=129, layout="thd"
    )
    result = DEFAULT_REGISTRY.resolve(
        spec,
        RuntimeSpec(
            "sm100",
            gpu_name="unknown",
            torch_version="2.11.0+cu130",
            triton_version="3.6.0",
        ),
        role="backward_dkv",
    )
    assert result.config_registration.id.endswith("backward_dkv.sm100.d512")
    assert (result.config.block_q, result.config.block_kv) == (64, 16)
    assert result.config.q_splits == 8


@pytest.mark.parametrize(
    "sequence_length,variant,expected_q_splits",
    [
        (1024, "backward_dkv_big", 8),
        (2048, "backward_dkv_small", 4),
        (8192, "backward_dkv_big", 1),
    ],
)
def test_dkv_grid_policy_is_registry_owned(
    sequence_length: int, variant: str, expected_q_splits: int
) -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_sliding").make_spec(
        dtype="bf16", training=True, query_length=sequence_length
    )
    result = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME, role="backward_dkv")
    assert variant in result.config_registration.id
    assert result.config.q_splits == expected_q_splits


def test_image_group_profile_selects_only_capable_implementation() -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_moe_image_group_sliding").make_spec(
        dtype="bf16", training=True
    )
    result = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME)
    assert result.implementation.supports_image_groups


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"dtype": "float32"}, "dtype=float32"),
        ({"dropout_p": 0.1}, "dropout_p=0.1"),
        ({"softcap": 30.0}, "softcap=30.0"),
    ],
)
def test_unsupported_semantics_fail_strictly_with_reason(
    changes: dict[str, object], reason: str
) -> None:
    fields = dict(
        q_heads=8,
        kv_heads=1,
        head_dim=512,
        dtype="bf16",
        causal=True,
    )
    fields.update(changes)
    with pytest.raises(NoKernelFound, match=reason):
        DEFAULT_REGISTRY.resolve(AttentionSpec(**fields), RuntimeSpec("sm100"))


def test_explanation_is_machine_readable_and_contains_rejections() -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_vision_full").make_spec(
        dtype="bf16", training=False
    )
    result = DEFAULT_REGISTRY.resolve(
        spec,
        RuntimeSpec(
            "sm100",
            gpu_name="NVIDIA B200",
            torch_version="2.11.0+cu130",
            triton_version="3.6.0",
        ),
    )
    explanation = result.to_dict()
    assert explanation["implementation"] == "triton_gqa_batched_v1"
    assert explanation["config_kind"] == "base"
    rejected = {
        item["id"]: item["reasons"]
        for item in explanation["implementation_candidates"]
        if not item["accepted"]
    }
    assert any("layout=bhsd" in reason for reason in rejected["triton_gqa_varlen_v1"])


def test_resolution_is_cached_after_first_explanation() -> None:
    spec = DEFAULT_MODEL_PROFILES.get("gemma4_e2b_text_full").make_spec(
        dtype="bf16", training=True, query_length=129
    )
    first = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME)
    second = DEFAULT_REGISTRY.resolve(spec, SM100_RUNTIME)
    assert second is first


def _minimal_registry(order: tuple[str, str]) -> AttentionKernelRegistry:
    registry = AttentionKernelRegistry()
    registrations = {
        name: ImplementationRegistration(
            id=name,
            layouts=frozenset({"bhsd"}),
            gpu_arches=frozenset({"sm100"}),
            dtypes=frozenset({"bfloat16"}),
            head_dims=frozenset({256}),
            supports_training=True,
            supports_causal=True,
            supports_noncausal=False,
            supports_sliding=True,
            supports_image_groups=False,
        )
        for name in order
    }
    for name in order:
        registry.register_implementation(registrations[name])
        registry.register_config(
            ConfigRegistration(
                id=f"{name}.config",
                implementation_id=name,
                gpu_arches=frozenset({"sm100"}),
                head_dims=frozenset({256}),
                config=KernelConfig(64, 64, 256, 4, 2),
                evidence_status="baseline",
                evidence="unit test",
            )
        )
    return registry


@pytest.mark.parametrize("order", [("candidate_a", "candidate_b"), ("candidate_b", "candidate_a")])
def test_equal_priority_ambiguity_never_depends_on_import_order(
    order: tuple[str, str]
) -> None:
    spec = AttentionSpec(8, 1, 256, "bf16", causal=True, window_size=128)
    with pytest.raises(AmbiguousKernel, match="candidate_a, candidate_b"):
        _minimal_registry(order).resolve(spec, RuntimeSpec("sm100"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_heads": 7, "kv_heads": 2},
        {"q_heads": 8, "kv_heads": 1, "window_size": 1, "causal": False},
        {"q_heads": 8, "kv_heads": 1, "layout": "thd", "image_groups": True},
        {"q_heads": 8, "kv_heads": 1, "image_groups": True, "causal": False},
    ],
)
def test_invalid_specs_fail_before_resolution(kwargs: dict[str, object]) -> None:
    fields = dict(q_heads=8, kv_heads=1, head_dim=256, dtype="bf16", causal=True)
    fields.update(kwargs)
    with pytest.raises(ValueError):
        AttentionSpec(**fields)
