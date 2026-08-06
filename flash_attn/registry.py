"""Pure-Python attention capability and kernel-config registry.

This module deliberately does not import Triton kernels. It describes attention
semantics, runtime identity, implementation capabilities, and launch configs so
selection can be tested on CPU and explained before any GPU launch.

Config selection is layered: every supported architecture has a compile-safe
``base`` record, while evidence-backed product tuning is an optional
higher-priority ``tuned_override``. Product-name matching must never be needed
to find the architecture base.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import math
from typing import Iterable, Literal


Layout = Literal["bhsd", "thd"]
EvidenceStatus = Literal["baseline", "verified"]
KernelRole = Literal["forward", "backward_dq", "backward_dkv"]
ConfigKind = Literal["base", "tuned_override"]


class RegistryError(RuntimeError):
    """Base class for deterministic registry failures."""


class NoKernelFound(RegistryError):
    """No implementation or config satisfies the requested contract."""


class AmbiguousKernel(RegistryError):
    """Multiple equally preferred registrations satisfy the contract."""


@dataclass(frozen=True)
class AttentionSpec:
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    causal: bool
    window_size: int = 0
    layout: Layout = "bhsd"
    training: bool = True
    image_groups: bool = False
    dropout_p: float = 0.0
    softcap: float | None = None
    batch_size: int = 1
    query_length: int = 1
    key_length: int | None = None

    def __post_init__(self) -> None:
        dtype = str(self.dtype).removeprefix("torch.").lower()
        aliases = {"fp16": "float16", "half": "float16", "bf16": "bfloat16"}
        object.__setattr__(self, "dtype", aliases.get(dtype, dtype))
        if self.key_length is None:
            object.__setattr__(self, "key_length", self.query_length)
        if self.q_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("q_heads and kv_heads must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported dtype spelling: {self.dtype}")
        if self.layout not in {"bhsd", "thd"}:
            raise ValueError(f"unsupported layout: {self.layout}")
        if self.window_size < 0:
            raise ValueError("window_size must be non-negative")
        if self.window_size and not self.causal:
            raise ValueError("sliding-window attention currently requires causal=True")
        if self.image_groups and self.layout != "bhsd":
            raise ValueError("image-group masking is only defined for batched layout")
        if self.image_groups and not self.causal:
            raise ValueError("image-group masking currently requires causal attention")
        if not math.isfinite(self.dropout_p) or self.dropout_p < 0.0 or self.dropout_p >= 1.0:
            raise ValueError("dropout_p must be in [0, 1)")
        if self.softcap is not None and (
            not math.isfinite(self.softcap) or self.softcap <= 0
        ):
            raise ValueError("softcap must be positive when provided")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.query_length <= 0 or self.key_length is None or self.key_length <= 0:
            raise ValueError("query_length and key_length must be positive")

    @property
    def is_sliding(self) -> bool:
        return self.window_size > 0


@dataclass(frozen=True)
class RuntimeSpec:
    gpu_arch: str
    gpu_name: str = "unknown"
    torch_version: str = "unknown"
    triton_version: str = "unknown"

    def __post_init__(self) -> None:
        arch = str(self.gpu_arch).lower().replace("compute_", "sm").replace("sm_", "sm")
        if arch.count(".") == 1 and all(part.isdigit() for part in arch.split(".")):
            major, minor = arch.split(".")
            arch = f"sm{major}{minor}"
        if arch.isdigit():
            arch = f"sm{arch}"
        object.__setattr__(self, "gpu_arch", arch)
        if not arch.startswith("sm") or not arch[2:].isdigit():
            raise ValueError(f"invalid gpu_arch: {self.gpu_arch}")

    @classmethod
    def from_torch_device(cls, device: object) -> "RuntimeSpec":
        """Detect runtime identity lazily so CPU-only registry tests stay pure."""
        import torch

        resolved = torch.device(device)
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        return _runtime_from_cuda_index(index)


@lru_cache(maxsize=None)
def _runtime_from_cuda_index(index: int) -> RuntimeSpec:
    import torch
    import triton

    major, minor = torch.cuda.get_device_capability(index)
    properties = torch.cuda.get_device_properties(index)
    return RuntimeSpec(
        gpu_arch=f"sm{major}{minor}",
        gpu_name=properties.name,
        torch_version=torch.__version__,
        triton_version=triton.__version__,
    )


@dataclass(frozen=True)
class KernelConfig:
    block_q: int
    block_kv: int
    block_d: int
    num_warps: int
    num_stages: int
    q_splits: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ImplementationRegistration:
    id: str
    layouts: frozenset[Layout]
    gpu_arches: frozenset[str]
    dtypes: frozenset[str]
    head_dims: frozenset[int]
    supports_training: bool
    supports_causal: bool
    supports_noncausal: bool
    supports_sliding: bool
    supports_image_groups: bool
    priority: int = 0

    def rejection_reasons(
        self, spec: AttentionSpec, runtime: RuntimeSpec
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if spec.layout not in self.layouts:
            reasons.append(f"layout={spec.layout} not in {sorted(self.layouts)}")
        if runtime.gpu_arch not in self.gpu_arches:
            reasons.append(f"gpu_arch={runtime.gpu_arch} not in {sorted(self.gpu_arches)}")
        if spec.dtype not in self.dtypes:
            reasons.append(f"dtype={spec.dtype} not in {sorted(self.dtypes)}")
        if spec.head_dim not in self.head_dims:
            reasons.append(f"head_dim={spec.head_dim} not in {sorted(self.head_dims)}")
        if spec.training and not self.supports_training:
            reasons.append("training=True is unsupported")
        if spec.causal and not self.supports_causal:
            reasons.append("causal=True is unsupported")
        if not spec.causal and not self.supports_noncausal:
            reasons.append("causal=False is unsupported")
        if spec.is_sliding and not self.supports_sliding:
            reasons.append("sliding-window masking is unsupported")
        if spec.image_groups and not self.supports_image_groups:
            reasons.append("image-group masking is unsupported")
        if spec.dropout_p:
            reasons.append(f"dropout_p={spec.dropout_p} is unsupported")
        if spec.softcap is not None:
            reasons.append(f"softcap={spec.softcap} is unsupported")
        return tuple(reasons)


@dataclass(frozen=True)
class ConfigRegistration:
    id: str
    implementation_id: str
    gpu_arches: frozenset[str]
    head_dims: frozenset[int]
    config: KernelConfig
    evidence_status: EvidenceStatus
    evidence: str
    config_kind: ConfigKind = "base"
    role: KernelRole = "forward"
    priority: int = 0
    dtypes: frozenset[str] = frozenset()
    training_modes: frozenset[bool] = frozenset({False, True})
    gpu_name_patterns: frozenset[str] = frozenset()
    torch_version_prefixes: frozenset[str] = frozenset()
    triton_version_prefixes: frozenset[str] = frozenset()
    batch_sizes: frozenset[int] = frozenset()
    q_head_counts: frozenset[int] = frozenset()
    kv_head_counts: frozenset[int] = frozenset()
    window_sizes: frozenset[int] = frozenset()
    min_query_length: int | None = None
    grid_probe_block_kv: int | None = None
    grid_ranges: tuple[tuple[int | None, int | None], ...] = ()
    q_split_target: int | None = None
    separate_dkv_scratch: bool = False
    relaxed_dkv_atomics: bool = False
    split_gqa_heads: bool = False
    bf16x2_dkv_atomics: bool = False

    def rejection_reasons(
        self,
        implementation_id: str,
        role: KernelRole,
        spec: AttentionSpec,
        runtime: RuntimeSpec,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if implementation_id != self.implementation_id:
            reasons.append(f"implementation={implementation_id} != {self.implementation_id}")
        if role != self.role:
            reasons.append(f"role={role} != {self.role}")
        if runtime.gpu_arch not in self.gpu_arches:
            reasons.append(f"gpu_arch={runtime.gpu_arch} not in {sorted(self.gpu_arches)}")
        if spec.head_dim not in self.head_dims:
            reasons.append(f"head_dim={spec.head_dim} not in {sorted(self.head_dims)}")
        if self.dtypes and spec.dtype not in self.dtypes:
            reasons.append(f"dtype={spec.dtype} not in {sorted(self.dtypes)}")
        if spec.training not in self.training_modes:
            reasons.append(f"training={spec.training} not in {sorted(self.training_modes)}")
        if self.gpu_name_patterns and not any(
            pattern.lower() in runtime.gpu_name.lower()
            for pattern in self.gpu_name_patterns
        ):
            reasons.append(
                f"gpu_name={runtime.gpu_name!r} does not match "
                f"{sorted(self.gpu_name_patterns)}"
            )
        if self.torch_version_prefixes and not self._matches_version(
            runtime.torch_version, self.torch_version_prefixes
        ):
            reasons.append(
                f"torch_version={runtime.torch_version} does not match "
                f"{sorted(self.torch_version_prefixes)}"
            )
        if self.triton_version_prefixes and not self._matches_version(
            runtime.triton_version, self.triton_version_prefixes
        ):
            reasons.append(
                f"triton_version={runtime.triton_version} does not match "
                f"{sorted(self.triton_version_prefixes)}"
            )
        if self.batch_sizes and spec.batch_size not in self.batch_sizes:
            reasons.append(f"batch_size={spec.batch_size} not in {sorted(self.batch_sizes)}")
        if self.q_head_counts and spec.q_heads not in self.q_head_counts:
            reasons.append(f"q_heads={spec.q_heads} not in {sorted(self.q_head_counts)}")
        if self.kv_head_counts and spec.kv_heads not in self.kv_head_counts:
            reasons.append(f"kv_heads={spec.kv_heads} not in {sorted(self.kv_head_counts)}")
        if self.window_sizes and spec.window_size not in self.window_sizes:
            reasons.append(f"window_size={spec.window_size} not in {sorted(self.window_sizes)}")
        if self.min_query_length is not None and spec.query_length < self.min_query_length:
            reasons.append(
                f"query_length={spec.query_length} < min_query_length={self.min_query_length}"
            )
        if self.grid_ranges:
            probe_block = self.grid_probe_block_kv or self.config.block_kv
            raw_grid = self.grid_size(spec, probe_block)
            if not any(
                (low is None or raw_grid >= low) and (high is None or raw_grid < high)
                for low, high in self.grid_ranges
            ):
                reasons.append(
                    f"raw_grid@BKV{probe_block}={raw_grid} not in {self.grid_ranges}"
                )
        return tuple(reasons)

    @staticmethod
    def _matches_version(version: str, prefixes: frozenset[str]) -> bool:
        return any(str(version).startswith(prefix) for prefix in prefixes)

    @staticmethod
    def grid_size(spec: AttentionSpec, block_kv: int) -> int:
        assert spec.key_length is not None
        effective_block = min(block_kv, 1 << (spec.key_length - 1).bit_length())
        return (
            math.ceil(spec.key_length / effective_block)
            * spec.batch_size
            * spec.kv_heads
        )

    def materialize(self, spec: AttentionSpec) -> KernelConfig:
        if self.q_split_target is None:
            return self.config
        raw_grid = self.grid_size(spec, self.config.block_kv)
        q_splits = 1
        while q_splits < 8 and raw_grid * q_splits < self.q_split_target:
            q_splits *= 2
        return replace(self.config, q_splits=q_splits)


@dataclass(frozen=True)
class CandidateExplanation:
    id: str
    accepted: bool
    priority: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Resolution:
    role: KernelRole
    implementation: ImplementationRegistration
    config_registration: ConfigRegistration
    resolved_config: KernelConfig
    implementation_candidates: tuple[CandidateExplanation, ...]
    config_candidates: tuple[CandidateExplanation, ...]

    @property
    def config(self) -> KernelConfig:
        return self.resolved_config

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "implementation": self.implementation.id,
            "config_id": self.config_registration.id,
            "config": asdict(self.config),
            "config_kind": self.config_registration.config_kind,
            "separate_dkv_scratch": self.config_registration.separate_dkv_scratch,
            "relaxed_dkv_atomics": self.config_registration.relaxed_dkv_atomics,
            "split_gqa_heads": self.config_registration.split_gqa_heads,
            "bf16x2_dkv_atomics": self.config_registration.bf16x2_dkv_atomics,
            "evidence_status": self.config_registration.evidence_status,
            "evidence": self.config_registration.evidence,
            "implementation_candidates": [asdict(item) for item in self.implementation_candidates],
            "config_candidates": [asdict(item) for item in self.config_candidates],
        }


def _choose_unique_highest(
    registrations: Iterable[object], *, kind: str
) -> object:
    registrations = tuple(registrations)
    if not registrations:
        raise NoKernelFound(f"no matching {kind}")
    max_priority = max(getattr(item, "priority") for item in registrations)
    winners = tuple(item for item in registrations if getattr(item, "priority") == max_priority)
    if len(winners) != 1:
        ids = sorted(getattr(item, "id") for item in winners)
        raise AmbiguousKernel(
            f"ambiguous {kind} at priority {max_priority}: {', '.join(ids)}"
        )
    return winners[0]


class AttentionKernelRegistry:
    def __init__(self) -> None:
        self._implementations: dict[str, ImplementationRegistration] = {}
        self._configs: dict[str, ConfigRegistration] = {}
        self._resolution_cache: dict[
            tuple[AttentionSpec, RuntimeSpec, KernelRole], Resolution
        ] = {}

    def register_implementation(self, registration: ImplementationRegistration) -> None:
        if registration.id in self._implementations:
            raise RegistryError(f"duplicate implementation id: {registration.id}")
        self._implementations[registration.id] = registration
        self._resolution_cache.clear()

    def register_config(self, registration: ConfigRegistration) -> None:
        if registration.id in self._configs:
            raise RegistryError(f"duplicate config id: {registration.id}")
        if registration.implementation_id not in self._implementations:
            raise RegistryError(
                f"config references unknown implementation: {registration.implementation_id}"
            )
        self._configs[registration.id] = registration
        self._resolution_cache.clear()

    def resolve(
        self,
        spec: AttentionSpec,
        runtime: RuntimeSpec,
        *,
        role: KernelRole = "forward",
    ) -> Resolution:
        cache_key = (spec, runtime, role)
        cached = self._resolution_cache.get(cache_key)
        if cached is not None:
            return cached
        impl_explanations = tuple(
            CandidateExplanation(
                id=item.id,
                accepted=not (reasons := item.rejection_reasons(spec, runtime)),
                priority=item.priority,
                reasons=reasons,
            )
            for item in sorted(self._implementations.values(), key=lambda item: item.id)
        )
        matching_impls = tuple(
            self._implementations[item.id] for item in impl_explanations if item.accepted
        )
        try:
            implementation = _choose_unique_highest(
                matching_impls, kind="kernel implementation"
            )
        except NoKernelFound as error:
            rejected = "; ".join(
                f"{item.id}: {', '.join(item.reasons)}" for item in impl_explanations
            )
            raise NoKernelFound(f"{error}; {rejected}") from None

        config_explanations = tuple(
            CandidateExplanation(
                id=item.id,
                accepted=not (
                    reasons := item.rejection_reasons(implementation.id, role, spec, runtime)
                ),
                priority=item.priority,
                reasons=reasons,
            )
            for item in sorted(self._configs.values(), key=lambda item: item.id)
        )
        matching_configs = tuple(
            self._configs[item.id] for item in config_explanations if item.accepted
        )
        try:
            config = _choose_unique_highest(matching_configs, kind="kernel config")
        except NoKernelFound as error:
            rejected = "; ".join(
                f"{item.id}: {', '.join(item.reasons)}" for item in config_explanations
            )
            raise NoKernelFound(f"{error}; {rejected}") from None
        resolution = Resolution(
            role=role,
            implementation=implementation,
            config_registration=config,
            resolved_config=config.materialize(spec),
            implementation_candidates=impl_explanations,
            config_candidates=config_explanations,
        )
        self._resolution_cache[cache_key] = resolution
        return resolution


def build_default_registry() -> AttentionKernelRegistry:
    registry = AttentionKernelRegistry()
    common = {
        "gpu_arches": frozenset({"sm90", "sm100"}),
        "dtypes": frozenset({"float16", "bfloat16"}),
        "head_dims": frozenset({64, 128, 256, 512}),
        "supports_training": True,
        "supports_causal": True,
        "supports_noncausal": True,
        "supports_sliding": True,
    }
    registry.register_implementation(
        ImplementationRegistration(
            id="triton_gqa_batched_v1",
            layouts=frozenset({"bhsd"}),
            supports_image_groups=True,
            **common,
        )
    )
    registry.register_implementation(
        ImplementationRegistration(
            id="triton_gqa_varlen_v1",
            layouts=frozenset({"thd"}),
            supports_image_groups=False,
            **common,
        )
    )

    def add_config(
        *,
        registration_id: str,
        implementation_id: str,
        arch: str,
        role: KernelRole,
        head_dim: int,
        config: KernelConfig,
        training_modes: frozenset[bool],
        config_kind: ConfigKind = "base",
        priority: int = 0,
        dtypes: frozenset[str] = frozenset(),
        gpu_name_patterns: frozenset[str] = frozenset(),
        evidence_status: EvidenceStatus | None = None,
        evidence: str | None = None,
        grid_ranges: tuple[tuple[int | None, int | None], ...] = (),
        grid_probe_block_kv: int | None = None,
        q_split_target: int | None = None,
        batch_sizes: frozenset[int] = frozenset(),
        q_head_counts: frozenset[int] = frozenset(),
        kv_head_counts: frozenset[int] = frozenset(),
        window_sizes: frozenset[int] = frozenset(),
        min_query_length: int | None = None,
        separate_dkv_scratch: bool = False,
        relaxed_dkv_atomics: bool = False,
        split_gqa_heads: bool = False,
        bf16x2_dkv_atomics: bool = False,
    ) -> None:
        b200_verified = arch == "sm100" and (
            implementation_id == "triton_gqa_batched_v1" or head_dim != 64
        )
        if evidence_status is None and b200_verified:
            evidence_status = "verified"
            evidence = evidence or "M1/M2 B200 correctness matrix, 2026-07-31 PDT"
        elif evidence_status is None and arch == "sm100":
            evidence_status = "baseline"
            evidence = evidence or "B200 varlen D=64 correctness not run"
        elif evidence_status is None:
            evidence_status = "baseline"
            evidence = evidence or (
                "compile-safe sm90 base from pre-refactor training/varlen wrappers; "
                "cross-SKU recertification pending"
            )
        assert evidence is not None
        registry.register_config(
            ConfigRegistration(
                id=registration_id,
                implementation_id=implementation_id,
                gpu_arches=frozenset({arch}),
                head_dims=frozenset({head_dim}),
                config=config,
                evidence_status=evidence_status,
                evidence=evidence,
                config_kind=config_kind,
                role=role,
                priority=priority,
                dtypes=dtypes,
                training_modes=training_modes,
                gpu_name_patterns=gpu_name_patterns,
                torch_version_prefixes=(
                    frozenset({"2.11"}) if arch == "sm100" else frozenset()
                ),
                triton_version_prefixes=(
                    frozenset({"3.6"}) if arch == "sm100" else frozenset()
                ),
                batch_sizes=batch_sizes,
                q_head_counts=q_head_counts,
                kv_head_counts=kv_head_counts,
                window_sizes=window_sizes,
                min_query_length=min_query_length,
                grid_ranges=grid_ranges,
                grid_probe_block_kv=grid_probe_block_kv,
                q_split_target=q_split_target,
                separate_dkv_scratch=separate_dkv_scratch,
                relaxed_dkv_atomics=relaxed_dkv_atomics,
                split_gqa_heads=split_gqa_heads,
                bf16x2_dkv_atomics=bf16x2_dkv_atomics,
            )
        )

    safe_forward = {
        64: KernelConfig(128, 64, 64, 4, 2),
        128: KernelConfig(128, 64, 128, 4, 2),
        256: KernelConfig(64, 64, 256, 4, 2),
        512: KernelConfig(32, 32, 512, 4, 2),
    }
    # These are deliberately not the sm90 base. They preserve historical H100
    # inference tuning and win only for an actual H100 product-name match.
    h100_inference_overrides = {
        256: KernelConfig(128, 64, 256, 8, 2),
        512: KernelConfig(64, 32, 512, 8, 2),
    }
    backward_dq = {
        64: KernelConfig(64, 64, 64, 4, 2),
        128: KernelConfig(64, 64, 128, 4, 2),
        256: KernelConfig(64, 64, 256, 4, 2),
        512: KernelConfig(32, 64, 512, 8, 2),
    }

    for implementation_id in ("triton_gqa_batched_v1", "triton_gqa_varlen_v1"):
        for arch in ("sm90", "sm100"):
            for head_dim in (64, 128, 256, 512):
                if implementation_id == "triton_gqa_varlen_v1":
                    forward_training_modes = frozenset({True})
                else:
                    forward_training_modes = frozenset({False, True})

                add_config(
                    registration_id=f"{implementation_id}.forward.{arch}.d{head_dim}",
                    implementation_id=implementation_id,
                    arch=arch,
                    role="forward",
                    head_dim=head_dim,
                    config=safe_forward[head_dim],
                    training_modes=forward_training_modes,
                )
                add_config(
                    registration_id=f"{implementation_id}.backward_dq.{arch}.d{head_dim}",
                    implementation_id=implementation_id,
                    arch=arch,
                    role="backward_dq",
                    head_dim=head_dim,
                    config=backward_dq[head_dim],
                    training_modes=frozenset({True}),
                )

                if head_dim == 512:
                    add_config(
                        registration_id=(
                            f"{implementation_id}.backward_dkv.{arch}.d{head_dim}"
                        ),
                        implementation_id=implementation_id,
                        arch=arch,
                        role="backward_dkv",
                        head_dim=head_dim,
                        config=KernelConfig(64, 16, 512, 4, 2),
                        training_modes=frozenset({True}),
                        q_split_target=256,
                    )
                    continue

                add_config(
                    registration_id=(
                        f"{implementation_id}.backward_dkv_big.{arch}.d{head_dim}"
                    ),
                    implementation_id=implementation_id,
                    arch=arch,
                    role="backward_dkv",
                    head_dim=head_dim,
                    config=KernelConfig(128, 64, head_dim, 8, 1),
                    training_modes=frozenset({True}),
                    grid_ranges=((None, 17), (128, None)),
                    grid_probe_block_kv=64,
                    q_split_target=128,
                )
                add_config(
                    registration_id=(
                        f"{implementation_id}.backward_dkv_small.{arch}.d{head_dim}"
                    ),
                    implementation_id=implementation_id,
                    arch=arch,
                    role="backward_dkv",
                    head_dim=head_dim,
                    config=KernelConfig(64, 32, head_dim, 4, 2),
                    training_modes=frozenset({True}),
                    grid_ranges=((17, 128),),
                    grid_probe_block_kv=64,
                    q_split_target=256,
                )

    # B200 varlen D512 dKV: e030 selected BQ16, e031/e032 confirmed it on
    # independent GPUs, and e033 passed the complete E2B/MoE workload family.
    # e063-e066 later selected BKV64 for healthy grids, while a paired 50-repeat
    # run confirmed that raw_grid=448 regresses. e081-e088 and e084 then selected
    # stages3 for that BKV64-only region across 8K-256K and packed distributions.
    # Keep BKV16 as the B200 fallback and the compile-safe sm100 base above for
    # non-B200 products and stacks.
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_tuned.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 16, 512, 4, 2, q_splits=1),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=100,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e030-e033 B200 varlen D512 full F+B: all correctness gates passed; "
            "six-cell family reached 1.956-3.144x same-semantics SDPA, 2026-08-04 PST"
        ),
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_single_qs2.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 4, 3, q_splits=2),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=300,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e116-e125 B200 single-sequence D512 full F+B: qs2 improved "
            "raw-grid 32-63 by 24-37% with correctness passing; packed and "
            "sub-2K workloads remain on prior configs, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((32, 64),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_e2b_single_qs8_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=8),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=600,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e173-e178 B200 E2B full D512: NSYS found dKV at 68.2%; "
            "q8/w8 improved raw-grid 32-105 full F+B by 1.1-37.7% versus "
            "the prior production ranges, with unchanged allocator peak and "
            "correctness passing; raw107+ stays on prior configs, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((32, 106),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_bf16_e2b_headgrid_qs2_w4_bf16x2.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 4, 3, q_splits=2),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=905,
        dtypes=frozenset({"bfloat16"}),
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e319-e321/e328-e329 B200 E2B BF16 full D512: raw-grid "
            "68-105 used "
            "head-grid q2/w4 with BF16x2 atomics; repeated same-card runs "
            "improved throughput 0.7-1.7% and reduced allocator peak "
            "14.1-14.9% versus q3/FP32 scratch; raw64 had a dV max-abs "
            "failure and remains excluded, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        split_gqa_heads=True,
        bf16x2_dkv_atomics=True,
        grid_ranges=((68, 106),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_fp16_e2b_single_qs13_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=13),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=800,
        dtypes=frozenset({"float16"}),
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e222-e227 B200 E2B FP16 full D512: q13 improved raw-grid 32 "
            "full F+B by 2.3-2.4% versus q11 across paired seeds, with "
            "unchanged allocator peak and correctness passing; raw33+ stays "
            "on the prior dtype-agnostic gates, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((32, 33),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_bf16_e2b_headgrid_qs3.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=3),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=900,
        dtypes=frozenset({"bfloat16"}),
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e286-e289 B200 E2B BF16 full D512: mapping GQA heads into the "
            "dKV grid with q3 improved raw-grid 32-105 by 2.4-4.7% versus "
            "the production q14/q11/q9/q8 ranges; 100-repeat paired "
            "correctness and allocator peaks passed, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        split_gqa_heads=True,
        grid_ranges=((32, 106),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_bf16_e2b_headgrid_qs1_bf16x2.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=1),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=910,
        dtypes=frozenset({"bfloat16"}),
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e291-e302/e314 B200 E2B BF16 full D512: head-grid q1 with "
            "relaxed BF16x2 atomics improved raw-grid 106-536 while "
            "matching q1 allocator peaks; versus q4 it also reduced peak "
            "memory about 14%. The corrected same-shape e314 gate removed "
            "the raw-grid 281-316 cliff by 34-35%. Scale16384 and 2K-256K "
            "correctness passed, "
            "2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        split_gqa_heads=True,
        bf16x2_dkv_atomics=True,
        grid_ranges=((106, 537),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_bf16_e2b_single_qs14_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=14),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=850,
        dtypes=frozenset({"bfloat16"}),
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e261-e271 B200 E2B BF16 full D512 after relaxed atomics: q14 "
            "improved raw-grid 32-34 and 45-76 full F+B by 2.1-5.3% "
            "versus the prior q11/q8 gates, with 100-repeat paired coverage, "
            "unchanged allocator peak, and correctness passing; intervening "
            "and raw77+ ranges stay on prior gates, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((32, 35), (45, 77)),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_e2b_single_qs11_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=11),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=700,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e190-e193 B200 E2B full D512: NCU exposed a 1.73-wave tail; "
            "q11 improved raw-grid 32-40 full F+B by 3.0-4.3% versus q8 "
            "with unchanged allocator peak and correctness passing; raw41+ "
            "is handled by later verified q9/q8 gates, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((32, 41),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_e2b_single_qs9_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=9),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=650,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e200-e201 B200 E2B full D512: q9 improved raw-grid 41-44 "
            "full F+B by 2.5-3.8% versus q8 with unchanged allocator peak "
            "and correctness passing; raw45+ stays q8, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((41, 45),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_fp16_moe_single_qs9_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=9),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=600,
        dtypes=frozenset({"float16"}),
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e224-e226 B200 MoE FP16 full D512: q9 improved raw-grid 64-70 "
            "full F+B by 1.8-2.7% versus q8 across paired seeds, with "
            "unchanged allocator peak and correctness passing; raw72+ stays "
            "q8, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({16}),
        kv_head_counts=frozenset({2}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((64, 72),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_bf16_moe_single_qs14_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=14),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=650,
        dtypes=frozenset({"bfloat16"}),
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e262-e270 B200 MoE BF16 full D512 after relaxed atomics: q14 "
            "improved exact raw-grid 64 full F+B by 3.77-3.84% across four "
            "paired GPUs, with unchanged allocator peak and correctness "
            "passing; odd/even tail-wave behavior keeps raw65+ on prior "
            "simple range gates, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({16}),
        kv_head_counts=frozenset({2}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((64, 65),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_moe_single_qs8_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=8),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=500,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e161-e165 B200 MoE full D512: q8 improved raw-grid 64-94 "
            "full F+B by 2.32-4.34% with paired confirmations, unchanged "
            "allocator peak, and correctness passing; raw96+ stays q4, "
            "2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({16}),
        kv_head_counts=frozenset({2}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((64, 96),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_moe_single_qs4_w8.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 8, 3, q_splits=4),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=400,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e153-e157 B200 MoE full D512: NSYS found dKV at 47%; w8 improved "
            "2K-3.5K full F+B by 2.44-5.34% with paired confirmations, "
            "unchanged allocator peak, and correctness passing, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({16}),
        kv_head_counts=frozenset({2}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((64, 128),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_single_qs4.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 4, 3, q_splits=4),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=300,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e116-e125 B200 single-sequence D512 full F+B: qs4 improved "
            "raw-grid 64-223 by 2.7-42% with correctness passing; raw-grid "
            "224+ and packed workloads remain on prior configs, 2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        min_query_length=2048,
        separate_dkv_scratch=True,
        relaxed_dkv_atomics=True,
        grid_ranges=((64, 224),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.backward_dkv_b200_bkv64_grid.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="backward_dkv",
        head_dim=512,
        config=KernelConfig(16, 64, 512, 4, 3, q_splits=1),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=200,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e063-e066 selected BKV64 for verified B200 grids; e081-e088/e084 "
            "selected stages3 with all correctness gates passing, 2.39-6.35% "
            "full F+B gain, and unchanged allocator peak; raw_grid=448 stayed "
            "on BKV16, 2026-08-04 PST"
        ),
        grid_ranges=((128, 257), (512, None)),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.forward_b200_e2b_bkv64.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="forward",
        head_dim=512,
        config=KernelConfig(32, 64, 512, 4, 2),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=200,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e183-e186 B200 E2B full D512: forward BKV64 improved full F+B "
            "by 2.0-4.0% at raw-grid 32-240 with unchanged allocator peak "
            "and correctness passing; longer and packed workloads stay BKV32, "
            "2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({8}),
        kv_head_counts=frozenset({1}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        grid_ranges=((32, 241),),
        grid_probe_block_kv=64,
    )
    add_config(
        registration_id=(
            "triton_gqa_varlen_v1.forward_b200_moe_bkv64.sm100.d512"
        ),
        implementation_id="triton_gqa_varlen_v1",
        arch="sm100",
        role="forward",
        head_dim=512,
        config=KernelConfig(32, 64, 512, 4, 2),
        training_modes=frozenset({True}),
        config_kind="tuned_override",
        priority=200,
        gpu_name_patterns=frozenset({"B200"}),
        evidence_status="verified",
        evidence=(
            "e183-e186 B200 MoE full D512: forward BKV64 improved full F+B "
            "by about 2% at raw-grid 64-96 with unchanged allocator peak and "
            "correctness passing; raw112+ and packed workloads stay BKV32, "
            "2026-08-05 PST"
        ),
        batch_sizes=frozenset({1}),
        q_head_counts=frozenset({16}),
        kv_head_counts=frozenset({2}),
        window_sizes=frozenset({0}),
        min_query_length=2048,
        grid_ranges=((64, 97),),
        grid_probe_block_kv=64,
    )

    for head_dim, config in h100_inference_overrides.items():
        add_config(
            registration_id=(
                f"triton_gqa_batched_v1.forward_h100_tuned.sm90.d{head_dim}"
            ),
            implementation_id="triton_gqa_batched_v1",
            arch="sm90",
            role="forward",
            head_dim=head_dim,
            config=config,
            training_modes=frozenset({False}),
            config_kind="tuned_override",
            priority=100,
            gpu_name_patterns=frozenset({"H100"}),
            evidence_status="baseline",
            evidence=(
                "historical H100 inference tuning from context/baseline.md; "
                "values preserved, current H100 recertification pending"
            ),
        )
    return registry


DEFAULT_REGISTRY = build_default_registry()
