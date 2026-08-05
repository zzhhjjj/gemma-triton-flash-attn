"""Model attention profiles translated into model-independent semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .registry import AttentionSpec, Layout


@dataclass(frozen=True)
class ModelAttentionProfile:
    id: str
    model_family: str
    attention_kind: str
    q_heads: int
    kv_heads: int
    head_dim: int
    causal: bool
    window_size: int
    layout: Layout = "bhsd"
    image_groups: bool = False

    def make_spec(
        self,
        *,
        dtype: str,
        training: bool,
        dropout_p: float = 0.0,
        softcap: float | None = None,
        batch_size: int = 1,
        query_length: int = 1,
        key_length: int | None = None,
        **overrides: object,
    ) -> AttentionSpec:
        profile = replace(self, **overrides) if overrides else self
        return AttentionSpec(
            q_heads=profile.q_heads,
            kv_heads=profile.kv_heads,
            head_dim=profile.head_dim,
            dtype=dtype,
            causal=profile.causal,
            window_size=profile.window_size,
            layout=profile.layout,
            training=training,
            image_groups=profile.image_groups,
            dropout_p=dropout_p,
            softcap=softcap,
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
        )


class ModelProfileRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, ModelAttentionProfile] = {}

    def register(self, profile: ModelAttentionProfile) -> None:
        if profile.id in self._profiles:
            raise ValueError(f"duplicate model profile id: {profile.id}")
        self._profiles[profile.id] = profile

    def get(self, profile_id: str) -> ModelAttentionProfile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            available = ", ".join(sorted(self._profiles))
            raise KeyError(f"unknown model profile {profile_id!r}; available: {available}") from None

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))


def build_default_model_profiles() -> ModelProfileRegistry:
    registry = ModelProfileRegistry()
    for profile in (
        ModelAttentionProfile(
            "gemma4_e2b_text_full", "gemma4_e2b", "text_full", 8, 1, 512, True, 0
        ),
        ModelAttentionProfile(
            "gemma4_e2b_text_sliding",
            "gemma4_e2b",
            "text_sliding",
            8,
            1,
            256,
            True,
            512,
        ),
        ModelAttentionProfile(
            "gemma4_moe_text_full", "gemma4_moe", "text_full", 16, 2, 512, True, 0
        ),
        ModelAttentionProfile(
            "gemma4_moe_text_sliding",
            "gemma4_moe",
            "text_sliding",
            16,
            8,
            256,
            True,
            1024,
        ),
        ModelAttentionProfile(
            "gemma4_moe_image_group_sliding",
            "gemma4_moe",
            "image_group_sliding",
            16,
            8,
            256,
            True,
            1024,
            image_groups=True,
        ),
        ModelAttentionProfile(
            "gemma4_vision_full", "gemma4", "vision_full", 12, 12, 64, False, 0
        ),
    ):
        registry.register(profile)
    return registry


DEFAULT_MODEL_PROFILES = build_default_model_profiles()
