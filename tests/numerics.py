from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Tolerance:
    """Stable multi-metric tolerance for FP16/BF16 kernel comparisons."""

    cosine_min: float = 0.9999
    max_abs: float = 5e-2
    mean_abs: float = 5e-4
    relative_l2: float = 2e-2


@dataclass(frozen=True)
class Metrics:
    cosine: float
    max_abs: float
    mean_abs: float
    relative_l2: float
    actual_norm: float
    expected_norm: float

    def format(self) -> str:
        return (
            f"cos={self.cosine:.9f} max_abs={self.max_abs:.3e} "
            f"mean_abs={self.mean_abs:.3e} rel_l2={self.relative_l2:.3e} "
            f"norms=({self.actual_norm:.3e}, {self.expected_norm:.3e})"
        )


def compare(actual: torch.Tensor, expected: torch.Tensor) -> Metrics:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )
    if not torch.isfinite(actual).all().item():
        raise AssertionError("actual tensor contains NaN or Inf")
    if not torch.isfinite(expected).all().item():
        raise AssertionError("reference tensor contains NaN or Inf")

    # Always accumulate comparison statistics in FP32. FP16 cosine can report
    # false failures for large tensors even when elementwise differences are at
    # the input dtype's rounding scale.
    actual_f = actual.detach().float().reshape(-1)
    expected_f = expected.detach().float().reshape(-1)
    diff = actual_f - expected_f

    actual_norm_t = torch.linalg.vector_norm(actual_f)
    expected_norm_t = torch.linalg.vector_norm(expected_f)
    actual_norm = actual_norm_t.item()
    expected_norm = expected_norm_t.item()
    denom = actual_norm_t * expected_norm_t
    if denom.item() <= 1e-24:
        cosine = 1.0 if torch.equal(actual_f, expected_f) else 0.0
    else:
        cosine = torch.dot(actual_f, expected_f).div(denom).item()

    expected_scale = max(expected_norm, 1e-12)
    return Metrics(
        cosine=cosine,
        max_abs=diff.abs().max().item() if diff.numel() else 0.0,
        mean_abs=diff.abs().mean().item() if diff.numel() else 0.0,
        relative_l2=torch.linalg.vector_norm(diff).item() / expected_scale,
        actual_norm=actual_norm,
        expected_norm=expected_norm,
    )


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    tolerance: Tolerance,
) -> Metrics:
    metrics = compare(actual, expected)
    failures: list[str] = []
    if metrics.cosine < tolerance.cosine_min:
        failures.append(f"cosine < {tolerance.cosine_min}")
    if metrics.max_abs > tolerance.max_abs:
        failures.append(f"max_abs > {tolerance.max_abs}")
    if metrics.mean_abs > tolerance.mean_abs:
        failures.append(f"mean_abs > {tolerance.mean_abs}")
    if metrics.relative_l2 > tolerance.relative_l2:
        failures.append(f"relative_l2 > {tolerance.relative_l2}")
    if failures:
        raise AssertionError(f"{name}: {', '.join(failures)}; {metrics.format()}")
    return metrics
