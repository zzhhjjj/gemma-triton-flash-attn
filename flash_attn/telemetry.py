"""Opt-in, process-local observability for attention registry selection.

Nothing is emitted or transmitted automatically. Callers explicitly open a
capture context around the model work they want to inspect, then consume a
JSON-compatible snapshot or compact text summary after the context exits.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import contextvars
from dataclasses import asdict
import json
from threading import Lock
from typing import Iterator, Literal, Mapping, TYPE_CHECKING


if TYPE_CHECKING:
    from .registry import AttentionSpec, Resolution, RuntimeSpec


TelemetryMode = Literal["summary", "debug"]


class AttentionSelectionTelemetry:
    """Aggregate registry selections without logging from the hot path.

    ``summary`` stores bounded config/semantic counters. ``debug`` additionally
    retains one full explanation per distinct immutable spec/runtime/role key.
    A recorder is local to the current process; FSDP jobs therefore get one
    independent snapshot per rank.
    """

    def __init__(
        self,
        mode: TelemetryMode = "summary",
        *,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        if mode not in {"summary", "debug"}:
            raise ValueError(f"unsupported telemetry mode: {mode}")
        self.mode = mode
        self.labels = dict(labels or {})
        self._selection_counts: Counter[tuple[object, ...]] = Counter()
        self._fallback_counts: Counter[tuple[str, str, str]] = Counter()
        self._debug_events: dict[tuple[object, ...], dict[str, object]] = {}
        self._lock = Lock()

    @staticmethod
    def _selection_key(
        spec: "AttentionSpec",
        runtime: "RuntimeSpec",
        resolution: "Resolution",
    ) -> tuple[object, ...]:
        return (
            resolution.role,
            resolution.implementation.id,
            resolution.config_registration.id,
            resolution.config_registration.config_kind,
            resolution.config_registration.evidence_status,
            runtime.gpu_arch,
            runtime.gpu_name,
            spec.layout,
            spec.training,
            spec.dtype,
            spec.causal,
            "sliding" if spec.is_sliding else "full",
            spec.window_size,
            spec.q_heads,
            spec.kv_heads,
            spec.head_dim,
            resolution.config.block_q,
            resolution.config.block_kv,
            resolution.config.block_d,
            resolution.config.num_warps,
            resolution.config.num_stages,
            resolution.config.q_splits,
        )

    @staticmethod
    def _selection_row(key: tuple[object, ...], count: int) -> dict[str, object]:
        names = (
            "role",
            "implementation",
            "config_id",
            "config_kind",
            "evidence_status",
            "gpu_arch",
            "gpu_name",
            "layout",
            "training",
            "dtype",
            "causal",
            "attention_mode",
            "window_size",
            "q_heads",
            "kv_heads",
            "head_dim",
            "block_q",
            "block_kv",
            "block_d",
            "num_warps",
            "num_stages",
            "q_splits",
        )
        return {"count": count, **dict(zip(names, key))}

    def record_selection(
        self,
        spec: "AttentionSpec",
        runtime: "RuntimeSpec",
        resolution: "Resolution",
    ) -> None:
        key = self._selection_key(spec, runtime, resolution)
        with self._lock:
            self._selection_counts[key] += 1
            debug_key = (
                spec,
                runtime,
                resolution.role,
                resolution.config_registration.id,
                resolution.config,
            )
            if self.mode == "debug" and debug_key not in self._debug_events:
                self._debug_events[debug_key] = {
                    "spec": asdict(spec),
                    "runtime": asdict(runtime),
                    "resolution": resolution.to_dict(),
                }

    def record_fallback(self, source: str, target: str, reason: str) -> None:
        """Record an explicit adapter route change, never an inferred fallback."""
        with self._lock:
            self._fallback_counts[(source, target, reason)] += 1

    def reset(self) -> None:
        with self._lock:
            self._selection_counts.clear()
            self._fallback_counts.clear()
            self._debug_events.clear()

    def snapshot(self) -> dict[str, object]:
        """Return a deterministic, JSON-compatible copy of current counters."""
        with self._lock:
            selection_items = sorted(
                self._selection_counts.items(), key=lambda item: repr(item[0])
            )
            fallback_items = sorted(self._fallback_counts.items())
            debug_items = sorted(
                self._debug_events.items(), key=lambda item: repr(item[0])
            )

        selections = [
            self._selection_row(key, count) for key, count in selection_items
        ]
        fallbacks = [
            {
                "count": count,
                "source": source,
                "target": target,
                "reason": reason,
            }
            for (source, target, reason), count in fallback_items
        ]
        snapshot: dict[str, object] = {
            "mode": self.mode,
            "labels": dict(self.labels),
            "total_selections": sum(row["count"] for row in selections),
            "total_fallbacks": sum(row["count"] for row in fallbacks),
            "selections": selections,
            "fallbacks": fallbacks,
        }
        if self.mode == "debug":
            snapshot["debug_events"] = [event for _, event in debug_items]
        return snapshot

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.snapshot(), indent=indent, sort_keys=True)

    def format_summary(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "attention registry selection summary",
            f"  selections: {snapshot['total_selections']}",
            f"  fallbacks: {snapshot['total_fallbacks']}",
        ]
        for row in snapshot["selections"]:
            lines.append(
                "  "
                f"{row['config_id']} [{row['role']}, {row['config_kind']}, "
                f"{row['gpu_arch']}, {row['attention_mode']}, "
                f"BQ={row['block_q']}, BKV={row['block_kv']}, "
                f"w={row['num_warps']}, s={row['num_stages']}, "
                f"QS={row['q_splits']}] x{row['count']}"
            )
        for row in snapshot["fallbacks"]:
            lines.append(
                "  fallback "
                f"{row['source']} -> {row['target']} "
                f"({row['reason']}) x{row['count']}"
            )
        return "\n".join(lines)


_active_telemetry: contextvars.ContextVar[AttentionSelectionTelemetry | None] = (
    contextvars.ContextVar("_attention_selection_telemetry", default=None)
)


@contextmanager
def capture_attention_selection(
    mode: TelemetryMode = "summary",
    *,
    labels: Mapping[str, object] | None = None,
) -> Iterator[AttentionSelectionTelemetry]:
    """Capture selections made in this context without printing or uploading."""
    recorder = AttentionSelectionTelemetry(mode, labels=labels)
    token = _active_telemetry.set(recorder)
    try:
        yield recorder
    finally:
        _active_telemetry.reset(token)


def record_attention_selection(
    spec: "AttentionSpec",
    runtime: "RuntimeSpec",
    resolution: "Resolution",
    *,
    recorder: AttentionSelectionTelemetry | None = None,
) -> None:
    if recorder is None:
        recorder = _active_telemetry.get()
    if recorder is not None:
        recorder.record_selection(spec, runtime, resolution)


def record_attention_fallback(source: str, target: str, reason: str) -> None:
    recorder = _active_telemetry.get()
    if recorder is not None:
        recorder.record_fallback(source, target, reason)


def get_active_attention_telemetry() -> AttentionSelectionTelemetry | None:
    """Internal hook for carrying the recorder through autograd contexts."""
    return _active_telemetry.get()
