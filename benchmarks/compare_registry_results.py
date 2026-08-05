#!/usr/bin/env python3
"""Compare two canonical attention benchmark records and enforce P2 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flash_attn.performance import write_json_exclusive
from flash_attn.regression import (
    IncompatibleBenchmarkResults,
    compare_benchmark_results,
)


def result_path(value: Path) -> Path:
    return value / "result.json" if value.is_dir() else value


def load_result(path: Path) -> dict[str, object]:
    resolved = result_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result must be a JSON object: {resolved}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-latency-regression", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional exclusive JSON path; an existing file is never replaced.",
    )
    args = parser.parse_args()
    verdict = compare_benchmark_results(
        load_result(args.baseline),
        load_result(args.candidate),
        max_latency_regression=args.max_latency_regression,
    )
    rendered = json.dumps(verdict, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        write_json_exclusive(args.output, verdict)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IncompatibleBenchmarkResults as error:
        raise SystemExit(f"incompatible benchmark results: {error}") from error
