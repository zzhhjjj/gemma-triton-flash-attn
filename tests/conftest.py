from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked gpu (disabled by default)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-gpu"):
        return

    skip_gpu = pytest.mark.skip(reason="requires --run-gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

