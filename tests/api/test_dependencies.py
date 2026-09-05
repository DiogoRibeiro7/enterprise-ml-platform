"""Concurrency regressions for API dependency initialization."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pytest import MonkeyPatch

from enterprise_ml_platform.api import dependencies


def test_metrics_collector_is_initialized_once_under_concurrency(
    monkeypatch: MonkeyPatch,
) -> None:
    """Concurrent first requests must not register duplicate timeseries."""
    constructed: list[object] = []

    class SlowMetricsCollector:
        def __init__(self) -> None:
            time.sleep(0.02)
            constructed.append(self)

    monkeypatch.setattr(dependencies, "_metrics", None)
    monkeypatch.setattr(dependencies, "MetricsCollector", SlowMetricsCollector)

    with ThreadPoolExecutor(max_workers=8) as executor:
        instances: list[Any] = list(
            executor.map(lambda _: dependencies.get_metrics(), range(8))
        )

    assert len(constructed) == 1
    assert len({id(instance) for instance in instances}) == 1
