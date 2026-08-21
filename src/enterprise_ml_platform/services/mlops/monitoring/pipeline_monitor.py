"""In memory monitoring of workflow executions."""

from __future__ import annotations

from typing import Any


class PipelineMonitor:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)
