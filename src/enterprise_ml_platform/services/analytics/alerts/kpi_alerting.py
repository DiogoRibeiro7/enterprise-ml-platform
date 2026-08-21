"""KPI based alerting utilities."""

from __future__ import annotations


class KPIAlerting:
    """Trigger callbacks when KPIs exceed configured thresholds."""

    def __init__(self) -> None:
        self.handlers: list[callable[[str, float, float], None]] = []

    def register_handler(self, handler: callable[[str, float, float], None]) -> None:
        self.handlers.append(handler)

    def check(self, metrics: dict[str, float], thresholds: dict[str, float]) -> None:
        for name, limit in thresholds.items():
            value = metrics.get(name)
            if value is not None and value < limit:
                for handler in self.handlers:
                    handler(name, value, limit)
