"""Hook into monitoring service for experiment metrics."""

from __future__ import annotations


class MonitoringIntegration:
    def emit_event(
        self, experiment: str, data
    ) -> None:  # pragma: no cover - placeholder
        del experiment, data
