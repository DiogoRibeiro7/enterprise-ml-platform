"""Model validation utilities."""

from __future__ import annotations


class ModelValidator:
    """Validate metrics against simple threshold rules."""

    def validate(self, metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
        return all(metrics.get(key, 0.0) >= limit for key, limit in thresholds.items())
