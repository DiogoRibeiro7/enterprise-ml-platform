"""Model validation utilities."""
from __future__ import annotations

from typing import Dict


class ModelValidator:
    """Validate metrics against simple threshold rules."""

    def validate(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
        for key, limit in thresholds.items():
            if metrics.get(key, 0.0) < limit:
                return False
        return True
