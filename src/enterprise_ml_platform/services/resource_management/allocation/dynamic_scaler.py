"""Dynamic scaling logic based on utilization targets."""

from __future__ import annotations


class DynamicScaler:
    """Decide when to scale resources up or down."""

    def __init__(self, target: float = 0.7, tolerance: float = 0.1) -> None:
        self.target = target
        self.tolerance = tolerance

    def decide(self, utilization: float) -> str:
        if utilization > self.target + self.tolerance:
            return "scale_up"
        if utilization < self.target - self.tolerance:
            return "scale_down"
        return "steady"
