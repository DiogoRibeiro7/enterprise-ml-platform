from __future__ import annotations

"""Allocate tracked costs to tenants."""

from ..monitoring.cost_tracker import CostTracker


class CostAllocator:
    def __init__(self, tracker: CostTracker) -> None:
        self.tracker = tracker

    def allocate(self, tenant: str, cost: float) -> None:
        """Associate cost with a tenant."""
        self.tracker.record_cost(tenant, tenant, "tenant", cost)
