"""Record and expose cost metrics."""

from __future__ import annotations

from collections import defaultdict

from prometheus_client import CollectorRegistry, Counter


class CostTracker:
    """Track cost per project/user/model."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._costs: dict[str, float] = {}
        self._history: dict[str, list[float]] = defaultdict(list)
        # Use a dedicated registry so multiple trackers can exist in tests
        self.registry = registry or CollectorRegistry()
        self.metric = Counter(
            "ml_cost_dollars_total",
            "Accrued ML cost in dollars",
            ["project", "user", "model"],
            registry=self.registry,
        )

    def record_cost(self, project: str, user: str, model: str, amount: float) -> None:
        key = f"{project}:{user}:{model}"
        self._costs[key] = self._costs.get(key, 0.0) + amount
        self._history[project].append(amount)
        self.metric.labels(project, user, model).inc(amount)

    def total_cost(
        self,
        project: str | None = None,
        user: str | None = None,
        model: str | None = None,
    ) -> float:
        total = 0.0
        for key, value in self._costs.items():
            p, u, m = key.split(":")
            if project and p != project:
                continue
            if user and u != user:
                continue
            if model and m != model:
                continue
            total += value
        return total

    def get_cost_history(self, project: str) -> list[float]:
        return self._history.get(project, [])
