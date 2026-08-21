"""Compare model versions based on stored metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelComparator:
    """Utility for comparing performance metrics between models."""

    def compare(
        self,
        metrics: dict[tuple[str, str], dict[str, float]],
        model_a: tuple[str, str],
        model_b: tuple[str, str],
    ) -> dict[str, float]:
        """Return metric deltas ``a - b`` for matching keys."""

        ma = metrics.get(model_a, {})
        mb = metrics.get(model_b, {})
        return {k: ma.get(k, 0.0) - mb.get(k, 0.0) for k in ma.keys() & mb.keys()}
