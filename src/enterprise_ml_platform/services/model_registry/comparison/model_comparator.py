from __future__ import annotations

"""Compare model versions based on stored metrics."""

from dataclasses import dataclass
from typing import Dict, Tuple, Any


@dataclass
class ModelComparator:
    """Utility for comparing performance metrics between models."""

    def compare(
        self,
        metrics: Dict[Tuple[str, str], Dict[str, float]],
        model_a: Tuple[str, str],
        model_b: Tuple[str, str],
    ) -> Dict[str, float]:
        """Return metric deltas ``a - b`` for matching keys."""

        ma = metrics.get(model_a, {})
        mb = metrics.get(model_b, {})
        return {k: ma.get(k, 0.0) - mb.get(k, 0.0) for k in ma.keys() & mb.keys()}
