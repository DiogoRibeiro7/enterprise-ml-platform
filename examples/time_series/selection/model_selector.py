from __future__ import annotations

from typing import Dict, Tuple


class ModelSelector:
    """Pick the best model according to a metric."""

    def select(
        self, results: Dict[str, Dict[str, float]], metric: str = "mae"
    ) -> Tuple[str, Dict[str, float]]:
        best_name, best_metrics = min(
            results.items(), key=lambda kv: kv[1].get(metric, float("inf"))
        )
        return best_name, best_metrics
