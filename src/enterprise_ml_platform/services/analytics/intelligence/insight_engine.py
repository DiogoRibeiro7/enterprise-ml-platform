"""Simple insight generation using heuristics."""
from __future__ import annotations

from typing import Dict, Iterable, List


class InsightEngine:
    """Derive basic textual insights from metrics."""

    def generate(self, metrics: Dict[str, float]) -> List[str]:
        insights: List[str] = []
        for name, value in metrics.items():
            if value > 0.9:
                insights.append(f"{name} is performing exceptionally well")
            elif value < 0.5:
                insights.append(f"{name} is underperforming")
        return insights
