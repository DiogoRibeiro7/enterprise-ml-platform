"""Simple insight generation using heuristics."""

from __future__ import annotations


class InsightEngine:
    """Derive basic textual insights from metrics."""

    def generate(self, metrics: dict[str, float]) -> list[str]:
        insights: list[str] = []
        for name, value in metrics.items():
            if value > 0.9:
                insights.append(f"{name} is performing exceptionally well")
            elif value < 0.5:
                insights.append(f"{name} is underperforming")
        return insights
