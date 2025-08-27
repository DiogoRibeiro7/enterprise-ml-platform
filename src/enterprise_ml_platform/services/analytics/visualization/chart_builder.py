"""Lightweight chart builder used by the analytics engine."""
from __future__ import annotations

from typing import Any, Dict, Iterable


class ChartBuilder:
    """Create chart descriptions.

    The real implementation would likely integrate with a plotting library
    such as matplotlib or Plotly.  Here we merely capture the intent by
    returning dictionaries describing the desired chart.
    """

    def build_charts(self, data: Iterable[Dict[str, Any]], metrics: Dict[str, float]) -> Iterable[Dict[str, Any]]:
        """Create a basic bar chart for each metric."""
        charts = []
        for name, value in metrics.items():
            charts.append({"type": "bar", "title": name, "value": value})
        return charts
