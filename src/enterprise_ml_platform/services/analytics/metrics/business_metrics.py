"""Business KPI calculation helpers."""
from __future__ import annotations

from typing import Dict, Iterable


class BusinessMetrics:
    """Computes simple aggregate metrics from record streams."""

    def compute_kpis(self, records: Iterable[Dict[str, float]]) -> Dict[str, float]:
        """Calculate basic KPIs from an iterable of numeric mappings."""
        totals: Dict[str, float] = {}
        count = 0
        for rec in records:
            count += 1
            for key, value in rec.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        if count:
            return {k: v / count for k, v in totals.items()}
        return {}
