"""Dynamic dashboard construction utilities."""
from __future__ import annotations

from typing import Any, Dict, Iterable


class DashboardBuilder:
    """Builds lightweight dashboard representations.

    In the real platform this component would interface with a UI layer or a
    BI tool.  For the purposes of the example it simply bundles metrics and
    chart descriptors into a dictionary structure.
    """

    def build(self, metrics: Dict[str, float], charts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble a dashboard description from metrics and charts."""
        return {"metrics": metrics, "charts": list(charts)}
