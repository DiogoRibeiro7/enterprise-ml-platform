"""Analytics for fraud detection operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FraudAnalytics:
    alerts: List[Dict] = field(default_factory=list)

    def record_alert(self, alert: Dict) -> None:
        self.alerts.append(alert)

    def summary(self) -> Dict[str, float]:
        total = len(self.alerts)
        avg_score = sum(a["score"] for a in self.alerts) / total if total else 0.0
        return {"alerts": total, "avg_score": avg_score}
