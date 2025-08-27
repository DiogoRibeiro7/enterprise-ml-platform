"""Alert management for suspicious transactions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FraudAlerting:
    alerts: List[Dict] = field(default_factory=list)

    def send_alert(self, txn: Dict, score: Dict) -> None:
        alert = {"txn_id": txn["id"], "score": score["probability"], "rules": score["rules"]}
        self.alerts.append(alert)

    def recent_alerts(self, limit: int = 10) -> List[Dict]:
        return self.alerts[-limit:]
