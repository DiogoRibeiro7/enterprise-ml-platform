"""Feature computations for fraud detection.

This module provides utilities to extract real-time features from incoming
transactions.  The implementation is intentionally lightweight so it can run in
<50ms per transaction while still showcasing common feature patterns used in
production systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List


@dataclass
class AccountHistory:
    """In-memory store tracking recent transactions for an account."""

    timestamps: List[datetime] = field(default_factory=list)
    amounts: List[float] = field(default_factory=list)

    def update(self, ts: datetime, amount: float) -> None:
        self.timestamps.append(ts)
        self.amounts.append(amount)
        # Keep only last 1 hour to bound memory
        cutoff = ts - timedelta(hours=1)
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
            self.amounts.pop(0)


class FraudFeatureExtractor:
    """Compute behavioural features for fraud detection."""

    def __init__(self) -> None:
        self._history: Dict[str, AccountHistory] = {}

    def compute(self, txn: Dict) -> Dict[str, float]:
        """Return a feature vector for *txn*.

        Parameters
        ----------
        txn: Dict
            Transaction payload containing ``account_id``, ``amount`` and
            ``timestamp`` fields.
        """
        account_id = txn["account_id"]
        amount = float(txn["amount"])
        ts = txn.get("timestamp", datetime.utcnow())

        history = self._history.setdefault(account_id, AccountHistory())
        history.update(ts, amount)

        velocity = len(history.timestamps) / max(
            (history.timestamps[-1] - history.timestamps[0]).total_seconds() / 60, 1e-3
        )
        avg_amount = sum(history.amounts) / len(history.amounts)

        return {
            "amount": amount,
            "avg_amount_last_hr": avg_amount,
            "txn_velocity_per_min": velocity,
        }
