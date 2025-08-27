"""Real-time transaction processing utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

from ..features.fraud_features import FraudFeatureExtractor


@dataclass
class TransactionProcessor:
    """Validate and enrich streaming transactions."""

    feature_extractor: FraudFeatureExtractor = field(default_factory=FraudFeatureExtractor)
    _seen_ids: Set[str] = field(default_factory=set)

    def process(self, txn: Dict) -> Dict:
        """Return enriched transaction with computed features.

        Duplicate transactions are ignored.  Minimal validation is performed to
        keep the example concise.
        """
        txn_id = str(txn.get("id"))
        if txn_id in self._seen_ids:
            raise ValueError("duplicate transaction")
        self._seen_ids.add(txn_id)

        features = self.feature_extractor.compute(txn)
        return {**txn, "features": features}

    def batch_process(self, txns: Iterable[Dict]) -> Iterable[Dict]:
        for txn in txns:
            try:
                yield self.process(txn)
            except ValueError:
                continue
