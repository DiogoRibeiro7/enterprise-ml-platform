"""End-to-end fraud detection example showcasing platform capabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .alerts.fraud_alerting import FraudAlerting
from .data_processing.transaction_processor import TransactionProcessor
from .investigation.case_management import CaseManagement
from .models.ensemble_fraud_model import EnsembleFraudModel
from .reporting.fraud_analytics import FraudAnalytics
from .rules.rule_engine import RuleEngine
from .scoring.risk_scorer import RiskScorer


@dataclass
class FraudDetectionSystem:
    """Orchestrates real-time fraud detection workflow."""

    processor: TransactionProcessor = field(default_factory=TransactionProcessor)
    rules: RuleEngine = field(default_factory=RuleEngine)
    model: EnsembleFraudModel = field(default_factory=EnsembleFraudModel)
    scorer: RiskScorer | None = None
    alerting: FraudAlerting = field(default_factory=FraudAlerting)
    cases: CaseManagement = field(default_factory=CaseManagement)
    analytics: FraudAnalytics = field(default_factory=FraudAnalytics)

    def __post_init__(self) -> None:
        # Seed with simple rule: large amount
        self.rules.add_rule("high_amount", lambda t: float(t["amount"]) > 10000)
        # Train model on synthetic dataset for demonstration
        X = np.random.randn(200, 3)
        y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
        self.model.fit(X, y)
        self.scorer = RiskScorer(self.model, self.rules)

    def process_transaction(self, txn: Dict) -> Dict:
        """Process ``txn`` and return decision information."""
        enriched = self.processor.process(txn)
        score = self.scorer.score(enriched)
        if score["probability"] > 0.8:
            self.alerting.send_alert(enriched, score)
            alert = self.alerting.recent_alerts(1)[0]
            self.analytics.record_alert(alert)
            case = self.cases.create_case(alert)
        else:
            case = None
        return {"txn": enriched, "score": score, "case": case}
