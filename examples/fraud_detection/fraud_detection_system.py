"""End-to-end fraud detection example showcasing platform capabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from redis.asyncio import Redis

from enterprise_ml_platform.services.feature_store import (
    FeatureRegistry,
    FeatureStoreConfig,
    FeatureStoreService,
    OfflineFeatureStore,
    OnlineFeatureStore,
)
from enterprise_ml_platform.services.monitoring.collectors.metrics_collector import (
    MetricsCollector,
)

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
    feature_store: FeatureStoreService | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Seed with simple rule: large amount
        self.rules.add_rule("high_amount", lambda t: float(t["amount"]) > 10000)
        # Train model on synthetic dataset for demonstration
        X = np.random.randn(200, 3)
        y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
        self.model.fit(X, y)
        self.scorer = RiskScorer(self.model, self.rules)
        # Initialize feature store for demonstration
        metrics = MetricsCollector()
        redis_client = Redis.from_url("redis://localhost:6379/0")
        online = OnlineFeatureStore(redis_client, metrics=metrics)
        offline = OfflineFeatureStore(metrics=metrics)
        registry = FeatureRegistry()
        cfg = FeatureStoreConfig()
        self.feature_store = FeatureStoreService(cfg, registry, online, offline)

    async def process_transaction(self, txn: Dict) -> Dict:
        """Process ``txn`` and return decision information."""
        enriched = self.processor.process(txn)
        if self.feature_store:
            df = pd.DataFrame(
                [
                    {
                        "entity_id": txn["account_id"],
                        **enriched["features"],
                    }
                ]
            )
            await self.feature_store.register_features("fraud_features", df)
            stored = await self.feature_store.get_online_features(
                "fraud_features", str(txn["account_id"]), list(enriched["features"].keys())
            )
            enriched["stored_features"] = stored
        score = self.scorer.score(enriched)
        if score["probability"] > 0.8:
            self.alerting.send_alert(enriched, score)
            alert = self.alerting.recent_alerts(1)[0]
            self.analytics.record_alert(alert)
            case = self.cases.create_case(alert)
        else:
            case = None
        return {"txn": enriched, "score": score, "case": case}
