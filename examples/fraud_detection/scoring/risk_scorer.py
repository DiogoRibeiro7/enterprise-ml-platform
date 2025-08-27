"""Combine model output and rules into a unified risk score."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..models.ensemble_fraud_model import EnsembleFraudModel
from ..rules.rule_engine import RuleEngine


@dataclass
class RiskScorer:
    model: EnsembleFraudModel
    rules: RuleEngine

    def score(self, txn: Dict) -> Dict[str, float | List[str]]:
        features = np.array([list(txn["features"].values())])
        model_prob = float(self.model.predict_proba(features)[0])
        triggered = self.rules.evaluate(txn)
        # simple heuristic: boost probability when rules fire
        score = model_prob + 0.1 * len(triggered)
        return {"probability": min(score, 1.0), "rules": triggered}
