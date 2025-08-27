"""Lightweight ensemble model for fraud detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression


@dataclass
class EnsembleFraudModel:
    """Combine supervised and anomaly models for robust detection."""

    lr: LogisticRegression = LogisticRegression(max_iter=100)
    iso: IsolationForest = IsolationForest(contamination=0.01, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.lr.fit(X, y)
        self.iso.fit(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for each sample in ``X``."""
        lr_prob = self.lr.predict_proba(X)[:, 1]
        iso_score = -self.iso.score_samples(X)  # higher -> more anomalous
        # Normalize iso scores to 0-1
        iso_prob = (iso_score - iso_score.min()) / (
            iso_score.max() - iso_score.min() + 1e-8
        )
        return 0.7 * lr_prob + 0.3 * iso_prob

    def explain(self, feature_names: list[str]) -> Dict[str, float]:
        """Return feature importance from the logistic regression model."""
        coeffs = self.lr.coef_[0]
        return {name: float(abs(c)) for name, c in zip(feature_names, coeffs)}
