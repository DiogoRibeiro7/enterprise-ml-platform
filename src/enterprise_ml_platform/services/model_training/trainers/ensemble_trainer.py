"""Ensemble model trainer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.metrics import accuracy_score, r2_score

from ....core.base_components import ModelTrainer


@dataclass
class EnsembleTrainer(ModelTrainer):
    """Trainer supporting voting and stacking ensembles."""

    estimators: Iterable[tuple[str, BaseEstimator]]
    task: str = "classification"
    method: str = "voting"
    final_estimator: BaseEstimator | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def _build_model(self) -> BaseEstimator:
        if self.task == "classification":
            if self.method == "stacking":
                return StackingClassifier(
                    estimators=list(self.estimators),
                    final_estimator=self.final_estimator,
                    **self.params,
                )
            return VotingClassifier(estimators=list(self.estimators), **self.params)
        else:
            if self.method == "stacking":
                return StackingRegressor(
                    estimators=list(self.estimators),
                    final_estimator=self.final_estimator,
                    **self.params,
                )
            return VotingRegressor(estimators=list(self.estimators), **self.params)

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaseEstimator:
        """Train the ensemble model.

        Validation arrays are accepted for API compatibility with other
        trainers but are currently unused.
        """
        model = self._build_model()
        model.fit(features, targets)
        return model

    def evaluate(
        self, model: BaseEstimator, features: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        preds = model.predict(features)
        if self.task == "classification":
            return {"accuracy": float(accuracy_score(targets, preds))}
        return {"r2": float(r2_score(targets, preds))}

    def save(self, model: BaseEstimator, path: str) -> None:
        joblib.dump(model, path)
