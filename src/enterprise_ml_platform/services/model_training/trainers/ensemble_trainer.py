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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

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
        """Score ``model`` on ``features``.

        Classification reports more than accuracy on purpose. On an imbalanced
        problem a model that predicts the majority class for every row scores
        as well as the base rate, so accuracy alone would promote it over one
        that actually separates the classes. ``majority_class_rate`` is
        reported alongside so the accuracy can be read against its own
        baseline.
        """
        preds = model.predict(features)
        if self.task != "classification":
            return {"r2": float(r2_score(targets, preds))}

        targets = np.asarray(targets)
        metrics = {
            "accuracy": float(accuracy_score(targets, preds)),
            "precision": float(precision_score(targets, preds, zero_division=0)),
            "recall": float(recall_score(targets, preds, zero_division=0)),
            "f1": float(f1_score(targets, preds, zero_division=0)),
            "majority_class_rate": float(np.bincount(targets).max() / len(targets)),
        }
        auc = self._roc_auc(model, features, targets)
        if auc is not None:
            metrics["roc_auc"] = auc
        return metrics

    @staticmethod
    def _roc_auc(
        model: BaseEstimator, features: np.ndarray, targets: np.ndarray
    ) -> float | None:
        """Return the ROC AUC, or ``None`` when it is not defined.

        It needs scores rather than labels, and it is undefined when only one
        class is present.
        """
        if len(np.unique(targets)) < 2 or not hasattr(model, "predict_proba"):
            return None
        try:
            scores = model.predict_proba(features)
        except Exception:  # pragma: no cover - estimator without fitted proba
            return None
        if scores.shape[1] != 2:
            return None
        return float(roc_auc_score(targets, scores[:, 1]))

    def save(self, model: BaseEstimator, path: str) -> None:
        joblib.dump(model, path)
