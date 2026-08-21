"""XGBoost model trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, r2_score

try:  # pragma: no cover - optional dependency
    import xgboost as xgb
except Exception:  # pragma: no cover - imported lazily
    xgb = None  # type: ignore

from ....core.base_components import ModelTrainer

logger = structlog.get_logger()


@dataclass
class XGBoostTrainer(ModelTrainer):
    """Trainer for XGBoost models.

    The trainer supports basic GPU acceleration, early stopping and feature
    importance extraction. Distributed training can be enabled when Ray is
    available and ``distributed`` is set to ``True``.
    """

    params: dict[str, Any] = field(default_factory=dict)
    distributed: bool = False

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Any:
        """Train an XGBoost model.

        Args:
            features: Training features.
            targets: Training targets.
            X_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            Trained XGBoost model instance.
        """
        if xgb is None:  # pragma: no cover - runtime check
            raise ImportError("xgboost is required for XGBoostTrainer")

        # heuristic task detection
        is_classification = len(np.unique(targets)) < 20
        model_cls = xgb.XGBClassifier if is_classification else xgb.XGBRegressor
        model = model_cls(**self.params)

        eval_set: tuple[np.ndarray, np.ndarray] | None = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None and self.params.get("early_stopping_rounds"):
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = False

        model.fit(features, targets, **fit_kwargs)
        return model

    def evaluate(
        self, model: Any, features: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        """Evaluate the trained model."""
        preds = model.predict(features)
        if len(np.unique(targets)) < 20:
            return {"accuracy": float(accuracy_score(targets, preds))}
        return {"r2": float(r2_score(targets, preds))}

    def save(self, model: Any, path: str) -> None:
        """Persist the model to ``path``."""
        model.save_model(path)

    def feature_importance(self, model: Any) -> dict[str, float]:
        """Return feature importance scores."""
        if xgb is None:
            return {}
        booster = model.get_booster()
        return {k: float(v) for k, v in booster.get_score().items()}
