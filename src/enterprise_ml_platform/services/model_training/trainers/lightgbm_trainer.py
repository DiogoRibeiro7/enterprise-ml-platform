"""LightGBM model trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, r2_score

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

from ....core.base_components import ModelTrainer

logger = structlog.get_logger()


@dataclass
class LightGBMTrainer(ModelTrainer):
    """Trainer for LightGBM models."""

    params: dict[str, Any] = field(default_factory=dict)
    distributed: bool = False

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Any:
        if lgb is None:  # pragma: no cover - runtime check
            raise ImportError("lightgbm is required for LightGBMTrainer")

        is_classification = len(np.unique(targets)) < 20
        model_cls = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
        model = model_cls(**self.params)

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_metric"] = "logloss" if is_classification else "l2"
            fit_kwargs["verbose"] = False

        model.fit(features, targets, **fit_kwargs)
        return model

    def evaluate(
        self, model: Any, features: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        preds = model.predict(features)
        if len(np.unique(targets)) < 20:
            preds = (preds > 0.5).astype(int)
            return {"accuracy": float(accuracy_score(targets, preds))}
        return {"r2": float(r2_score(targets, preds))}

    def save(self, model: Any, path: str) -> None:
        model.booster_.save_model(path)

    def feature_importance(self, model: Any) -> dict[str, float]:
        if lgb is None:
            return {}
        return {
            k: float(v)
            for k, v in zip(
                model.feature_name_, model.feature_importances_, strict=True
            )
        }
