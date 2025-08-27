from __future__ import annotations

"""High level model training orchestration service."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import asyncio
import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:  # pragma: no cover - optional dependency
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

from .trainers.xgboost_trainer import XGBoostTrainer
from .trainers.lightgbm_trainer import LightGBMTrainer
from .trainers.ensemble_trainer import EnsembleTrainer
from .optimization.hyperparameter_optimizer import HyperparameterOptimizer
from .explainability.model_explainer import ModelExplainer

logger = structlog.get_logger()


@dataclass
class ModelConfig:
    """Configuration for model training."""

    algorithm: str
    params: Dict[str, Any] = field(default_factory=dict)
    optimization: Optional[Dict[str, Any]] = None
    ensemble: Optional[Dict[str, Any]] = None
    explainability: Optional[Dict[str, Any]] = None
    distributed: bool = False


class ModelTrainingService:
    """Service coordinating model training, optimisation and explainability."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        if tracking_uri and mlflow:
            mlflow.set_tracking_uri(tracking_uri)
        self.logger = logger.bind(service="model-training")

    async def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Optional[ModelConfig] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a model according to ``config``.

        ``config`` is optional to preserve backwards compatibility with older
        callers.  When omitted a small voting ensemble of logistic regression
        and decision tree classifiers is trained, mimicking the configuration
        used in the test-suite.

        Returns:
            Tuple of trained model and evaluation metrics.
        """
        if config is None:
            config = ModelConfig(
                algorithm="ensemble",
                ensemble={
                    "estimators": [
                        ("lr", LogisticRegression(max_iter=100)),
                        ("dt", DecisionTreeClassifier(max_depth=3)),
                    ],
                    "task": "classification",
                    "method": "voting",
                },
            )

        trainer = self._build_trainer(config)

        params = dict(config.params)
        if config.optimization:
            optimizer = HyperparameterOptimizer()
            params.update(
                await optimizer.optimize(
                    lambda p: self._build_trainer(
                        ModelConfig(
                            algorithm=config.algorithm,
                            params=p,
                            ensemble=config.ensemble,
                            distributed=config.distributed,
                        )
                    ),
                    X,
                    y,
                    config.optimization,
                )
            )
            trainer = self._build_trainer(
                ModelConfig(
                    algorithm=config.algorithm,
                    params=params,
                    ensemble=config.ensemble,
                    distributed=config.distributed,
                )
            )

        model = await asyncio.to_thread(trainer.train, X, y, X_val, y_val)
        metrics = await asyncio.to_thread(
            trainer.evaluate,
            model,
            X_val if X_val is not None else X,
            y_val if y_val is not None else y,
        )

        if mlflow:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

        if config.explainability:
            explainer = ModelExplainer()
            explanations = explainer.explain(model, X, config.explainability)
            if mlflow:
                for key, value in explanations.items():
                    mlflow.log_metric(
                        f"{key}_rows",
                        float(getattr(value, "shape", (len(value),))[0]),
                    )

        return model, metrics

    def _build_trainer(self, config: ModelConfig) -> Any:
        algo = config.algorithm.lower()
        if algo == "xgboost":
            return XGBoostTrainer(params=config.params, distributed=config.distributed)
        if algo == "lightgbm":
            return LightGBMTrainer(params=config.params, distributed=config.distributed)
        if algo == "ensemble":
            if not config.ensemble:
                raise ValueError("ensemble configuration required for ensemble algorithm")
            return EnsembleTrainer(**config.ensemble)
        raise ValueError(f"Unknown algorithm {config.algorithm}")
