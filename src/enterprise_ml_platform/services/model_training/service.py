"""High level model training orchestration service."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:  # pragma: no cover - optional dependency
    import mlflow
    import mlflow.sklearn
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

from .trainers.ensemble_trainer import EnsembleTrainer

try:  # pragma: no cover - optional trainers
    from .trainers.xgboost_trainer import XGBoostTrainer
except Exception:  # pragma: no cover
    XGBoostTrainer = None  # type: ignore

try:  # pragma: no cover - optional trainers
    from .trainers.lightgbm_trainer import LightGBMTrainer
except Exception:  # pragma: no cover
    LightGBMTrainer = None  # type: ignore

from ..monitoring.serving_drift import DRIFT_REFERENCE_ARTIFACT, DriftReference
from .explainability.model_explainer import ModelExplainer
from .optimization.hyperparameter_optimizer import HyperparameterOptimizer

logger = structlog.get_logger()


@dataclass
class ModelConfig:
    """Configuration for model training."""

    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] | None = None
    ensemble: dict[str, Any] | None = None
    explainability: dict[str, Any] | None = None
    distributed: bool = False


class ModelTrainingService:
    """Service coordinating model training, optimisation and explainability."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "enterprise-ml-platform",
        artifact_location: str | None = None,
    ) -> None:
        """Configure the service and, if a tracking URI is known, MLflow.

        Tracking is opt-in. Without an explicit ``tracking_uri`` or
        ``MLFLOW_TRACKING_URI`` nothing is logged, so training never writes a
        tracking store into whatever directory the process happens to run in.

        Args:
            tracking_uri: MLflow tracking URI. Defaults to
                ``MLFLOW_TRACKING_URI``; tracking stays off when neither is set.
            experiment_name: Experiment runs are recorded under.
            artifact_location: Where model artifacts are stored. Defaults to
                ``MLFLOW_ARTIFACT_ROOT``. Without one MLflow falls back to
                ``./mlruns``, so artifacts land in the process's working
                directory even when the tracking store is elsewhere.
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location or os.getenv("MLFLOW_ARTIFACT_ROOT")
        self.tracking_enabled = bool(self.tracking_uri) and mlflow is not None
        if self.tracking_enabled and self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._ensure_experiment()
        #: Identifiers of the most recent tracked run, for the model registry.
        self.last_run_id: str | None = None
        self.last_model_uri: str | None = None
        self.logger = logger.bind(service="model-training")

    def _ensure_experiment(self) -> None:
        """Select the experiment, creating it with the configured artifact root.

        ``mlflow.set_experiment`` creates a missing experiment using MLflow's
        default artifact location, which cannot be changed afterwards. The
        experiment therefore has to be created explicitly the first time.
        """
        existing = mlflow.get_experiment_by_name(self.experiment_name)
        if existing is None:
            mlflow.create_experiment(
                self.experiment_name, artifact_location=self.artifact_location
            )
        mlflow.set_experiment(self.experiment_name)

    async def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: ModelConfig | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[Any, dict[str, float]]:
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

        explanations = None
        if config.explainability:
            explainer = ModelExplainer()
            explanations = explainer.explain(model, X, config.explainability)

        if self.tracking_enabled:
            self._log_run(config, params, model, metrics, X, explanations)

        return model, metrics

    def _log_run(
        self,
        config: ModelConfig,
        params: dict[str, Any],
        model: Any,
        metrics: dict[str, float],
        X: np.ndarray,
        explanations: dict[str, Any] | None,
    ) -> None:
        """Record one training run, including the model artifact.

        Everything is written inside an explicit run. Logging outside a run
        makes MLflow start an implicit one and materialise a tracking store in
        the current working directory.
        """
        with mlflow.start_run(run_name=config.algorithm) as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)
            feature_names = getattr(model, "feature_names_in_", None)
            drift_reference = DriftReference.from_array(X, feature_names)
            mlflow.log_dict(drift_reference.to_dict(), DRIFT_REFERENCE_ARTIFACT)
            if explanations:
                for key, value in explanations.items():
                    mlflow.log_metric(
                        f"{key}_rows",
                        float(getattr(value, "shape", (len(value),))[0]),
                    )
            self.last_run_id = run.info.run_id
            try:
                info = mlflow.sklearn.log_model(
                    model,
                    name="model",
                    input_example=X[:1],
                    # Pin the format. MLflow picks skops when it happens to be
                    # installed, and skops refuses to serialise types it does
                    # not trust, so the artifact would be logged or silently
                    # skipped depending on what else is in the environment.
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                self.last_model_uri = info.model_uri
            except Exception as exc:  # pragma: no cover - flavour mismatch
                self.last_model_uri = None
                self.logger.warning("model_artifact_not_logged", error=str(exc))

    def _build_trainer(self, config: ModelConfig) -> Any:
        algo = config.algorithm.lower()
        if algo == "xgboost":
            return XGBoostTrainer(params=config.params, distributed=config.distributed)
        if algo == "lightgbm":
            return LightGBMTrainer(params=config.params, distributed=config.distributed)
        if algo == "ensemble":
            if not config.ensemble:
                raise ValueError(
                    "ensemble configuration required for ensemble algorithm"
                )
            return EnsembleTrainer(**config.ensemble)
        raise ValueError(f"Unknown algorithm {config.algorithm}")
