from __future__ import annotations

"""Hyperparameter optimization utilities."""

from dataclasses import dataclass
from typing import Any, Dict

import asyncio
import structlog

try:  # pragma: no cover - optional dependency
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore

logger = structlog.get_logger()


@dataclass
class HyperparameterOptimizer:
    """Wrapper around Optuna studies."""

    study_name: str = "model-optimization"

    async def optimize(
        self,
        trainer_factory,
        X,
        y,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run hyperparameter optimisation.

        Args:
            trainer_factory: Callable returning a ``ModelTrainer`` when passed
                a parameter dictionary. Using a factory allows new trainers to
                be instantiated for each trial.
            X: Training features.
            y: Training targets.
            config: Optimization configuration. Must include ``params`` and
                may specify ``n_trials`` and ``direction``.
        Returns:
            Best set of hyperparameters identified by the optimizer.
        """
        if optuna is None:  # pragma: no cover - runtime check
            raise ImportError("optuna is required for hyperparameter optimization")

        def objective(trial: "optuna.trial.Trial") -> float:
            params: Dict[str, Any] = {}
            for name, spec in config["params"].items():
                if spec["type"] == "int":
                    params[name] = trial.suggest_int(name, spec["low"], spec["high"])
                elif spec["type"] == "float":
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"], log=spec.get("log", False)
                    )
                else:
                    params[name] = trial.suggest_categorical(name, spec["choices"])
            trainer = trainer_factory(params)
            model = trainer.train(X, y)
            metrics = trainer.evaluate(model, X, y)
            # Assume maximization of the first metric returned
            return list(metrics.values())[0]

        study = optuna.create_study(direction=config.get("direction", "maximize"), study_name=self.study_name)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: study.optimize(objective, n_trials=config.get("n_trials", 10)),
        )
        return study.best_params
