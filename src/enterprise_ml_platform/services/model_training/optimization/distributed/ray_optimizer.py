"""Distributed hyperparameter optimisation via Ray Tune."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional dependency
    import ray
    from ray import tune
except Exception:  # pragma: no cover
    ray = None  # type: ignore
    tune = None  # type: ignore


@dataclass
class RayOptimizer:
    """Wrapper around Ray Tune for distributed optimisation."""

    metric: str = "metric"

    async def optimize(
        self,
        trainer_factory,
        X,
        y,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if ray is None or tune is None:  # pragma: no cover - runtime check
            raise ImportError("ray[tune] is required for distributed optimization")

        search_space: dict[str, Any] = {}
        for name, spec in config["params"].items():
            if spec["type"] in {"int", "float"}:
                search_space[name] = tune.uniform(spec["low"], spec["high"])
            else:
                search_space[name] = tune.choice(spec["choices"])

        def objective(trial_config):
            trainer = trainer_factory(trial_config)
            model = trainer.train(X, y)
            metrics = trainer.evaluate(model, X, y)
            tune.report(**{self.metric: list(metrics.values())[0]})

        tuner = tune.Tuner(
            objective,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                mode=config.get("direction", "max"),
                metric=self.metric,
                num_samples=config.get("n_trials", 10),
            ),
        )
        result = await asyncio.get_event_loop().run_in_executor(None, tuner.fit)
        return result.get_best_result(
            metric=self.metric, mode=config.get("direction", "max")
        ).config
