import numpy as np
import pytest

from enterprise_ml_platform.services.model_training.optimization.distributed import (
    RayOptimizer,
)
from enterprise_ml_platform.services.model_training.optimization.hyperparameter_optimizer import (
    HyperparameterOptimizer,
)
from enterprise_ml_platform.services.model_training.optimization.optimizers import (
    BayesianOptimizer,
)


class DummyTrainer:
    def __init__(self, params):
        self.params = params

    def train(self, X, y):  # pragma: no cover - no training required
        return None

    def evaluate(self, model, X, y):
        x = self.params["x"]
        return {"score": -((x - 2) ** 2)}


def trainer_factory(params):
    return DummyTrainer(params)


@pytest.mark.asyncio
async def test_bayesian_optimizer_finds_optimum():
    np.random.seed(0)
    optimizer = BayesianOptimizer(n_init=2)
    best = await optimizer.optimize(
        trainer_factory,
        None,
        None,
        {"params": {"x": {"type": "float", "low": -5, "high": 5}}, "n_trials": 8},
    )
    assert abs(best["x"] - 2) < 1.0


@pytest.mark.asyncio
async def test_hyperparameter_optimizer_selects_bayesian():
    np.random.seed(0)
    optimizer = HyperparameterOptimizer()
    best = await optimizer.optimize(
        trainer_factory,
        None,
        None,
        {
            "algorithm": "bayesian",
            "params": {"x": {"type": "float", "low": -5, "high": 5}},
            "n_trials": 8,
        },
    )
    assert abs(best["x"] - 2) < 1.0


@pytest.mark.asyncio
async def test_ray_optimizer_optional():
    # importorskip states the dependency without leaving an "unused" import
    # that a linter will helpfully delete, taking the guard with it.
    pytest.importorskip("ray.tune", reason="ray[tune] not installed")
    optimizer = RayOptimizer()
    best = await optimizer.optimize(
        trainer_factory,
        None,
        None,
        {"params": {"x": {"type": "float", "low": -5, "high": 5}}, "n_trials": 1},
    )
    assert "x" in best
