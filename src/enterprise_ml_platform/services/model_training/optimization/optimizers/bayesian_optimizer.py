from __future__ import annotations

"""Simple Bayesian optimisation using Gaussian Processes."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import asyncio
import math
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


@dataclass
class BayesianOptimizer:
    """Bayesian optimiser leveraging a Gaussian Process surrogate model."""

    n_init: int = 3

    async def optimize(
        self,
        trainer_factory,
        X,
        y,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        params_config = config["params"]
        n_trials = config.get("n_trials", 10)
        param_names: List[str] = []
        bounds: List[Tuple[float, float]] = []
        types: List[str] = []
        for name, spec in params_config.items():
            param_names.append(name)
            types.append(spec["type"])
            bounds.append((float(spec["low"]), float(spec["high"])) )

        async def evaluate(vec: np.ndarray) -> float:
            params: Dict[str, Any] = {}
            for i, name in enumerate(param_names):
                val = vec[i]
                if types[i] == "int":
                    val = int(round(val))
                params[name] = val
            trainer = trainer_factory(params)
            model = trainer.train(X, y)
            metrics = trainer.evaluate(model, X, y)
            return list(metrics.values())[0]

        def sample_random() -> np.ndarray:
            vals = []
            for (low, high), t in zip(bounds, types):
                if t == "int":
                    vals.append(np.random.randint(math.floor(low), math.ceil(high) + 1))
                else:
                    vals.append(np.random.uniform(low, high))
            return np.array(vals, dtype=float)

        X_samples: List[np.ndarray] = []
        y_samples: List[float] = []

        for _ in range(min(self.n_init, n_trials)):
            x0 = sample_random()
            y0 = await evaluate(x0)
            X_samples.append(x0)
            y_samples.append(y0)

        while len(X_samples) < n_trials:
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
            )
            X_arr = np.vstack(X_samples)
            y_arr = np.array(y_samples)
            gp.fit(X_arr, y_arr)

            candidates = np.array([sample_random() for _ in range(100)])
            mu, sigma = gp.predict(candidates, return_std=True)
            best = np.max(y_arr)
            with np.errstate(divide="warn"):
                z = (mu - best) / sigma
                ei = (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)
                ei[sigma == 0.0] = 0.0
            x_next = candidates[int(np.argmax(ei))]
            y_next = await evaluate(x_next)
            X_samples.append(x_next)
            y_samples.append(y_next)

        best_idx = int(np.argmax(y_samples))
        best_vec = X_samples[best_idx]
        best_params: Dict[str, Any] = {}
        for i, name in enumerate(param_names):
            val = best_vec[i]
            if types[i] == "int":
                val = int(round(val))
            best_params[name] = val
        return best_params
