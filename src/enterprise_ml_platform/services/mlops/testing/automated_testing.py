"""Light‑weight automated testing helpers for ML models."""

from __future__ import annotations

from typing import Any

import numpy as np


class AutomatedTesting:
    """Run a minimal suite of tests against a model."""

    def run(
        self, model: Any, X: np.ndarray | None, y: np.ndarray | None
    ) -> dict[str, float]:
        if X is None or y is None:
            return {"tests": 0}
        preds = model.predict(X)
        accuracy = float(np.mean(preds == y))
        return {"accuracy": accuracy}
