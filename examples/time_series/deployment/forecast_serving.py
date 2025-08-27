from __future__ import annotations

from typing import Any
import numpy as np


class ForecastServing:
    """Simple wrapper around a trained model for serving forecasts."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
