from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class ForecastingModels:
    """Collection of light-weight forecasting models."""

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=10, random_state=0),
        }

    def fit(self, name: str, X: np.ndarray, y: np.ndarray) -> Any:
        model = self.models[name]
        model.fit(X, y)
        return model

    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        return self.models[name].predict(X)
