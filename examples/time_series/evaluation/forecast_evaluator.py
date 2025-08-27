from __future__ import annotations

from typing import Dict
import numpy as np


class ForecastEvaluator:
    """Compute common accuracy metrics for forecasts."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        return {"mae": mae, "rmse": rmse, "mape": mape}
