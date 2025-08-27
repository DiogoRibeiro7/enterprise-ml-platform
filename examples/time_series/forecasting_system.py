from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .preprocessing.ts_preprocessor import TSPreprocessor
from .models.forecasting_models import ForecastingModels
from .evaluation.forecast_evaluator import ForecastEvaluator
from .selection.model_selector import ModelSelector
from .visualization.ts_visualizer import TSVisualizer
from .decomposition.seasonal_decomposer import SeasonalDecomposer
from .anomaly.anomaly_detector import TSAnomalyDetector
from .deployment.forecast_serving import ForecastServing


class TimeSeriesForecastingSystem:
    """High level orchestrator for time series forecasting workflows."""

    def __init__(self, freq: str = "D") -> None:
        self.preprocessor = TSPreprocessor(freq)
        self.models = ForecastingModels()
        self.evaluator = ForecastEvaluator()
        self.selector = ModelSelector()
        self.visualizer = TSVisualizer()
        self.decomposer = SeasonalDecomposer()
        self.anomaly_detector = TSAnomalyDetector()
        self.serving: ForecastServing | None = None

    def run(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Train all models and return metrics for the best one."""
        df = self.preprocessor.fit_transform(df, target)
        X = df.drop(columns=[target]).values
        y = df[target].values
        results: Dict[str, Dict[str, float]] = {}
        for name in self.models.models:
            model = self.models.fit(name, X, y)
            preds = self.models.predict(name, X)
            results[name] = self.evaluator.evaluate(y, preds)
        best_name, best_metrics = self.selector.select(results)
        self.serving = ForecastServing(self.models.models[best_name])
        return best_metrics

    def forecast(self, future_features: np.ndarray) -> np.ndarray:
        if not self.serving:
            raise RuntimeError("system not yet trained")
        return self.serving.predict(future_features)
