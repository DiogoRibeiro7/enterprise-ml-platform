from __future__ import annotations

"""Prometheus metrics collector helpers."""

from prometheus_client import Counter, Gauge, Histogram


class MetricsCollector:
    """Create and update core Prometheus metrics."""

    def __init__(self) -> None:
        self.prediction_total = Counter(
            "ml_predictions_total", "Total predictions", ["model"]
        )
        self.prediction_latency = Histogram(
            "ml_prediction_latency_seconds", "Prediction latency", ["model"]
        )
        self.accuracy = Gauge("ml_model_accuracy", "Model accuracy", ["model"])
        self.drift = Gauge(
            "ml_feature_drift_score", "Feature drift score", ["feature"]
        )

    def record_prediction(self, model: str, latency: float) -> None:
        self.prediction_total.labels(model).inc()
        self.prediction_latency.labels(model).observe(latency)

    def set_accuracy(self, model: str, value: float) -> None:
        self.accuracy.labels(model).set(value)

    def set_drift(self, feature: str, score: float) -> None:
        self.drift.labels(feature).set(score)
