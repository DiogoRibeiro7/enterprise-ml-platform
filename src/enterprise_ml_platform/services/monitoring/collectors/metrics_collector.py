from __future__ import annotations

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram


class MetricsCollector:
    """Create and update core Prometheus metrics.

    Metrics are registered against ``registry``.  Passing an explicit registry
    is required whenever more than one collector may live in the same process
    (tests, multi-tenant workers), because Prometheus refuses to register the
    same timeseries twice in a single registry.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else REGISTRY
        self.prediction_total = Counter(
            "ml_predictions_total",
            "Total predictions",
            ["model"],
            registry=self.registry,
        )
        self.prediction_latency = Histogram(
            "ml_prediction_latency_seconds",
            "Prediction latency",
            ["model"],
            registry=self.registry,
        )
        self.accuracy = Gauge(
            "ml_model_accuracy", "Model accuracy", ["model"], registry=self.registry
        )
        self.drift = Gauge(
            "ml_feature_drift_score",
            "Feature drift score",
            ["feature"],
            registry=self.registry,
        )
        # Feature store metrics
        self.feature_latency = Histogram(
            "ml_feature_serving_latency_seconds",
            "Latency of feature serving",
            ["store"],
            registry=self.registry,
        )
        self.feature_cache_hits = Counter(
            "ml_feature_cache_hits_total",
            "Feature store cache hits",
            ["store"],
            registry=self.registry,
        )
        self.feature_cache_misses = Counter(
            "ml_feature_cache_misses_total",
            "Feature store cache misses",
            ["store"],
            registry=self.registry,
        )

    def record_prediction(self, model: str, latency: float) -> None:
        self.prediction_total.labels(model).inc()
        self.prediction_latency.labels(model).observe(latency)

    def record_feature_serving(self, store: str, latency: float, hit: bool) -> None:
        self.feature_latency.labels(store).observe(latency)
        if hit:
            self.feature_cache_hits.labels(store).inc()
        else:
            self.feature_cache_misses.labels(store).inc()

    def set_accuracy(self, model: str, value: float) -> None:
        self.accuracy.labels(model).set(value)

    def set_drift(self, feature: str, score: float) -> None:
        self.drift.labels(feature).set(score)
