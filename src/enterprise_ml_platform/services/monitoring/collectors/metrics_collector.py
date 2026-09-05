"""Prometheus metrics shared by the platform services."""

from __future__ import annotations

from contextlib import suppress

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
            "Total successful prediction outputs",
            ["model", "version"],
            registry=self.registry,
        )
        self.prediction_requests = Counter(
            "ml_prediction_requests_total",
            "Total inference requests by outcome",
            ["model", "version", "outcome"],
            registry=self.registry,
        )
        self.prediction_latency = Histogram(
            "ml_prediction_latency_seconds",
            "Model inference latency",
            ["model", "version", "outcome"],
            registry=self.registry,
        )
        self.accuracy = Gauge(
            "ml_model_accuracy", "Model accuracy", ["model"], registry=self.registry
        )
        self.drift = Gauge(
            "ml_feature_drift_score",
            "Feature drift score",
            ["model", "version", "feature"],
            registry=self.registry,
        )
        self.drift_detected = Gauge(
            "ml_feature_drift_detected",
            "Whether a feature drift score meets the configured threshold",
            ["model", "version", "feature"],
            registry=self.registry,
        )
        self.drift_ready = Gauge(
            "ml_drift_monitor_ready",
            "Whether enough serving rows exist to evaluate drift",
            ["model", "version"],
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

    def record_prediction(
        self,
        model: str,
        latency: float,
        *,
        version: str = "unknown",
        item_count: int = 1,
    ) -> None:
        """Record a successful inference request.

        ``prediction_requests`` counts API calls, while ``prediction_total``
        counts scored rows. Keeping both prevents a large batch from looking
        operationally identical to a single prediction.

        Args:
            model: Stable model name.
            latency: Model inference latency in seconds.
            version: Immutable registry version, or ``unknown`` for legacy
                callers that do not expose one yet.
            item_count: Number of rows scored by the request.

        Raises:
            ValueError: If ``item_count`` is not positive.
        """
        if item_count < 1:
            raise ValueError("item_count must be at least 1")
        self.prediction_requests.labels(model, version, "success").inc()
        self.prediction_total.labels(model, version).inc(item_count)
        self.prediction_latency.labels(model, version, "success").observe(latency)

    def record_prediction_error(
        self,
        model: str,
        latency: float,
        *,
        version: str = "unknown",
    ) -> None:
        """Record a model inference failure without counting scored rows."""
        self.prediction_requests.labels(model, version, "error").inc()
        self.prediction_latency.labels(model, version, "error").observe(latency)

    def record_feature_serving(self, store: str, latency: float, hit: bool) -> None:
        self.feature_latency.labels(store).observe(latency)
        if hit:
            self.feature_cache_hits.labels(store).inc()
        else:
            self.feature_cache_misses.labels(store).inc()

    def set_accuracy(self, model: str, value: float) -> None:
        self.accuracy.labels(model).set(value)

    def set_drift(
        self,
        feature: str,
        score: float,
        *,
        model: str = "unknown",
        version: str = "unknown",
        detected: bool | None = None,
    ) -> None:
        """Publish a feature drift score for an immutable model version."""
        self.drift.labels(model, version, feature).set(score)
        if detected is not None:
            self.drift_detected.labels(model, version, feature).set(int(detected))

    def set_drift_ready(self, model: str, version: str, *, ready: bool) -> None:
        """Publish whether a serving window can be evaluated yet."""
        self.drift_ready.labels(model, version).set(int(ready))

    def clear_drift(self, model: str, version: str, features: tuple[str, ...]) -> None:
        """Remove metric children for an unloaded model version."""
        for feature in features:
            for gauge in (self.drift, self.drift_detected):
                with suppress(KeyError):
                    gauge.remove(model, version, feature)
        with suppress(KeyError):
            self.drift_ready.remove(model, version)
