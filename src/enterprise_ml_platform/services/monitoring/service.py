from __future__ import annotations

"""Central monitoring service for the ML platform.

The :class:`MonitoringService` orchestrates metric collection, drift detection,
performance tracking and alert dispatching.  The implementation focuses on
extensibility while keeping the runtime footprint light for tests.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog

from .collectors.metrics_collector import MetricsCollector
from .drift_detection.drift_analyzer import DriftAnalyzer
from .performance.performance_monitor import PerformanceMonitor
from .alerting.alert_manager import AlertManager
from .alerting.rules_engine import Alert, AlertRule, RulesEngine

logger = structlog.get_logger(__name__)


@dataclass
class PredictionEvent:
    """Event information used for monitoring a prediction call."""

    model_name: str
    latency: float
    predicted: float
    actual: Optional[float] = None
    features: Optional[Dict[str, Any]] = None


class MonitoringService:
    """High level facade tying together monitoring components.

    The service exposes a minimal async API used by the prediction service
    or batch jobs to record inference events.  It updates Prometheus metrics,
    analyses drift, tracks model accuracy and triggers alerting rules.

    Example:
        >>> service = MonitoringService()
        >>> event = PredictionEvent("demo", 0.05, 1, 0, {"f1": 0.1})
        >>> asyncio.run(service.handle_event(event))
    """

    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        drift_analyzer: Optional[DriftAnalyzer] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        alert_manager: Optional[AlertManager] = None,
        rules_engine: Optional[RulesEngine] = None,
    ) -> None:
        self.metrics = metrics or MetricsCollector()
        self.drift_analyzer = drift_analyzer or DriftAnalyzer()
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.alert_manager = alert_manager or AlertManager()
        self.rules_engine = rules_engine or RulesEngine()

    async def handle_event(self, event: PredictionEvent) -> None:
        """Process a prediction event and update monitoring state."""
        logger.debug("handle_event", model=event.model_name)
        self.metrics.record_prediction(event.model_name, event.latency)

        metric_values: Dict[str, float] = {}

        if event.actual is not None:
            accuracy = self.performance_monitor.update(
                event.model_name, event.actual, event.predicted
            )
            self.metrics.set_accuracy(event.model_name, accuracy)
            metric_values[f"{event.model_name}_accuracy"] = accuracy

        if event.features:
            drift_scores = self.drift_analyzer.check(event.features)
            for feature, score in drift_scores.items():
                self.metrics.set_drift(feature, score)
                metric_values[f"{feature}_drift"] = score

        alerts = self.rules_engine.evaluate(metric_values)
        if alerts:
            await self.alert_manager.dispatch(alerts)

    async def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule with the service."""
        self.rules_engine.add_rule(rule)

    async def close(self) -> None:
        """Release resources held by the monitoring components."""
        await self.alert_manager.close()
