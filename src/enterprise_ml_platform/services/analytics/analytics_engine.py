"""Analytics orchestration engine.

This module glues together the various analytics subcomponents
(dashboards, metrics, reporting, alerts, ...).  The implementation is a
minimal synchronous façade that demonstrates how these pieces could be
composed.  Heavy lifting such as persistence, authentication or real
visualisation rendering would be provided by specialised services in a
full deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

from .dashboards.dashboard_builder import DashboardBuilder
from .metrics.business_metrics import BusinessMetrics
from .reporting.report_generator import ReportGenerator
from .visualization.chart_builder import ChartBuilder
from .intelligence.insight_engine import InsightEngine
from .export.data_exporter import DataExporter
from .scheduling.report_scheduler import ReportScheduler
from .alerts.kpi_alerting import KPIAlerting


@dataclass
class AnalyticsEngine:
    """High level entry point for analytics tasks.

    The engine wires together the individual components and exposes a
    simplified API used in tests and examples.  Each component can be
    swapped out with more sophisticated implementations as the platform
    evolves.
    """

    dashboard_builder: DashboardBuilder = field(default_factory=DashboardBuilder)
    metrics: BusinessMetrics = field(default_factory=BusinessMetrics)
    reporter: ReportGenerator = field(default_factory=ReportGenerator)
    charts: ChartBuilder = field(default_factory=ChartBuilder)
    insights: InsightEngine = field(default_factory=InsightEngine)
    exporter: DataExporter = field(default_factory=DataExporter)
    scheduler: ReportScheduler = field(default_factory=ReportScheduler)
    alerting: KPIAlerting = field(default_factory=KPIAlerting)

    def run(self, data: Iterable[Dict[str, Any]], kpi_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Execute a full analytics cycle on ``data``.

        Args:
            data: Iterable of records to analyse.
            kpi_thresholds: Optional mapping of KPI name to alert threshold.

        Returns:
            Dictionary containing metrics, insights and report text.
        """
        metrics = self.metrics.compute_kpis(data)
        charts = self.charts.build_charts(data, metrics)
        dashboard = self.dashboard_builder.build(metrics, charts)
        insights = self.insights.generate(metrics)
        report = self.reporter.generate(metrics, insights)

        if kpi_thresholds:
            self.alerting.check(metrics, kpi_thresholds)

        return {
            "metrics": metrics,
            "dashboard": dashboard,
            "insights": insights,
            "report": report,
        }

    def schedule_report(self, cron: str, job: Callable[[], None]) -> None:
        """Register a reporting job with the internal scheduler."""
        self.scheduler.add_job(cron, job)

    def export(self, data: Iterable[Dict[str, Any]], path: str, fmt: str = "csv") -> None:
        """Convenience wrapper around :class:`DataExporter`."""
        self.exporter.export(data, path, fmt)
