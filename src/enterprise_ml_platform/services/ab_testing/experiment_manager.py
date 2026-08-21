"""Manage A/B testing experiments and traffic routing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .decision_engine import DecisionEngine
from .monitoring.experiment_tracker import ExperimentTracker
from .statistical_analyzer import StatisticalAnalyzer
from .traffic_router import TrafficRouter


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    variants: dict[str, str]  # variant name -> model identifier
    traffic_split: dict[str, float]
    success_metric: str = "conversion"  # metric tracked for significance
    geo_overrides: dict[str, str] = field(default_factory=dict)
    demo_overrides: dict[str, str] = field(default_factory=dict)


class ExperimentManager:
    """Coordinate experiments, routing, tracking, and analysis."""

    def __init__(
        self,
        analyzer: StatisticalAnalyzer | None = None,
        decision_engine: DecisionEngine | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.decision_engine = decision_engine or DecisionEngine(self.analyzer)
        self.tracker = tracker or ExperimentTracker()
        self._experiments: dict[str, ExperimentConfig] = {}
        self._routers: dict[str, TrafficRouter] = {}
        self._lock = asyncio.Lock()

    async def create_experiment(self, cfg: ExperimentConfig) -> None:
        async with self._lock:
            self._experiments[cfg.name] = cfg
            self._routers[cfg.name] = TrafficRouter(cfg)

    async def get_variant(
        self, experiment: str, session_id: str, attributes: dict[str, Any] | None = None
    ) -> str:
        router = self._routers[experiment]
        variant = router.route(session_id, attributes or {})
        self.tracker.record_assignment(experiment, variant)
        return variant

    async def record_outcome(
        self, experiment: str, variant: str, value: float, success: bool
    ) -> None:
        self.tracker.record_outcome(experiment, variant, value, success)

    async def analyze(self, experiment: str) -> dict[str, Any]:
        data = self.tracker.get_metrics(experiment)
        return self.analyzer.analyze(data)

    async def decide(self, experiment: str, criteria: dict[str, Any]) -> dict[str, Any]:
        analysis = await self.analyze(experiment)
        decision = self.decision_engine.decide(experiment, analysis, criteria)
        return {"analysis": analysis, "decision": decision}

    async def update_split(self, experiment: str, new_split: dict[str, float]) -> None:
        async with self._lock:
            cfg = self._experiments[experiment]
            cfg.traffic_split = new_split
            self._routers[experiment].update_split(new_split)
