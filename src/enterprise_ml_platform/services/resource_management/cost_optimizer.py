"""Orchestrate cost tracking and resource allocation."""

from __future__ import annotations

from .allocation.dynamic_scaler import DynamicScaler
from .allocation.gpu_scheduler import GPUScheduler
from .allocation.queue_manager import QueueManager
from .allocation.spot_manager import SpotManager
from .monitoring.anomaly_detector import AnomalyDetector
from .monitoring.cost_tracker import CostTracker
from .monitoring.usage_analyzer import UsageAnalyzer


class CostOptimizer:
    """Coordinate cost monitoring and basic resource allocation."""

    def __init__(
        self,
        tracker: CostTracker | None = None,
        analyzer: UsageAnalyzer | None = None,
        scaler: DynamicScaler | None = None,
        queue: QueueManager | None = None,
        gpu: GPUScheduler | None = None,
        spot: SpotManager | None = None,
        anomaly: AnomalyDetector | None = None,
    ) -> None:
        self.tracker = tracker or CostTracker()
        self.analyzer = analyzer or UsageAnalyzer()
        self.scaler = scaler or DynamicScaler()
        self.queue = queue or QueueManager()
        self.gpu = gpu or GPUScheduler(total_gpus=0)
        self.spot = spot or SpotManager()
        self.anomaly = anomaly or AnomalyDetector()

    async def record_cost(
        self, project: str, user: str, model: str, amount: float
    ) -> None:
        """Record cost and run anomaly detection."""
        self.tracker.record_cost(project, user, model, amount)
        self.analyzer.record_usage(project, "cost", amount)

    async def submit_job(self, job_id: str, resources: dict[str, int]) -> None:
        """Register a job and allocate required resources."""
        self.queue.enqueue(job_id, resources)
        gpus = resources.get("gpus", 0)
        if gpus:
            self.gpu.allocate(job_id, gpus)

    def scale_decision(self, utilization: float) -> str:
        """Return scaling decision based on current utilization."""
        return self.scaler.decide(utilization)

    def detect_cost_anomaly(self, project: str) -> bool:
        """Check if latest cost entry for project is anomalous."""
        history = self.tracker.get_cost_history(project)
        return self.anomaly.detect(history)
