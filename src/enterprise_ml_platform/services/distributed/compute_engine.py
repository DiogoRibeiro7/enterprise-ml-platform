from __future__ import annotations
"""Distributed computing orchestrator.

The :class:`ComputeEngine` glues together the various cluster managers, the
resource tracker, scheduler, monitor and failure detector.  The implementation
is intentionally compact but showcases how different distributed frameworks
could be abstracted behind a single interface.
"""

from typing import Any, Callable

import structlog

from .ray.ray_cluster_manager import RayClusterManager
from .dask.dask_cluster_manager import DaskClusterManager
from .spark.spark_cluster_manager import SparkClusterManager
from .scheduling.task_scheduler import TaskScheduler
from .resource.resource_manager import ResourceManager
from .fault_tolerance.failure_detector import FailureDetector
from .load_balancing.load_balancer import LoadBalancer
from .monitoring.cluster_monitor import ClusterMonitor

logger = structlog.get_logger(__name__)


class ComputeEngine:
    """Coordinate distributed computation across multiple frameworks."""

    def __init__(self) -> None:
        self.ray_manager = RayClusterManager()
        self.dask_manager = DaskClusterManager()
        self.spark_manager = SparkClusterManager()
        self.resource_manager = ResourceManager()
        self.load_balancer = LoadBalancer(
            {
                "ray": self.ray_manager,
                "dask": self.dask_manager,
                "spark": self.spark_manager,
            }
        )
        self.scheduler = TaskScheduler(self.load_balancer)
        self.failure_detector = FailureDetector()
        self.monitor = ClusterMonitor(self.load_balancer.managers)

    # ------------------------------------------------------------------ clusters
    def start(self, framework: str) -> None:
        logger.info("starting cluster", framework=framework)
        self.load_balancer.choose_manager(framework).start_cluster()

    def stop(self, framework: str) -> None:
        logger.info("stopping cluster", framework=framework)
        self.load_balancer.choose_manager(framework).stop_cluster()

    # ---------------------------------------------------------------- tasks
    def submit(
        self, fn: Callable[..., Any], framework: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Schedule a task on the given framework."""
        return self.scheduler.schedule(fn, framework, *args, **kwargs)

    # ---------------------------------------------------------------- monitoring
    def metrics(self) -> dict[str, bool]:
        return self.monitor.metrics()
