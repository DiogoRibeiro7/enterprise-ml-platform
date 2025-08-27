"""Pipeline orchestration system for the Enterprise ML Platform.

This module provides a fully asynchronous pipeline engine capable of
executing complex machine learning workflows with dependency management,
parallel execution, retries and resilience mechanisms.

Example:
    >>> import asyncio
    >>> class Ingest(BasePipelineStage):
    ...     async def run(self, context: ExecutionContext) -> str:
    ...         await asyncio.sleep(0.1)
    ...         return "data"
    >>> class Train(BasePipelineStage):
    ...     def __init__(self) -> None:
    ...         super().__init__("train", dependencies={"ingest"})
    ...     async def run(self, context: ExecutionContext) -> str:
    ...         await asyncio.sleep(0.1)
    ...         return "model"
    >>> async def main():
    ...     stages = [Ingest("ingest"), Train()]
    ...     async with PipelineOrchestrator(stages) as orchestrator:
    ...         ctx = ExecutionContext(run_id="demo")
    ...         results = await orchestrator.run(ctx)
    ...     return results["train"].output
    >>> asyncio.run(main())
    'model'
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Set
from abc import ABC, abstractmethod

import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger()


class StageStatus(str, Enum):
    """Execution status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionContext:
    """Holds runtime information for a pipeline run.

    Attributes:
        run_id: Unique identifier for the pipeline execution.
        params: Arbitrary parameters that stages may consume.
        metadata: Metadata collected during the run.
        artifacts: Mapping of artifact names to locations.
    """

    run_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Result produced by a pipeline stage.

    Attributes:
        name: Name of the stage.
        status: Final status of the stage.
        output: Optional payload returned by the stage.
        error: Exception raised during execution, if any.
        metrics: Custom metrics emitted by the stage.
        started_at: Unix timestamp when execution began.
        ended_at: Unix timestamp when execution finished.
    """

    name: str
    status: StageStatus
    output: Any = None
    error: Optional[BaseException] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0

    @property
    def duration(self) -> float:
        """Return execution duration in seconds."""
        return self.ended_at - self.started_at if self.ended_at else 0.0


@dataclass
class RetryPolicy:
    """Configuration for retry behaviour."""

    max_retries: int = 0
    backoff_factor: float = 2.0
    base_delay: float = 1.0


@dataclass
class CircuitBreaker:
    """Simple circuit breaker to guard failing stages."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    failure_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0.0, init=False)
    state: str = field(default="closed", init=False)

    def allow(self) -> bool:
        """Return True if execution is permitted."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        return True

    def record_success(self) -> None:
        """Reset breaker after a successful call."""
        self.state = "closed"
        self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failure and open circuit if threshold exceeded."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class BasePipelineStage(ABC):
    """Abstract base class for all pipeline stages.

    Subclasses must implement :meth:`run`.
    """

    def __init__(
        self,
        name: str,
        dependencies: Optional[Iterable[str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self.name = name
        self.dependencies: Set[str] = set(dependencies or [])
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = CircuitBreaker()
        self.logger = structlog.get_logger().bind(stage=name)

    @abstractmethod
    async def run(self, context: ExecutionContext) -> Any:
        """Execute the stage.

        Args:
            context: Execution context for the pipeline run.

        Returns:
            Result produced by the stage.
        """

    async def rollback(self, context: ExecutionContext, result: StageResult) -> None:
        """Rollback side effects if the pipeline fails downstream."""
        return

    async def cleanup(self, context: ExecutionContext) -> None:
        """Release resources allocated by the stage."""
        return


class PipelineOrchestrator:
    """Coordinate the execution of pipeline stages.

    The orchestrator resolves dependencies between stages and executes them
    concurrently using ``asyncio``. It supports retry policies, circuit
    breakers, metrics collection and progress reporting.

    Example:
        async with PipelineOrchestrator(stages) as orchestrator:
            context = ExecutionContext(run_id="123")
            results = await orchestrator.run(context)
    """

    def __init__(
        self,
        stages: Iterable[BasePipelineStage],
        concurrency: int = 4,
    ) -> None:
        self.stages: Dict[str, BasePipelineStage] = {
            stage.name: stage for stage in stages
        }
        self.concurrency = max(1, concurrency)
        self.logger = structlog.get_logger().bind(component="orchestrator")
        self._graph = {name: set(stage.dependencies) for name, stage in self.stages.items()}
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        for name, deps in self._graph.items():
            for dep in deps:
                self._dependents[dep].add(name)
        self._validate_graph()
        self._exit_stack = AsyncExitStack()
        self._total = len(self.stages)
        self._completed = 0
        self._init_metrics()

    async def __aenter__(self) -> "PipelineOrchestrator":
        await self._exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._exit_stack.__aexit__(exc_type, exc, tb)

    def _init_metrics(self) -> None:
        self.metric_duration = Histogram(
            "pipeline_stage_duration_seconds", "Time spent executing a stage", ["stage"]
        )
        self.metric_success = Counter(
            "pipeline_stage_success_total", "Number of successful stage executions", ["stage"]
        )
        self.metric_failure = Counter(
            "pipeline_stage_failure_total", "Number of failed stage executions", ["stage"]
        )

    def _validate_graph(self) -> None:
        in_degree = {name: len(deps) for name, deps in self._graph.items()}
        queue = deque([n for n, d in in_degree.items() if d == 0])
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in self._dependents.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if visited != len(self._graph):
            raise ValueError("Cyclic dependencies detected in pipeline")

    @property
    def progress(self) -> float:
        """Return pipeline execution progress between 0 and 1."""
        return self._completed / self._total if self._total else 1.0

    def generate_execution_graph(self) -> Mapping[str, Set[str]]:
        """Return adjacency list representing stage dependencies."""
        return {k: set(v) for k, v in self._graph.items()}

    async def run(self, context: ExecutionContext) -> Mapping[str, StageResult]:
        """Execute the pipeline.

        Args:
            context: Execution context for this run.

        Returns:
            Mapping of stage name to :class:`StageResult`.
        """
        results: Dict[str, StageResult] = {}
        pending: Dict[str, Set[str]] = {k: set(v) for k, v in self._graph.items()}
        ready = deque([name for name, deps in pending.items() if not deps])
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks: Dict[str, asyncio.Task] = {}

        async def _run_stage(stage_name: str) -> None:
            stage = self.stages[stage_name]
            async with semaphore:
                result = await self._execute_with_retry(stage, context)
                results[stage_name] = result
                await stage.cleanup(context)
                self._completed += 1
                self.logger.info(
                    "stage_completed",
                    stage=stage_name,
                    status=result.status.value,
                    progress=self.progress,
                    duration=result.duration,
                )
                if result.status is StageStatus.SUCCESS:
                    for dependent in self._dependents.get(stage_name, set()):
                        pending[dependent].discard(stage_name)
                        if not pending[dependent]:
                            ready.append(dependent)
                else:
                    await self._rollback(context, results)
                    raise RuntimeError(f"Stage '{stage_name}' failed") from result.error

        while ready or tasks:
            while ready and len(tasks) < self.concurrency:
                name = ready.popleft()
                tasks[name] = asyncio.create_task(_run_stage(name))
            if not tasks:
                continue
            done, _ = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                for key, task in list(tasks.items()):
                    if task is finished:
                        if task.exception():
                            await asyncio.gather(*tasks.values(), return_exceptions=True)
                            raise task.exception()
                        tasks.pop(key)
        return results

    async def _rollback(
        self, context: ExecutionContext, results: Mapping[str, StageResult]
    ) -> None:
        """Rollback all successfully executed stages."""
        for name, result in results.items():
            if result.status is StageStatus.SUCCESS:
                try:
                    await self.stages[name].rollback(context, result)
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.error("rollback_failed", stage=name, error=str(exc))

    async def _execute_with_retry(
        self, stage: BasePipelineStage, context: ExecutionContext
    ) -> StageResult:
        """Execute a stage honouring retry and circuit breaker policies."""
        if not stage.circuit_breaker.allow():
            return StageResult(name=stage.name, status=StageStatus.SKIPPED, error=RuntimeError("circuit open"))

        attempt = 0
        delay = stage.retry_policy.base_delay
        while True:
            attempt += 1
            result = StageResult(name=stage.name, status=StageStatus.RUNNING)
            try:
                result.started_at = time.time()
                output = await stage.run(context)
                result.output = output
                result.status = StageStatus.SUCCESS
                result.ended_at = time.time()
                stage.circuit_breaker.record_success()
                self.metric_duration.labels(stage=stage.name).observe(result.duration)
                self.metric_success.labels(stage=stage.name).inc()
                return result
            except Exception as exc:
                result.error = exc
                stage.circuit_breaker.record_failure()
                self.metric_failure.labels(stage=stage.name).inc()
                if attempt > stage.retry_policy.max_retries or not stage.circuit_breaker.allow():
                    result.status = StageStatus.FAILED
                    result.ended_at = time.time()
                    return result
                await asyncio.sleep(delay)
                delay *= stage.retry_policy.backoff_factor
