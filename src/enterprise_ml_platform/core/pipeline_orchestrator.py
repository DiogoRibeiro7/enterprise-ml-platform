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
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Any

import structlog
from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram

logger = structlog.get_logger()


class StageStatus(StrEnum):
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
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


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
    error: BaseException | None = None
    metrics: dict[str, float] = field(default_factory=dict)
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
        dependencies: Iterable[str] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.name = name
        self.dependencies: set[str] = set(dependencies or [])
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
        *,
        metrics_registry: CollectorRegistry | None = None,
    ) -> None:
        self.stages: dict[str, BasePipelineStage] = {
            stage.name: stage for stage in stages
        }
        self.concurrency = max(1, concurrency)
        self.logger = structlog.get_logger().bind(component="orchestrator")
        self._graph = {
            name: set(stage.dependencies) for name, stage in self.stages.items()
        }
        self._dependents: dict[str, set[str]] = defaultdict(set)
        for name, deps in self._graph.items():
            for dep in deps:
                self._dependents[dep].add(name)
        self._validate_graph()
        self._exit_stack = AsyncExitStack()
        self._total = len(self.stages)
        self._completed = 0
        self._init_metrics(metrics_registry)

    async def __aenter__(self) -> PipelineOrchestrator:
        await self._exit_stack.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._exit_stack.__aexit__(exc_type, exc, tb)

    def _init_metrics(self, registry: CollectorRegistry | None = None) -> None:
        """Register pipeline metrics against ``registry``.

        An explicit registry is needed whenever several orchestrators live in
        the same process, since Prometheus rejects duplicate timeseries within
        a single registry.
        """
        self.metrics_registry = registry if registry is not None else REGISTRY
        self.metric_duration = Histogram(
            "pipeline_stage_duration_seconds",
            "Time spent executing a stage",
            ["stage"],
            registry=self.metrics_registry,
        )
        self.metric_success = Counter(
            "pipeline_stage_success_total",
            "Number of successful stage executions",
            ["stage"],
            registry=self.metrics_registry,
        )
        self.metric_failure = Counter(
            "pipeline_stage_failure_total",
            "Number of failed stage executions",
            ["stage"],
            registry=self.metrics_registry,
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

    def generate_execution_graph(self) -> Mapping[str, set[str]]:
        """Return adjacency list representing stage dependencies."""
        return {k: set(v) for k, v in self._graph.items()}

    async def run(self, context: ExecutionContext) -> Mapping[str, StageResult]:
        """Execute the pipeline.

        Stages are scheduled as soon as their dependencies succeed, bounded by
        the configured concurrency. When a stage fails, scheduling stops, every
        outstanding stage is cancelled and awaited, and only then are the
        stages that actually completed rolled back -- in reverse completion
        order, so a compensating action never runs before the action it
        compensates for.

        Args:
            context: Execution context for this run.

        Returns:
            Mapping of stage name to :class:`StageResult`.

        Raises:
            RuntimeError: If any stage fails. Rollback has already run.
        """
        results: dict[str, StageResult] = {}
        completed_order: list[str] = []
        pending: dict[str, set[str]] = {k: set(v) for k, v in self._graph.items()}
        ready = deque([name for name, deps in pending.items() if not deps])
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks: dict[str, asyncio.Task] = {}
        failure: StageResult | None = None
        self._completed = 0

        async def _run_stage(stage_name: str) -> StageResult:
            """Execute a single stage. Never mutates shared state."""
            stage = self.stages[stage_name]
            async with semaphore:
                result = await self._execute_with_retry(stage, context)
            await stage.cleanup(context)
            return result

        while ready or tasks:
            while failure is None and ready and len(tasks) < self.concurrency:
                name = ready.popleft()
                tasks[name] = asyncio.create_task(_run_stage(name))
            if not tasks:
                break

            done, _ = await asyncio.wait(
                tasks.values(), return_when=asyncio.FIRST_COMPLETED
            )
            # Results are recorded here, on the single coordinating coroutine,
            # so ``results`` is never mutated while it is being read.
            for finished in done:
                name = next(k for k, t in tasks.items() if t is finished)
                tasks.pop(name)
                result = self._result_from_task(name, finished)
                results[name] = result
                completed_order.append(name)
                self._completed += 1
                self.logger.info(
                    "stage_completed",
                    stage=name,
                    status=result.status.value,
                    progress=self.progress,
                    duration=result.duration,
                )
                if result.status is StageStatus.SUCCESS:
                    for dependent in self._dependents.get(name, set()):
                        pending[dependent].discard(name)
                        if not pending[dependent]:
                            ready.append(dependent)
                elif failure is None:
                    failure = result

            if failure is not None:
                break

        if failure is not None:
            await self._cancel_outstanding(tasks, results, completed_order)
            await self._rollback(context, results, completed_order)
            raise RuntimeError(f"Stage '{failure.name}' failed") from failure.error

        return results

    @staticmethod
    def _result_from_task(name: str, task: asyncio.Task) -> StageResult:
        """Convert a finished task into a :class:`StageResult`.

        ``_execute_with_retry`` returns a result rather than raising, so an
        exception here means the stage's ``cleanup`` or the task machinery
        itself failed. Either way the stage did not succeed.
        """
        if task.cancelled():
            return StageResult(
                name=name,
                status=StageStatus.SKIPPED,
                error=asyncio.CancelledError(),
                ended_at=time.time(),
            )
        exc = task.exception()
        if exc is not None:
            return StageResult(
                name=name, status=StageStatus.FAILED, error=exc, ended_at=time.time()
            )
        result: StageResult = task.result()
        return result

    async def _cancel_outstanding(
        self,
        tasks: dict[str, asyncio.Task],
        results: dict[str, StageResult],
        completed_order: list[str],
    ) -> None:
        """Cancel in-flight stages and record whatever they managed to do.

        A stage that finished between the failure and the cancellation landing
        has already produced side effects, so it is recorded as completed and
        rolled back with the rest.
        """
        if not tasks:
            return
        items = list(tasks.items())
        for _, task in items:
            task.cancel()
        await asyncio.gather(*(task for _, task in items), return_exceptions=True)
        for name, task in items:
            result = self._result_from_task(name, task)
            results[name] = result
            if result.status is StageStatus.SUCCESS:
                completed_order.append(name)
                self.logger.warning("stage_completed_during_abort", stage=name)
            else:
                self.logger.info("stage_cancelled", stage=name)
        tasks.clear()

    async def _rollback(
        self,
        context: ExecutionContext,
        results: Mapping[str, StageResult],
        completed_order: Iterable[str] | None = None,
    ) -> None:
        """Roll back successful stages in reverse completion order.

        Args:
            context: Execution context for this run.
            results: Results recorded so far.
            completed_order: Order in which stages completed. Defaults to the
                insertion order of ``results``.
        """
        order = list(completed_order) if completed_order is not None else list(results)
        for name in reversed(order):
            result = results.get(name)
            if result is None or result.status is not StageStatus.SUCCESS:
                continue
            try:
                await self.stages[name].rollback(context, result)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error("rollback_failed", stage=name, error=str(exc))

    async def _execute_with_retry(
        self, stage: BasePipelineStage, context: ExecutionContext
    ) -> StageResult:
        """Execute a stage honouring retry and circuit breaker policies."""
        if not stage.circuit_breaker.allow():
            return StageResult(
                name=stage.name,
                status=StageStatus.SKIPPED,
                error=RuntimeError("circuit open"),
            )

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
                if (
                    attempt > stage.retry_policy.max_retries
                    or not stage.circuit_breaker.allow()
                ):
                    result.status = StageStatus.FAILED
                    result.ended_at = time.time()
                    return result
                await asyncio.sleep(delay)
                delay *= stage.retry_policy.backoff_factor
