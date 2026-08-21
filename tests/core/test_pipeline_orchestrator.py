"""Regression tests for :mod:`enterprise_ml_platform.core.pipeline_orchestrator`.

These cover the failure path specifically: a stage failing while sibling
stages are still in flight must not leave un-compensated side effects behind.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

import pytest
from prometheus_client import CollectorRegistry

from enterprise_ml_platform.core.pipeline_orchestrator import (
    BasePipelineStage,
    ExecutionContext,
    PipelineOrchestrator,
    RetryPolicy,
    StageStatus,
)


class RecordingStage(BasePipelineStage):
    """Stage that appends every side effect it performs to a shared list."""

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        dependencies: Iterable[str] | None = None,
        delay: float = 0.0,
        fail: bool = False,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(name, dependencies=dependencies, retry_policy=retry_policy)
        self.events = events
        self.delay = delay
        self.fail = fail

    async def run(self, context: ExecutionContext) -> str:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ValueError(f"{self.name} exploded")
        # Reaching this line means the stage committed a side effect.
        self.events.append(f"run:{self.name}")
        return self.name

    async def rollback(self, context: ExecutionContext, result) -> None:
        self.events.append(f"rollback:{self.name}")


def _orchestrator(stages, **kwargs) -> PipelineOrchestrator:
    """Build an orchestrator on an isolated Prometheus registry."""
    return PipelineOrchestrator(stages, metrics_registry=CollectorRegistry(), **kwargs)


# ----------------------------------------------------------------------
# Graph handling
# ----------------------------------------------------------------------
async def test_runs_stages_in_dependency_order() -> None:
    events: list[str] = []
    stages = [
        RecordingStage("ingest", events),
        RecordingStage("train", events, dependencies={"ingest"}),
        RecordingStage("deploy", events, dependencies={"train"}),
    ]
    async with _orchestrator(stages) as orch:
        results = await orch.run(ExecutionContext(run_id="ok"))

    assert events == ["run:ingest", "run:train", "run:deploy"]
    assert all(r.status is StageStatus.SUCCESS for r in results.values())
    assert orch.progress == 1.0


def test_cyclic_graph_is_rejected() -> None:
    events: list[str] = []
    stages = [
        RecordingStage("a", events, dependencies={"b"}),
        RecordingStage("b", events, dependencies={"a"}),
    ]
    with pytest.raises(ValueError, match="Cyclic"):
        _orchestrator(stages)


# ----------------------------------------------------------------------
# Failure path
# ----------------------------------------------------------------------
async def test_failure_cancels_in_flight_siblings_before_rollback() -> None:
    """A sibling still running when a stage fails must not commit its effect.

    Regression: rollback used to be invoked from inside the failing stage's
    own task while siblings kept running, and the remaining tasks were then
    awaited rather than cancelled -- so a slow sibling would finish, create
    side effects, and never be compensated.
    """
    events: list[str] = []
    stages = [
        RecordingStage("setup", events),
        RecordingStage("fails", events, dependencies={"setup"}, delay=0.01, fail=True),
        RecordingStage("slow", events, dependencies={"setup"}, delay=5.0),
    ]

    started = time.perf_counter()
    async with _orchestrator(stages, concurrency=4) as orch:
        with pytest.raises(RuntimeError, match="Stage 'fails' failed"):
            await orch.run(ExecutionContext(run_id="abort"))
    elapsed = time.perf_counter() - started

    # The slow sibling was cancelled, so its side effect never happened...
    assert "run:slow" not in events
    # ...and we did not sit and wait for it either.
    assert elapsed < 2.0
    # The stage that did commit an effect was compensated.
    assert "rollback:setup" in events


async def test_rollback_runs_in_reverse_completion_order() -> None:
    """Compensating actions must unwind, not replay, the completion order."""
    events: list[str] = []
    stages = [
        RecordingStage("first", events),
        RecordingStage("second", events, dependencies={"first"}),
        RecordingStage("third", events, dependencies={"second"}, fail=True),
    ]
    async with _orchestrator(stages) as orch:
        with pytest.raises(RuntimeError):
            await orch.run(ExecutionContext(run_id="unwind"))

    rollbacks = [e for e in events if e.startswith("rollback:")]
    assert rollbacks == ["rollback:second", "rollback:first"]


class AwaitingRollbackStage(RecordingStage):
    """Stage whose compensating action does I/O, as a real one would."""

    async def rollback(self, context: ExecutionContext, result) -> None:
        await asyncio.sleep(0.05)
        self.events.append(f"rollback:{self.name}")


async def test_failure_with_many_parallel_siblings_does_not_corrupt_results() -> None:
    """Regression: rollback iterated the results dict while tasks wrote to it.

    Rollback used to run from inside the failing stage's task. The moment it
    awaited -- which any rollback doing real I/O does -- a sibling could
    finish and insert into ``results`` mid-iteration, so the caller saw
    ``RuntimeError: dictionary changed size during iteration`` instead of the
    pipeline's own failure, and the remaining compensations never ran.
    """
    events: list[str] = []
    stages = [AwaitingRollbackStage("root", events)]
    stages += [
        AwaitingRollbackStage(f"leaf{i}", events, dependencies={"root"}, delay=0.03)
        for i in range(8)
    ]
    stages.append(
        AwaitingRollbackStage(
            "boom", events, dependencies={"root"}, delay=0.001, fail=True
        )
    )

    async with _orchestrator(stages, concurrency=16) as orch:
        with pytest.raises(RuntimeError) as excinfo:
            await orch.run(ExecutionContext(run_id="storm"))

    assert "Stage 'boom' failed" in str(excinfo.value)
    assert "changed size during iteration" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


async def test_cancelled_stages_are_reported_as_skipped() -> None:
    """Callers must be able to tell what ran from what was abandoned."""
    events: list[str] = []
    captured: dict = {}

    class CapturingOrchestrator(PipelineOrchestrator):
        async def _rollback(self, context, results, completed_order=None):
            captured.update(results)
            await super()._rollback(context, results, completed_order)

    stages = [
        RecordingStage("setup", events),
        RecordingStage("fails", events, dependencies={"setup"}, delay=0.01, fail=True),
        RecordingStage("slow", events, dependencies={"setup"}, delay=5.0),
    ]
    orch = CapturingOrchestrator(
        stages, concurrency=4, metrics_registry=CollectorRegistry()
    )
    with pytest.raises(RuntimeError):
        await orch.run(ExecutionContext(run_id="statuses"))

    assert captured["setup"].status is StageStatus.SUCCESS
    assert captured["fails"].status is StageStatus.FAILED
    assert captured["slow"].status is StageStatus.SKIPPED


async def test_retries_exhaust_before_the_pipeline_aborts() -> None:
    attempts = {"n": 0}
    events: list[str] = []

    class FlakyStage(RecordingStage):
        async def run(self, context: ExecutionContext) -> str:
            attempts["n"] += 1
            raise ValueError("flaky")

    stages = [
        FlakyStage(
            "flaky", events, retry_policy=RetryPolicy(max_retries=2, base_delay=0.0)
        )
    ]
    async with _orchestrator(stages) as orch:
        with pytest.raises(RuntimeError):
            await orch.run(ExecutionContext(run_id="retry"))

    assert attempts["n"] == 3  # initial attempt plus two retries
