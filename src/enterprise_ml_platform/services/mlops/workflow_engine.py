"""Minimal MLOps workflow orchestration engine.

This module glues together the various helper components that make up the
MLOps framework.  The goal is not to be feature complete but to provide a
light‑weight abstraction that mimics the behaviour of a real system: building
pipelines, executing tests and validations, deploying models and handling
rollbacks.  The engine is intentionally synchronous and in‑memory so it can be
used in unit tests without external services.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .approval.approval_workflow import ApprovalWorkflow
from .ci_cd.pipeline_builder import PipelineBuilder
from .deployment.deployment_automator import DeploymentAutomator
from .experimentation.experiment_tracker import ExperimentTracker
from .monitoring.pipeline_monitor import PipelineMonitor
from .rollback.rollback_automator import RollbackAutomator
from .testing.automated_testing import AutomatedTesting
from .validation.model_validator import ModelValidator

Step = Callable[[Any], Any]


@dataclass
class WorkflowEngine:
    """High level orchestrator for end‑to‑end ML workflows."""

    pipeline_builder: PipelineBuilder = field(default_factory=PipelineBuilder)
    tester: AutomatedTesting = field(default_factory=AutomatedTesting)
    validator: ModelValidator = field(default_factory=ModelValidator)
    deployer: DeploymentAutomator = field(default_factory=DeploymentAutomator)
    monitor: PipelineMonitor = field(default_factory=PipelineMonitor)
    rollback: RollbackAutomator = field(default_factory=RollbackAutomator)
    tracker: ExperimentTracker = field(default_factory=ExperimentTracker)
    approval: ApprovalWorkflow = field(default_factory=ApprovalWorkflow)

    def run(
        self,
        initial_model: Any,
        *,
        steps: Iterable[Step] | None = None,
        test_data: tuple[Any, Any] | None = None,
        validation_thresholds: dict[str, float] | None = None,
        environment: str = "dev",
    ) -> tuple[str, dict[str, float]]:
        """Execute an end‑to‑end workflow.

        Args:
            initial_model: starting model that will flow through the pipeline.
            steps: optional additional transformation functions.
            test_data: tuple of ``(X, y)`` used for automated tests.
            validation_thresholds: metric thresholds for ``ModelValidator``.
            environment: deployment environment name.

        Returns:
            Tuple of deployment identifier and gathered test metrics.
        """

        pipeline = self.pipeline_builder.build(list(steps or []))
        model = initial_model
        for step in pipeline:
            model = step(model)

        X, y = test_data if test_data else (None, None)
        metrics = self.tester.run(model, X, y)
        if not self.validator.validate(metrics, validation_thresholds or {}):
            # simulate rollback procedure when validation fails
            last = self.rollback.rollback()
            raise ValueError(f"validation failed; rolled back to {last}")

        exp_id = self.tracker.log_experiment({"steps": len(pipeline)}, metrics)
        if not self.approval.request_approval(exp_id):
            raise PermissionError("model not approved")

        deployment_id = self.deployer.deploy(model, environment)
        self.monitor.record({"deployment_id": deployment_id, "metrics": metrics})
        self.rollback.checkpoint(deployment_id)
        return deployment_id, metrics
