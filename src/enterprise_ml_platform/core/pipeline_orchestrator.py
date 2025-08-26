# Enterprise ML Pipeline - Core Architecture
# File: core/pipeline_orchestrator.py

import asyncio
import logging
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from contextlib import asynccontextmanager
import structlog
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class PipelineStage(str, Enum):
    """Pipeline execution stages"""

    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MONITORING = "monitoring"


class ExecutionStatus(str, Enum):
    """Execution status states"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """Execution context for pipeline runs"""

    run_id: str
    experiment_id: str
    environment: str
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StageResult:
    """Result from pipeline stage execution"""

    stage: PipelineStage
    status: ExecutionStatus
    output: Optional[Any] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[Exception] = None
    duration_seconds: float = 0.0


class PipelineStageProtocol(Protocol):
    """Protocol for pipeline stages"""

    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute the pipeline stage"""
        ...

    async def validate(self, context: ExecutionContext) -> bool:
        """Validate stage prerequisites"""
        ...

    async def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup stage resources"""
        ...


class BasePipelineStage(ABC):
    """Base class for pipeline stages"""

    def __init__(self, name: str, stage_type: PipelineStage):
        self.name = name
        self.stage_type = stage_type
        self.logger = structlog.get_logger().bind(stage=name)

    @abstractmethod
    async def _execute_stage(self, context: ExecutionContext) -> StageResult:
        """Implement stage-specific logic"""
        pass

    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute stage with error handling and logging"""
        import time

        start_time = time.time()

        self.logger.info("Starting stage execution", run_id=context.run_id)

        try:
            # Validate prerequisites
            if not await self.validate(context):
                raise ValueError(f"Stage validation failed: {self.name}")

            # Execute stage
            result = await self._execute_stage(context)
            result.duration_seconds = time.time() - start_time

            self.logger.info(
                "Stage completed successfully",
                run_id=context.run_id,
                duration=result.duration_seconds,
                metrics=result.metrics,
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Stage execution failed",
                run_id=context.run_id,
                error=str(e),
                duration=duration,
            )

            return StageResult(
                stage=self.stage_type,
                status=ExecutionStatus.FAILED,
                error=e,
                duration_seconds=duration,
            )

    async def validate(self, context: ExecutionContext) -> bool:
        """Default validation - can be overridden"""
        return True

    async def cleanup(self, context: ExecutionContext) -> None:
        """Default cleanup - can be overridden"""
        pass


class PipelineOrchestrator:
    """Advanced pipeline orchestrator with dependency management"""

    def __init__(
        self,
        stages: List[BasePipelineStage],
        max_parallel: int = 3,
        retry_policy: Optional[Dict] = None,
    ):
        self.stages = {stage.stage_type: stage for stage in stages}
        self.max_parallel = max_parallel
        self.retry_policy = retry_policy or {"max_retries": 3, "backoff_factor": 2}
        self.logger = structlog.get_logger().bind(component="orchestrator")

        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph()

    def _build_dependency_graph(self) -> Dict[PipelineStage, List[PipelineStage]]:
        """Build stage dependency graph"""
        # Default linear dependency for ML pipeline
        stages_order = [
            PipelineStage.DATA_INGESTION,
            PipelineStage.DATA_VALIDATION,
            PipelineStage.FEATURE_ENGINEERING,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.MODEL_VALIDATION,
            PipelineStage.MODEL_DEPLOYMENT,
            PipelineStage.MONITORING,
        ]

        graph = {}
        for i, stage in enumerate(stages_order):
            graph[stage] = stages_order[:i]  # All previous stages as dependencies

        return graph

    async def execute_pipeline(
        self, context: ExecutionContext
    ) -> Dict[PipelineStage, StageResult]:
        """Execute pipeline with dependency management and parallelization"""

        self.logger.info("Starting pipeline execution", run_id=context.run_id)

        results: Dict[PipelineStage, StageResult] = {}
        completed = set()
        failed = set()

        # Create semaphore for parallel execution control
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_stage_with_retry(stage: PipelineStage) -> StageResult:
            """Execute single stage with retry logic"""

            async with semaphore:
                stage_instance = self.stages[stage]

                for attempt in range(self.retry_policy["max_retries"] + 1):
                    try:
                        result = await stage_instance.execute(context)

                        if result.status == ExecutionStatus.SUCCESS:
                            return result
                        elif attempt < self.retry_policy["max_retries"]:
                            wait_time = self.retry_policy["backoff_factor"] ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return result

                    except Exception as e:
                        if attempt < self.retry_policy["max_retries"]:
                            wait_time = self.retry_policy["backoff_factor"] ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return StageResult(
                                stage=stage, status=ExecutionStatus.FAILED, error=e
                            )

        # Execute stages respecting dependencies
        while len(completed) + len(failed) < len(self.stages):
            # Find stages ready to execute
            ready_stages = []

            for stage in self.stages:
                if (
                    stage not in completed
                    and stage not in failed
                    and stage
                    not in [
                        task.get_name()
                        for task in asyncio.all_tasks()
                        if hasattr(task, "get_name")
                    ]
                ):
                    # Check if all dependencies are completed
                    dependencies = self.dependency_graph.get(stage, [])
                    if all(dep in completed for dep in dependencies):
                        ready_stages.append(stage)

            if not ready_stages:
                # Check if we're stuck due to failures
                if failed:
                    self.logger.error(
                        "Pipeline execution halted due to stage failures",
                        failed_stages=list(failed),
                    )
                    break
                else:
                    await asyncio.sleep(1)  # Wait for running stages
                    continue

            # Execute ready stages
            tasks = []
            for stage in ready_stages:
                task = asyncio.create_task(execute_stage_with_retry(stage))
                task.set_name(stage.value)
                tasks.append(task)

            # Wait for at least one task to complete
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for task in done:
                stage_name = task.get_name()
                stage = PipelineStage(stage_name)
                result = task.result()

                results[stage] = result

                if result.status == ExecutionStatus.SUCCESS:
                    completed.add(stage)
                    # Update context with stage artifacts and metrics
                    context.artifacts.update(result.artifacts)
                    context.metrics.update(result.metrics)
                else:
                    failed.add(stage)

            # Cancel remaining tasks if critical failure
            for task in pending:
                if not task.done():
                    task.cancel()

        # Cleanup all stages
        await self._cleanup_stages(context, results)

        self.logger.info(
            "Pipeline execution completed",
            run_id=context.run_id,
            completed_stages=len(completed),
            failed_stages=len(failed),
        )

        return results

    async def _cleanup_stages(
        self, context: ExecutionContext, results: Dict[PipelineStage, StageResult]
    ):
        """Cleanup all pipeline stages"""
        cleanup_tasks = []

        for stage_type, stage_instance in self.stages.items():
            if stage_type in results:  # Only cleanup stages that were executed
                task = asyncio.create_task(stage_instance.cleanup(context))
                cleanup_tasks.append(task)

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)


class PipelineMetrics:
    """Pipeline metrics collection and reporting"""

    def __init__(self):
        self.metrics_store = {}

    def record_stage_metrics(self, stage: PipelineStage, metrics: Dict[str, float]):
        """Record metrics for a pipeline stage"""
        if stage not in self.metrics_store:
            self.metrics_store[stage] = []

        self.metrics_store[stage].append(
            {**metrics, "timestamp": asyncio.get_event_loop().time()}
        )

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics summary"""
        summary = {}

        for stage, metrics_list in self.metrics_store.items():
            if metrics_list:
                latest_metrics = metrics_list[-1]
                summary[stage.value] = {
                    "latest_metrics": latest_metrics,
                    "total_executions": len(metrics_list),
                    "average_duration": sum(
                        m.get("duration_seconds", 0) for m in metrics_list
                    )
                    / len(metrics_list),
                }

        return summary


# Dependency Injection Container
class ApplicationContainer(containers.DeclarativeContainer):
    """Dependency injection container for the ML pipeline"""

    # Configuration
    config = providers.Configuration()

    # Logging
    logger = providers.Singleton(structlog.get_logger)

    # Metrics
    metrics = providers.Singleton(PipelineMetrics)

    # Pipeline stages (these would be defined in separate modules)
    # data_ingestion_stage = providers.Factory(DataIngestionStage)
    # data_validation_stage = providers.Factory(DataValidationStage)
    # ... etc

    # Pipeline orchestrator
    pipeline_orchestrator = providers.Factory(
        PipelineOrchestrator,
        stages=providers.List(),  # Injected from specific implementations
        max_parallel=config.pipeline.max_parallel,
        retry_policy=config.pipeline.retry_policy,
    )


# Context managers for resource management
@asynccontextmanager
async def pipeline_execution_context(
    run_id: str, experiment_id: str, config: Dict[str, Any]
):
    """Context manager for pipeline execution"""
    context = ExecutionContext(
        run_id=run_id,
        experiment_id=experiment_id,
        environment=config.get("environment", "development"),
        config=config,
    )

    logger.info("Pipeline context created", run_id=run_id, experiment_id=experiment_id)

    try:
        yield context
    finally:
        logger.info("Pipeline context cleanup", run_id=run_id)


# Advanced error handling and circuit breaker pattern
class CircuitBreaker:
    """Circuit breaker for pipeline stage resilience"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        import time

        current_time = time.time()

        if self.state == "OPEN":
            if current_time - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e


# Usage example with proper separation
async def main():
    """Example usage of the enterprise ML pipeline"""

    # This would typically be loaded from configuration files
    config = {
        "environment": "production",
        "pipeline": {
            "max_parallel": 3,
            "retry_policy": {"max_retries": 3, "backoff_factor": 2},
        },
        "model": {"name": "fraud_detection_v2", "version": "1.0.0"},
    }

    # Initialize dependency container
    container = ApplicationContainer()
    container.config.from_dict(config)

    # This would be populated with actual stage implementations
    stages = []  # DataIngestionStage(), DataValidationStage(), etc.

    orchestrator = PipelineOrchestrator(stages)

    # Execute pipeline
    async with pipeline_execution_context("run-001", "exp-001", config) as context:
        results = await orchestrator.execute_pipeline(context)

        # Process results
        for stage, result in results.items():
            if result.status == ExecutionStatus.SUCCESS:
                logger.info(f"Stage {stage} completed successfully")
            else:
                logger.error(f"Stage {stage} failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
