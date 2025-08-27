import pytest

from enterprise_ml_platform.services.resource_management.cost_optimizer import CostOptimizer
from enterprise_ml_platform.services.resource_management.monitoring.cost_tracker import CostTracker
from enterprise_ml_platform.services.resource_management.monitoring.usage_analyzer import UsageAnalyzer
from enterprise_ml_platform.services.resource_management.monitoring.anomaly_detector import AnomalyDetector
from enterprise_ml_platform.services.resource_management.allocation.dynamic_scaler import DynamicScaler
from enterprise_ml_platform.services.resource_management.allocation.gpu_scheduler import GPUScheduler
from enterprise_ml_platform.services.resource_management.allocation.queue_manager import QueueManager


@pytest.mark.asyncio
async def test_cost_tracker_and_analyzer() -> None:
    tracker = CostTracker()
    analyzer = UsageAnalyzer(window=3)
    tracker.record_cost("proj", "user", "model", 5.0)
    tracker.record_cost("proj", "user", "model", 5.0)
    assert tracker.total_cost("proj") == 10.0

    analyzer.record_usage("proj", "cpu", 1)
    analyzer.record_usage("proj", "cpu", 3)
    assert analyzer.moving_average("proj", "cpu") == 2.0


def test_anomaly_detector() -> None:
    detector = AnomalyDetector()
    assert not detector.detect([1, 1, 1])
    assert detector.detect([1, 1, 10])


def test_dynamic_scaler() -> None:
    scaler = DynamicScaler(target=0.5, tolerance=0.1)
    assert scaler.decide(0.8) == "scale_up"
    assert scaler.decide(0.2) == "scale_down"
    assert scaler.decide(0.52) == "steady"


def test_gpu_scheduler_and_queue_manager() -> None:
    gpu = GPUScheduler(total_gpus=4)
    assert gpu.allocate("job1", 2)
    assert not gpu.allocate("job2", 3)
    gpu.release("job1")
    assert gpu.available() == 4

    queue = QueueManager()
    queue.enqueue("job1", {"cpu": 1})
    queue.enqueue("job2", {"cpu": 1})
    job_id, res = queue.dequeue()
    assert job_id == "job1"
    assert res == {"cpu": 1}
    assert len(queue) == 1


@pytest.mark.asyncio
async def test_cost_optimizer_integration() -> None:
    optimizer = CostOptimizer(gpu=GPUScheduler(total_gpus=2))
    await optimizer.record_cost("proj", "user", "model", 1)
    await optimizer.record_cost("proj", "user", "model", 1)
    await optimizer.record_cost("proj", "user", "model", 100)
    assert optimizer.detect_cost_anomaly("proj")

    await optimizer.submit_job("job1", {"gpus": 1})
    decision = optimizer.scale_decision(0.9)
    assert decision == "scale_up"
