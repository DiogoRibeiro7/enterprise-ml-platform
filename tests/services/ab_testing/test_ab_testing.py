import numpy as np
import pytest

from enterprise_ml_platform.services.ab_testing import (
    ExperimentConfig,
    ExperimentManager,
    StatisticalAnalyzer,
)


@pytest.mark.asyncio
async def test_traffic_and_analysis() -> None:
    cfg = ExperimentConfig(
        name="exp1",
        variants={"a": "modelA", "b": "modelB"},
        traffic_split={"a": 0.7, "b": 0.3},
    )
    manager = ExperimentManager()
    await manager.create_experiment(cfg)

    counts = {"a": 0, "b": 0}
    for i in range(500):
        sid = f"s{i}"
        variant = await manager.get_variant("exp1", sid)
        counts[variant] += 1
        assert variant == await manager.get_variant("exp1", sid)
        value = 1.0 if variant == "b" else 0.5
        await manager.record_outcome("exp1", variant, value, variant == "b")

    assert 0.55 < counts["a"] / 500 < 0.85

    analysis = await manager.analyze("exp1")
    assert analysis["t_test"]["effect"] > 0
    assert analysis["bayesian_prob"] > 0


def test_t_test_significance() -> None:
    analyzer = StatisticalAnalyzer()
    a = np.random.normal(0, 1, 100)
    b = np.random.normal(1, 1, 100)
    res = analyzer.t_test(a, b)
    assert res["p_value"] < 0.05
