import pytest

from enterprise_ml_platform.services.streaming.continuous_learning import (
    DriftAdapter,
    IncrementalTrainer,
    ModelWarmer,
    OnlineLearner,
)


@pytest.mark.asyncio
async def test_online_learner_updates_and_predicts() -> None:
    learner = OnlineLearner()
    trainer = IncrementalTrainer(learner, classes=[0, 1])
    await trainer.update([0.0, 0.0], 0)
    await trainer.update([1.0, 1.0], 1)
    pred = await learner.predict([1.0, 1.0])
    assert pred in (0, 1)


@pytest.mark.asyncio
async def test_drift_adapter_resets_model_on_errors() -> None:
    learner = OnlineLearner()
    trainer = IncrementalTrainer(learner, classes=[0, 1])
    await trainer.update([0.0, 0.0], 0)
    adapter = DriftAdapter(learner, threshold=0.1, window=5)
    for _ in range(5):
        await adapter.report(1, 0)
    assert learner._classes is None  # learner reset


@pytest.mark.asyncio
async def test_model_warmer_pretrains() -> None:
    learner = OnlineLearner()
    warmer = ModelWarmer(learner, [([0.0, 0.0], 0), ([1.0, 1.0], 1)], classes=[0, 1])
    await warmer.warm()
    pred = await learner.predict([1.0, 1.0])
    assert pred == 1
