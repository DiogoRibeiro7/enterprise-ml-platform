import asyncio

import pytest

from enterprise_ml_platform.services.streaming.feature_engineering import (
    FeatureCache,
    StreamFeatureEngine,
    StreamJoiner,
    TimeWindowAggregator,
)


@pytest.mark.asyncio
async def test_stream_feature_engine_enriches_and_windows() -> None:
    joiner = StreamJoiner("user_id", {"u1": {"age": 30}})
    window = TimeWindowAggregator("value", window_size=0.5)
    engine = StreamFeatureEngine(window_ops=[window], joiners=[joiner], cache=FeatureCache())

    features1 = await engine.compute({"entity_id": "u1", "user_id": "u1", "value": 1.0})
    assert features1["age"] == 30
    await asyncio.sleep(0.1)
    features2 = await engine.compute({"entity_id": "u1", "user_id": "u1", "value": 2.0})
    assert features2["value_time_avg"] >= 1.0
    # cache should return enriched features
    cached = await engine.cache.get("u1")
    assert cached and cached["age"] == 30
