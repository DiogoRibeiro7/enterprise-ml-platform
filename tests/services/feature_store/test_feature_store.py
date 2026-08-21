import fakeredis.aioredis as fakeredis
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from enterprise_ml_platform.services.feature_engineering import (
    FeatureEngineeringService,
)
from enterprise_ml_platform.services.feature_store import (
    FeatureRegistry,
    FeatureStoreConfig,
    FeatureStoreService,
    OfflineFeatureStore,
    OnlineFeatureStore,
)
from enterprise_ml_platform.services.monitoring.collectors.metrics_collector import (
    MetricsCollector,
)


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "num2": [2, 4, 6, 8, 10],
            "cat": ["a", "b", "a", "c", "b"],
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        }
    )
    target = pd.Series([0, 1, 0, 1, 0])
    return df, target


@pytest.mark.asyncio
async def test_online_feature_store_serving() -> None:
    metrics = MetricsCollector(CollectorRegistry())
    redis_client = fakeredis.FakeRedis()
    online = OnlineFeatureStore(redis_client, metrics=metrics, ttl_seconds=60)
    offline = OfflineFeatureStore(metrics=metrics)
    registry = FeatureRegistry()
    cfg = FeatureStoreConfig(redis_url="redis://fake")
    store = FeatureStoreService(cfg, registry, online, offline)
    df = pd.DataFrame(
        [{"entity_id": "1", "f1": 0.5, "f2": 1.0, "timestamp": pd.Timestamp.utcnow()}]
    )
    await store.register_features("test", df)
    res = await store.get_online_features("test", "1", ["f1", "f2"])
    assert res["f1"] == 0.5
    assert metrics.feature_cache_hits.labels("online")._value.get() == 1
    await store.close()
    await redis_client.close()


@pytest.mark.asyncio
async def test_feature_engineering_integration(sample_data) -> None:
    df, target = sample_data
    df["entity_id"] = [str(i) for i in range(len(df))]
    metrics = MetricsCollector(CollectorRegistry())
    redis_client = fakeredis.FakeRedis()
    online = OnlineFeatureStore(redis_client, metrics=metrics, ttl_seconds=60)
    offline = OfflineFeatureStore(metrics=metrics)
    registry = FeatureRegistry()
    cfg = FeatureStoreConfig(redis_url="redis://fake")
    store = FeatureStoreService(cfg, registry, online, offline)
    service = FeatureEngineeringService(
        {
            "transformers": {
                "numerical": {"polynomial_degree": 2, "bins": 2},
                "categorical": {"one_hot_threshold": 3},
                "temporal": {"reference_date": "2020-01-01"},
            },
            "feature_selection": {"enabled": False},
        }
    )
    service.feature_store = store
    engineered, _ = await service.engineer_features(df, target)
    res = await store.get_online_features(
        "engineered_features", "0", [c for c in engineered.columns if c != "entity_id"]
    )
    assert res
    await service.shutdown()
    await store.close()
    await redis_client.close()
