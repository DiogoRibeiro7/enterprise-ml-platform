import asyncio
import pathlib
import sys

import fakeredis.aioredis as fakeredis
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

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

API_KEY = {"X-API-Key": "secret"}


def _build_feature_store() -> FeatureStoreService:
    metrics = MetricsCollector()
    redis_client = fakeredis.FakeRedis()
    online = OnlineFeatureStore(redis_client, metrics=metrics)
    offline = OfflineFeatureStore(metrics=metrics)
    registry = FeatureRegistry()
    cfg = FeatureStoreConfig(redis_url="redis://fake")
    return FeatureStoreService(cfg, registry, online, offline)


def _cleanup(store: FeatureStoreService) -> None:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(store.online.redis.close())
    loop.run_until_complete(store.close())


def test_feature_store_stats_endpoint():
    store = _build_feature_store()
    app = FastAPI()

    @app.get("/api/v1/feature-store/stats")
    async def stats():
        metrics = store.metrics
        hits = metrics.feature_cache_hits.labels("online")._value.get()
        misses = metrics.feature_cache_misses.labels("online")._value.get()
        return {"cache_hits": hits, "cache_misses": misses}

    client = TestClient(app)
    resp = client.get("/api/v1/feature-store/stats", headers=API_KEY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["cache_hits"] == 0
    assert data["cache_misses"] == 0
    _cleanup(store)
