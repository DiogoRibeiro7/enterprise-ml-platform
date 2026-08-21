"""Regression tests for feature key namespacing.

The online store used to key entries on the entity id alone. Entity ids are
only unique within a feature set, so ``customer_features`` and
``risk_features`` both wrote entity ``42`` to the Redis hash ``42`` -- each
one silently serving the other's values, and every new version overwriting
the previous one.
"""

from __future__ import annotations

import fakeredis.aioredis as fakeredis
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from enterprise_ml_platform.services.feature_store import (
    FeatureRegistry,
    FeatureStoreConfig,
    FeatureStoreService,
    OfflineFeatureStore,
    OnlineFeatureStore,
)
from enterprise_ml_platform.services.feature_store.online_store import build_key
from enterprise_ml_platform.services.monitoring.collectors.metrics_collector import (
    MetricsCollector,
)


@pytest.fixture
async def store():
    """A feature store service backed by an isolated fake Redis."""
    metrics = MetricsCollector(CollectorRegistry())
    redis_client = fakeredis.FakeRedis()
    service = FeatureStoreService(
        FeatureStoreConfig(redis_url="redis://fake"),
        FeatureRegistry(),
        OnlineFeatureStore(redis_client, metrics=metrics, ttl_seconds=60),
        OfflineFeatureStore(metrics=metrics),
    )
    yield service
    await service.close()


def _frame(entity_id: str, **features) -> pd.DataFrame:
    return pd.DataFrame([{"entity_id": entity_id, **features}])


# ----------------------------------------------------------------------
async def test_feature_sets_sharing_an_entity_id_do_not_collide(store) -> None:
    await store.register_features("customer_features", _frame("42", tenure=7.0))
    await store.register_features("risk_features", _frame("42", tenure=99.0))

    customer = await store.get_online_features("customer_features", "42", ["tenure"])
    risk = await store.get_online_features("risk_features", "42", ["tenure"])

    assert customer == {"tenure": 7.0}
    assert risk == {"tenure": 99.0}


async def test_registering_a_second_version_does_not_overwrite_the_first(store) -> None:
    v1 = await store.register_features("customer_features", _frame("42", score=0.1))
    v2 = await store.register_features("customer_features", _frame("42", score=0.9))
    assert v1 != v2, "each registration must produce a distinct version"

    latest = await store.get_online_features("customer_features", "42", ["score"])
    pinned = await store.get_online_features(
        "customer_features", "42", ["score"], version=v1
    )

    assert latest == {"score": 0.9}, "the latest version should be served by default"
    assert pinned == {"score": 0.1}, "the earlier version must still be retrievable"


async def test_redis_keys_are_namespaced_by_feature_set_and_version(store) -> None:
    version = await store.register_features(
        "customer_features", _frame("42", score=0.5)
    )

    keys = sorted(k.decode() for k in await store.online.redis.keys("*"))

    assert keys == [build_key("customer_features", version, "42")]
    assert keys[0] == f"features:customer_features:{version}:42"
    assert "42" not in keys, "the bare entity id must never be used as a key"


async def test_features_of_one_entity_do_not_leak_to_another(store) -> None:
    df = pd.DataFrame(
        [
            {"entity_id": "42", "score": 0.1},
            {"entity_id": "43", "score": 0.2},
        ]
    )
    await store.register_features("customer_features", df)

    assert await store.get_online_features("customer_features", "42", ["score"]) == {
        "score": 0.1
    }
    assert await store.get_online_features("customer_features", "43", ["score"]) == {
        "score": 0.2
    }


# ----------------------------------------------------------------------
# Miss handling
# ----------------------------------------------------------------------
async def test_cache_miss_falls_back_to_offline_and_repopulates_the_right_key(
    store,
) -> None:
    version = await store.register_features(
        "customer_features", _frame("42", score=0.5)
    )
    await store.online.redis.flushall()  # evict the online cache

    served = await store.get_online_features("customer_features", "42", ["score"])

    assert served == {"score": 0.5}
    keys = [k.decode() for k in await store.online.redis.keys("*")]
    assert keys == [build_key("customer_features", version, "42")]


async def test_unregistered_feature_set_returns_empty(store) -> None:
    assert await store.get_online_features("never_registered", "42", ["score"]) == {}


async def test_unknown_entity_returns_empty(store) -> None:
    await store.register_features("customer_features", _frame("42", score=0.5))
    assert await store.get_online_features("customer_features", "999", ["score"]) == {}


async def test_partial_feature_miss_returns_empty(store) -> None:
    """A partially served vector would silently change the model's input."""
    await store.register_features("customer_features", _frame("42", score=0.5))

    served = await store.get_online_features(
        "customer_features", "42", ["score", "not_a_feature"]
    )

    assert served == {}


async def test_point_in_time_lookup_is_scoped_to_the_feature_set(store) -> None:
    """Time travel must answer from the right feature set, and only it."""
    history = pd.DataFrame(
        [
            {"entity_id": "42", "score": 0.1, "timestamp": pd.Timestamp("2024-01-01")},
            {"entity_id": "42", "score": 0.9, "timestamp": pd.Timestamp("2024-06-01")},
        ]
    )
    await store.register_features("customer_features", history)
    await store.register_features("risk_features", _frame("42", score=99.0))

    early = await store.get_online_features(
        "customer_features", "42", ["score"], as_of=pd.Timestamp("2024-03-01")
    )
    late = await store.get_online_features(
        "customer_features", "42", ["score"], as_of=pd.Timestamp("2024-09-01")
    )

    assert early == {"score": 0.1}, "must not see a value recorded after as_of"
    assert late == {"score": 0.9}


async def test_point_in_time_lookup_returns_only_requested_features(store) -> None:
    """The same query must answer identically on every retrieval path."""
    df = pd.DataFrame(
        [
            {
                "entity_id": "42",
                "score": 0.1,
                "tenure": 7.0,
                "timestamp": pd.Timestamp("2024-01-01"),
            }
        ]
    )
    await store.register_features("customer_features", df)

    served = await store.get_online_features(
        "customer_features", "42", ["score"], as_of=pd.Timestamp("2024-06-01")
    )

    assert served == {"score": 0.1}


async def test_ttl_is_applied_to_namespaced_keys(store) -> None:
    version = await store.register_features(
        "customer_features", _frame("42", score=0.5)
    )

    ttl = await store.online.redis.ttl(build_key("customer_features", version, "42"))

    assert 0 < ttl <= 60
