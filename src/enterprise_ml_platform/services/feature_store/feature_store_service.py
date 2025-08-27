from __future__ import annotations
"""High level feature store service orchestrating online/offline stores."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import datetime as dt

import pandas as pd
import structlog

from .online_store import OnlineFeatureStore
from .offline_store import OfflineFeatureStore
from .feature_registry import FeatureRegistry
from .validators import FeatureSchemaValidator
from ..monitoring.collectors.metrics_collector import MetricsCollector


@dataclass
class FeatureStoreConfig:
    """Configuration for the feature store service."""

    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 3600
    feast_repo: Optional[str] = None


class FeatureStoreService:
    """Coordinate feature registration and retrieval."""

    def __init__(
        self,
        config: FeatureStoreConfig,
        registry: FeatureRegistry,
        online: OnlineFeatureStore,
        offline: OfflineFeatureStore,
        validator: Optional[FeatureSchemaValidator] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.online = online
        self.offline = offline
        self.validator = validator
        self.logger = structlog.get_logger(__name__)
        self.metrics = online.metrics or MetricsCollector()
        self.feast_repo = config.feast_repo

    # ------------------------------------------------------------------
    async def register_features(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        version: Optional[str] = None,
        lineage: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Register ``df`` as feature set ``name`` and return the version."""

        logger = self.logger.bind(correlation_id=correlation_id)
        version = version or dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        schema = {c: str(t) for c, t in df.dtypes.items()}
        if self.validator:
            self.validator.validate(df)
        self.registry.register(name, version, schema, lineage)
        await self.offline.write_features(name, version, df)

        # Populate online cache for latest snapshot
        if "entity_id" in df.columns:
            for _, row in df.iterrows():
                entity = str(row["entity_id"])
                feats = row.drop(labels=["entity_id", "timestamp"], errors="ignore")
                await self.online.set_features(entity, feats.to_dict())

        if self.feast_repo:
            try:  # pragma: no cover - optional dependency
                from feast import FeatureStore as FeastStore

                store = FeastStore(repo_path=self.feast_repo)
                store.write_to_online_store(df)
            except Exception as exc:  # pragma: no cover
                logger.warning("Feast integration failed", error=str(exc))
        return version

    # ------------------------------------------------------------------
    async def get_online_features(
        self,
        name: str,
        entity_id: str,
        features: Iterable[str],
        *,
        as_of: Optional[pd.Timestamp] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Retrieve features for ``entity_id``.

        If ``as_of`` is provided the lookup is served from the offline store to
        enable time travel queries.  Otherwise the online cache is used with a
        fallback to the offline store on cache miss.
        """

        logger = self.logger.bind(correlation_id=correlation_id)
        if as_of is not None:
            desc = self.registry.get(name)
            return await self.offline.get_features(name, desc.version, entity_id, as_of)

        result = await self.online.get_features(entity_id, features)
        if result:
            return result
        # Cache miss -> fallback to offline store using latest version
        desc = self.registry.get(name)
        offline = await self.offline.get_features(name, desc.version, entity_id)
        if offline:
            await self.online.set_features(entity_id, offline)
        else:
            logger.warning("feature_not_found", entity_id=entity_id, name=name)
        return offline

    # ------------------------------------------------------------------
    async def close(self) -> None:
        await self.online.close()
