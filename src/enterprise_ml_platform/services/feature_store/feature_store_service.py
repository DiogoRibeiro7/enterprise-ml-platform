"""High level feature store service orchestrating online/offline stores."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import structlog

from ..monitoring.collectors.metrics_collector import MetricsCollector
from .feature_registry import FeatureRegistry
from .offline_store import OfflineStore
from .online_store import OnlineFeatureStore
from .validators import FeatureSchemaValidator


@dataclass
class FeatureStoreConfig:
    """Configuration for the feature store service."""

    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 3600
    feast_repo: str | None = None


class FeatureStoreService:
    """Coordinate feature registration and retrieval."""

    def __init__(
        self,
        config: FeatureStoreConfig,
        registry: FeatureRegistry,
        online: OnlineFeatureStore,
        offline: OfflineStore,
        validator: FeatureSchemaValidator | None = None,
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
        version: str | None = None,
        lineage: dict[str, str] | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Register ``df`` as feature set ``name`` and return the version."""

        logger = self.logger.bind(correlation_id=correlation_id)
        version = version or dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S%f")
        schema = {str(c): str(t) for c, t in df.dtypes.items()}
        if self.validator:
            self.validator.validate(df)
        self.registry.register(name, version, schema, lineage)
        await self.offline.write_features(name, version, df)

        # Populate online cache for this version's snapshot
        if "entity_id" in df.columns:
            for _, row in df.iterrows():
                entity = str(row["entity_id"])
                feats = row.drop(labels=["entity_id", "timestamp"], errors="ignore")
                await self.online.set_features(
                    name,
                    version,
                    entity,
                    {str(k): v for k, v in feats.to_dict().items()},
                )

        if self.feast_repo:
            try:  # pragma: no cover - optional dependency
                from feast import FeatureStore as FeastStore

                store = FeastStore(repo_path=self.feast_repo)
                store.write_to_online_store(df)
            except Exception as exc:  # pragma: no cover
                logger.warning("Feast integration failed", error=str(exc))
        return version

    # ------------------------------------------------------------------
    def resolve_version(self, name: str) -> str | None:
        """Return the latest registered version of ``name``, or ``None``."""
        try:
            return self.registry.get(name).version
        except KeyError:
            return None

    # ------------------------------------------------------------------
    async def get_online_features(
        self,
        name: str,
        entity_id: str,
        features: Iterable[str],
        *,
        version: str | None = None,
        as_of: pd.Timestamp | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, float]:
        """Retrieve features for ``entity_id`` within feature set ``name``.

        Lookups are always scoped to a single feature set version, so two
        feature sets sharing an entity id cannot serve each other's values.
        ``version`` defaults to the latest registered one.

        If ``as_of`` is provided the lookup is served from the offline store to
        enable time travel queries. Otherwise the online cache is used with a
        fallback to the offline store on cache miss.
        """

        logger = self.logger.bind(correlation_id=correlation_id)
        requested = list(features)
        resolved = version or self.resolve_version(name)
        if resolved is None:
            logger.warning("feature_set_not_registered", name=name)
            return {}

        if as_of is not None:
            stored = await self.offline.get_features(name, resolved, entity_id, as_of)
            return self._select(stored, requested)

        result = await self.online.get_features(name, resolved, entity_id, requested)
        if result:
            return result
        # Cache miss -> fall back to the offline store for the same version
        stored = await self.offline.get_features(name, resolved, entity_id)
        if stored:
            await self.online.set_features(name, resolved, entity_id, stored)
        else:
            logger.warning(
                "feature_not_found", entity_id=entity_id, name=name, version=resolved
            )
        return self._select(stored, requested)

    # ------------------------------------------------------------------
    @staticmethod
    def _select(stored: dict[str, float], requested: Iterable[str]) -> dict[str, float]:
        """Return the requested subset, or ``{}`` if any feature is missing.

        The online and offline paths must answer the same query identically,
        otherwise a model silently receives a different feature vector
        depending on whether the cache happened to be warm.
        """
        requested = list(requested)
        if not requested:
            return {}
        if any(name not in stored for name in requested):
            return {}
        return {name: stored[name] for name in requested}

    # ------------------------------------------------------------------
    async def close(self) -> None:
        await self.online.close()
