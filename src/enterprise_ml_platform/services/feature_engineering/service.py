"""Feature engineering service orchestrating transformers and selection."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import structlog

try:  # pragma: no cover - optional dependency
    from dask.distributed import Client as DaskClient
except ImportError:  # pragma: no cover
    DaskClient = None

from redis.asyncio import Redis

from ..feature_store import (
    FeatureRegistry,
    FeatureStoreConfig,
    FeatureStoreService,
    OfflineFeatureStore,
    OnlineFeatureStore,
)
from ..monitoring.collectors.metrics_collector import MetricsCollector
from .selectors.feature_selector import FeatureSelector
from .transformers.categorical_transformer import CategoricalFeatureTransformer
from .transformers.composite_transformer import CompositeFeatureTransformer
from .transformers.numerical_transformer import NumericalFeatureTransformer
from .transformers.temporal_transformer import TemporalFeatureTransformer

IDENTIFIER_COLUMNS = ("entity_id", "timestamp")

FeatureTransformerType = (
    NumericalFeatureTransformer
    | CategoricalFeatureTransformer
    | TemporalFeatureTransformer
    | CompositeFeatureTransformer
)


@dataclass(frozen=True)
class FeatureEngineeringMetrics:
    """Metrics returned by :class:`FeatureEngineeringService`."""

    features_created: int
    features_selected: int


class FeatureEngineeringService:
    """Apply configured feature transformers and feature selection."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.logger = structlog.get_logger(__name__)
        self.client: Any | None = None
        if self.config.get("use_dask") and DaskClient is not None:
            self.client = DaskClient(processes=False)

        self.feature_store: FeatureStoreService | None = None
        feature_store_config = self.config.get("feature_store")
        if feature_store_config is not None:
            if not isinstance(feature_store_config, Mapping):
                raise TypeError("feature_store configuration must be a mapping")
            metrics = MetricsCollector()
            redis_url = str(
                feature_store_config.get("redis_url", "redis://localhost:6379/0")
            )
            ttl_seconds = int(feature_store_config.get("ttl_seconds", 3600))
            feast_repo_value = feature_store_config.get("feast_repo")
            feast_repo = None if feast_repo_value is None else str(feast_repo_value)
            redis_client = Redis.from_url(redis_url)
            online = OnlineFeatureStore(
                redis_client, ttl_seconds=ttl_seconds, metrics=metrics
            )
            offline = OfflineFeatureStore(metrics=metrics)
            registry = FeatureRegistry()
            store_config = FeatureStoreConfig(
                redis_url=redis_url,
                ttl_seconds=ttl_seconds,
                feast_repo=feast_repo,
            )
            self.feature_store = FeatureStoreService(
                store_config, registry, online, offline
            )

    def _config_section(self, name: str) -> dict[str, Any]:
        section = self.config.get(name, {})
        if not isinstance(section, Mapping):
            raise TypeError(f"{name} configuration must be a mapping")
        return dict(section)

    async def engineer_features(
        self, data: pd.DataFrame, target: pd.Series | None = None
    ) -> tuple[pd.DataFrame, FeatureEngineeringMetrics]:
        """Build, select, and optionally persist features for ``data``."""
        frame = data.copy()
        identifiers = frame[
            [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
        ].copy()
        frame = frame.drop(columns=list(identifiers.columns))
        feature_input_width = frame.shape[1]

        transformers: list[FeatureTransformerType] = []
        transformer_config = self._config_section("transformers")
        if "numerical" in transformer_config:
            transformers.append(
                NumericalFeatureTransformer(dict(transformer_config["numerical"]))
            )
        if "categorical" in transformer_config:
            transformers.append(
                CategoricalFeatureTransformer(dict(transformer_config["categorical"]))
            )
        if "temporal" in transformer_config:
            transformers.append(
                TemporalFeatureTransformer(dict(transformer_config["temporal"]))
            )
        if "composite" in transformer_config:
            transformers.append(
                CompositeFeatureTransformer(dict(transformer_config["composite"]))
            )

        for transformer in transformers:
            if target is not None and isinstance(
                transformer, CategoricalFeatureTransformer
            ):
                frame = transformer.fit_transform(frame, target)
            else:
                frame = transformer.fit_transform(frame)
        features_created = frame.shape[1] - feature_input_width

        frame = frame.select_dtypes(exclude=["datetime", "datetimetz"]).copy()
        frame = frame.fillna(0)

        selector = FeatureSelector(self._config_section("feature_selection"))
        selected = selector.select(frame, target)
        metrics = FeatureEngineeringMetrics(
            features_created=features_created,
            features_selected=selected.shape[1],
        )

        if not identifiers.empty:
            selected = pd.concat(
                [
                    identifiers.reset_index(drop=True),
                    selected.reset_index(drop=True),
                ],
                axis=1,
            )

        if self.feature_store:
            await self.feature_store.register_features("engineered_features", selected)
        elif self.config.get("feast_repo"):
            try:
                from feast import FeatureStore

                store = FeatureStore(repo_path=str(self.config["feast_repo"]))
                store.write_to_online_store(selected)
            except Exception as exc:  # pragma: no cover - optional dependency
                self.logger.warning("Feast integration failed", error=str(exc))
        return selected, metrics

    async def shutdown(self) -> None:
        """Close optional distributed and feature-store resources."""
        if self.client:
            close_result = self.client.close()
            if inspect.isawaitable(close_result):
                await close_result
        if self.feature_store:
            await self.feature_store.close()
