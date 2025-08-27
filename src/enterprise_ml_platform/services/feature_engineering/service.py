"""Feature engineering service orchestrating transformers and selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import structlog

try:  # pragma: no cover - optional dependency
    from dask.distributed import Client
except Exception:  # pragma: no cover
    Client = None  # type: ignore

from .transformers.numerical_transformer import NumericalFeatureTransformer
from .transformers.categorical_transformer import CategoricalFeatureTransformer
from .transformers.temporal_transformer import TemporalFeatureTransformer
from .transformers.composite_transformer import CompositeFeatureTransformer
from .selectors.feature_selector import FeatureSelector


@dataclass
class FeatureEngineeringMetrics:
    """Simple metrics returned by :class:`FeatureEngineeringService`."""

    features_created: int
    features_selected: int


class FeatureEngineeringService:
    """Apply feature transformers and selection.

    The service optionally utilises Dask for distributed processing.  Integration
    with a Feast feature store is supported when ``feast_repo`` is supplied.
    """

    def __init__(self, config: Dict[str, Dict]) -> None:
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.client: Optional[Client] = None
        if config.get("use_dask") and Client is not None:
            self.client = Client(processes=False)

    async def engineer_features(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, FeatureEngineeringMetrics]:
        df = data.copy()
        transformers = []
        cfg = self.config.get("transformers", {})
        if "numerical" in cfg:
            transformers.append(NumericalFeatureTransformer(cfg["numerical"]))
        if "categorical" in cfg:
            transformers.append(CategoricalFeatureTransformer(cfg["categorical"]))
        if "temporal" in cfg:
            transformers.append(TemporalFeatureTransformer(cfg["temporal"]))
        if "composite" in cfg:
            transformers.append(CompositeFeatureTransformer(cfg["composite"]))

        for t in transformers:
            if target is not None and isinstance(t, CategoricalFeatureTransformer):
                df = t.fit_transform(df, target)
            else:
                df = t.fit_transform(df)
        created = df.shape[1] - data.shape[1]

        # Remove remaining non-numeric columns before selection and fill missing values
        df = df.select_dtypes(exclude=["datetime", "datetimetz"]).copy()
        df = df.fillna(0)

        selector_cfg = self.config.get("feature_selection", {})
        selector = FeatureSelector(selector_cfg)
        df_selected = selector.select(df, target)
        metrics = FeatureEngineeringMetrics(
            features_created=created,
            features_selected=df_selected.shape[1],
        )

        if self.config.get("feast_repo"):
            try:
                from feast import FeatureStore

                store = FeatureStore(repo_path=self.config["feast_repo"])
                store.write_to_online_store(df_selected)
            except Exception as exc:  # pragma: no cover - optional dependency
                self.logger.warning("Feast integration failed", error=str(exc))
        return df_selected, metrics

    async def shutdown(self) -> None:
        if self.client:
            await self.client.close()
