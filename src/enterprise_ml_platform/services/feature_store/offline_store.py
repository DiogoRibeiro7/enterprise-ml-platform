from __future__ import annotations
"""Lightweight offline feature store using pandas DataFrames."""

from typing import Dict, Optional
import pandas as pd

from ..monitoring.collectors.metrics_collector import MetricsCollector


class OfflineFeatureStore:
    """In-memory offline store supporting time-travel queries."""

    def __init__(self, metrics: Optional[MetricsCollector] = None) -> None:
        self._store: Dict[tuple[str, str], pd.DataFrame] = {}
        self.metrics = metrics

    # ------------------------------------------------------------------
    async def write_features(
        self, name: str, version: str, df: pd.DataFrame
    ) -> None:
        self._store[(name, version)] = df.copy()

    # ------------------------------------------------------------------
    async def get_features(
        self,
        name: str,
        version: str,
        entity_id: str,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        df = self._store.get((name, version))
        if df is None:
            return {}
        row = df[df["entity_id"] == entity_id]
        if row.empty:
            return {}
        if as_of is not None and "timestamp" in row:
            row = row[row["timestamp"] <= as_of]
            if row.empty:
                return {}
            row = row.sort_values("timestamp").iloc[-1:]
        result = row.drop(columns=["entity_id", "timestamp"], errors="ignore")
        return result.iloc[0].to_dict()
