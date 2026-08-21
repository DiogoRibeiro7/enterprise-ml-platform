"""Temporal feature transformer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ....core.base_components import FeatureTransformer


@dataclass
class TemporalFeatureTransformer(FeatureTransformer):
    """Generate features from datetime columns."""

    config: dict[str, Any]

    def fit(self, data: pd.DataFrame) -> TemporalFeatureTransformer:  # type: ignore[override]
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        result = data.copy()
        date_cols = data.select_dtypes(include=["datetime", "datetimetz"]).columns
        ref_date = pd.to_datetime(self.config.get("reference_date", datetime.utcnow()))
        for col in date_cols:
            series = pd.to_datetime(data[col])
            result[f"{col}_sin_day"] = np.sin(2 * np.pi * series.dt.dayofyear / 365)
            result[f"{col}_cos_day"] = np.cos(2 * np.pi * series.dt.dayofyear / 365)
            result[f"{col}_is_weekend"] = series.dt.weekday >= 5
            result[f"{col}_days_since_ref"] = (series - ref_date).dt.days
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        return self.transform(data)
