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
    """Generate cyclical and relative features from datetime columns."""

    config: dict[str, Any]

    def fit(self, data: pd.DataFrame) -> TemporalFeatureTransformer:
        """Return this stateless transformer for protocol compatibility."""
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Expand every datetime column into temporal model features."""
        result = data.copy()
        date_columns = data.select_dtypes(include=["datetime", "datetimetz"]).columns
        reference_date = pd.to_datetime(
            self.config.get("reference_date", datetime.now())
        )
        for column in date_columns:
            series = pd.to_datetime(data[column])
            result[f"{column}_sin_day"] = np.sin(2 * np.pi * series.dt.dayofyear / 365)
            result[f"{column}_cos_day"] = np.cos(2 * np.pi * series.dt.dayofyear / 365)
            result[f"{column}_is_weekend"] = series.dt.weekday >= 5
            result[f"{column}_days_since_ref"] = (series - reference_date).dt.days
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform ``data``; no fitting is required for temporal features."""
        return self.transform(data)
