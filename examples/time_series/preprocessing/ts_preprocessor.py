from __future__ import annotations

import pandas as pd
from typing import Optional


class TSPreprocessor:
    """Basic time series preprocessing and feature engineering."""

    def __init__(self, freq: Optional[str] = None) -> None:
        self.freq = freq

    def fit_transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Fill gaps, normalise and create simple lag features."""
        series = df[target]
        if self.freq:
            series = series.asfreq(self.freq)
        series = series.fillna(method="ffill").fillna(method="bfill")
        df[target] = series
        df[f"{target}_lag1"] = series.shift(1).fillna(series.mean())
        df[f"{target}_rolling_mean3"] = series.rolling(window=3, min_periods=1).mean()
        return df
