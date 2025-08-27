from __future__ import annotations

from typing import Dict
import pandas as pd


class SeasonalDecomposer:
    """Naive seasonal-trend decomposition using moving averages."""

    def __init__(self, period: int = 12) -> None:
        self.period = period

    def decompose(self, series: pd.Series) -> Dict[str, pd.Series]:
        trend = series.rolling(window=self.period, min_periods=1, center=True).mean()
        seasonal = series - trend
        residual = series - trend - seasonal
        return {"trend": trend, "seasonal": seasonal, "residual": residual}
