from __future__ import annotations

import pandas as pd
import numpy as np


class TSAnomalyDetector:
    """Z-score based anomaly detector for time series."""

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def detect(self, series: pd.Series) -> pd.Series:
        z = (series - series.mean()) / series.std(ddof=0)
        return z.abs() > self.threshold
