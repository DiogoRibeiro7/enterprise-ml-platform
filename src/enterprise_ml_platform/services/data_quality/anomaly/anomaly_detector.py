"""Simple statistical anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AnomalyDetector:
    """Detect outliers using a Z-score approach."""

    threshold: float = 3.0

    def detect(self, df: pd.DataFrame) -> dict[str, list[int]]:
        """Return indices of rows considered anomalous per column."""

        anomalies: dict[str, list[int]] = {}
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            series = numeric[col]
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0:
                continue
            z_scores = ((series - mean) / std).abs()
            idx = z_scores[z_scores > self.threshold].index.tolist()
            if idx:
                anomalies[col] = idx
        return anomalies
