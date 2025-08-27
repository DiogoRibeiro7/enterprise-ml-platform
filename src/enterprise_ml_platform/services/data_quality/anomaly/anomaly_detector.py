"""Simple statistical anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class AnomalyDetector:
    """Detect outliers using a Z-score approach."""

    threshold: float = 3.0

    def detect(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Return indices of rows considered anomalous per column."""

        anomalies: Dict[str, List[int]] = {}
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
