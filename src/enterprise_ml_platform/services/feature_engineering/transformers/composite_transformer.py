"""Composite feature transformer for interaction and ratio features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from ....core.base_components import FeatureTransformer


@dataclass
class CompositeFeatureTransformer(FeatureTransformer):
    """Create features derived from combinations of existing columns."""

    config: Dict[str, Any]

    def fit(self, data: pd.DataFrame) -> CompositeFeatureTransformer:  # type: ignore[override]
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        result = data.copy()
        # Ratio features
        for left, right in self.config.get("ratios", []):
            if left in data.columns and right in data.columns:
                denom = data[right].replace(0, pd.NA)
                result[f"{left}_over_{right}"] = data[left] / denom
        # Interaction features
        for pair in self.config.get("interactions", []):
            a, b = pair
            if a in data.columns and b in data.columns:
                result[f"{a}_x_{b}"] = data[a] * data[b]
        # Custom functions
        for name, func in self.config.get("custom", {}).items():
            result[name] = func(data)
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        return self.transform(data)
