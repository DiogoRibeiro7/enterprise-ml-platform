"""Composite feature transformer for interaction and ratio features."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from ....core.base_components import FeatureTransformer


@dataclass
class CompositeFeatureTransformer(FeatureTransformer):
    """Create features derived from combinations of existing columns."""

    config: dict[str, Any]

    def fit(self, data: pd.DataFrame) -> CompositeFeatureTransformer:
        """Return this stateless transformer for protocol compatibility."""
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create configured ratio, interaction, and custom features."""
        result = data.copy()
        ratios = cast(Sequence[tuple[str, str]], self.config.get("ratios", ()))
        for left, right in ratios:
            if left in data.columns and right in data.columns:
                denominator = data[right].replace(0, pd.NA)
                result[f"{left}_over_{right}"] = data[left] / denominator

        interactions = cast(
            Sequence[tuple[str, str]], self.config.get("interactions", ())
        )
        for left, right in interactions:
            if left in data.columns and right in data.columns:
                result[f"{left}_x_{right}"] = data[left] * data[right]

        custom_features = cast(
            dict[str, Callable[[pd.DataFrame], Any]],
            self.config.get("custom", {}),
        )
        for name, create_feature in custom_features.items():
            result[name] = create_feature(data)
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform ``data``; no fitting is required for composite features."""
        return self.transform(data)
