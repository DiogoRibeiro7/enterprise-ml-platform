"""Categorical feature transformer with multiple encoding strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ....core.base_components import FeatureTransformer


@dataclass
class CategoricalFeatureTransformer(FeatureTransformer):
    """Encode categorical columns using one-hot, target, or frequency encoding."""

    config: dict[str, Any]
    _encoders: dict[str, dict[str, float]] = field(init=False, default_factory=dict)
    _one_hot_cols: list[str] = field(init=False, default_factory=list)

    def fit(
        self, data: pd.DataFrame, target: pd.Series | None = None
    ) -> CategoricalFeatureTransformer:
        """Learn encodings from ``data`` and replace any previously fitted state."""
        self._encoders.clear()
        self._one_hot_cols.clear()

        categorical_columns = data.select_dtypes(include=["object", "category"]).columns
        threshold = int(self.config.get("one_hot_threshold", 10))
        for column in categorical_columns:
            series = data[column].fillna("__MISSING__").astype(str)
            if series.nunique() <= threshold:
                self._one_hot_cols.append(column)
            elif target is not None:
                means = target.groupby(series).mean()
                self._encoders[column] = {
                    str(category): float(mean) for category, mean in means.items()
                }
            else:
                frequencies = series.value_counts(normalize=True)
                self._encoders[column] = {
                    str(category): float(frequency)
                    for category, frequency in frequencies.items()
                }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the encodings learned by :meth:`fit`."""
        result = data.copy()
        for column in self._one_hot_cols:
            dummies = pd.get_dummies(
                result[column].fillna("__MISSING__").astype(str), prefix=column
            )
            result = result.drop(columns=[column]).join(dummies)
        for column, mapping in self._encoders.items():
            series = result[column].fillna("__MISSING__").astype(str)
            result[column] = series.map(mapping).fillna(0.0)
        return result

    def fit_transform(
        self, data: pd.DataFrame, target: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit this transformer and transform ``data`` in one operation."""
        return self.fit(data, target).transform(data)
