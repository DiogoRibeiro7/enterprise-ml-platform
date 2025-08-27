"""Categorical feature transformer with multiple encoding strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from ....core.base_components import FeatureTransformer


@dataclass
class CategoricalFeatureTransformer(FeatureTransformer):
    """Encode categorical columns using a variety of strategies.

    Parameters
    ----------
    config:
        ``one_hot_threshold`` determines the maximum cardinality for one-hot
        encoding.  Columns above this threshold will be frequency encoded unless
        a target is supplied, in which case target encoding is used.
    """

    config: Dict[str, Any]
    _encoders: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)
    _one_hot_cols: list[str] = field(init=False, default_factory=list)
    _freq_cols: list[str] = field(init=False, default_factory=list)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> CategoricalFeatureTransformer:  # type: ignore[override]
        cat_cols = data.select_dtypes(include=["object", "category"]).columns
        threshold = int(self.config.get("one_hot_threshold", 10))
        for col in cat_cols:
            series = data[col].astype(str).fillna("__MISSING__")
            n_unique = series.nunique()
            if n_unique <= threshold:
                self._one_hot_cols.append(col)
            else:
                if target is not None:
                    means = target.groupby(series).mean()
                    self._encoders[col] = means.to_dict()
                else:
                    freq = series.value_counts(normalize=True)
                    self._encoders[col] = freq.to_dict()
                    self._freq_cols.append(col)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        result = data.copy()
        for col in self._one_hot_cols:
            dummies = pd.get_dummies(result[col].astype(str).fillna("__MISSING__"), prefix=col)
            result = result.drop(columns=[col]).join(dummies)
        for col, mapping in self._encoders.items():
            series = result[col].astype(str).fillna("__MISSING__")
            result[col] = series.map(mapping).fillna(0.0)
        return result

    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:  # type: ignore[override]
        return self.fit(data, target).transform(data)
