"""Numerical feature transformer.

This module implements common numerical feature engineering techniques such as
scaling, polynomial features and basic outlier handling.  The implementation is
kept intentionally lightweight but showcases how production ready components can
be structured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ....core.base_components import FeatureTransformer


@dataclass
class NumericalFeatureTransformer(FeatureTransformer):
    """Transformer for numerical columns.

    Parameters
    ----------
    config:
        Transformation options. Supported keys are:
        ``scaler`` ("standard", "robust" or "quantile"), ``polynomial_degree`` and
        ``interaction_only`` (bool).
    """

    config: dict[str, Any]
    _scaler: Any | None = field(init=False, default=None)
    _poly: PolynomialFeatures | None = field(init=False, default=None)
    _outlier_bounds: dict[str, tuple[float, float]] = field(
        init=False, default_factory=dict
    )

    def fit(self, data: pd.DataFrame) -> NumericalFeatureTransformer:  # type: ignore[override]
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if self.config.get("scaler") == "robust":
            self._scaler = RobustScaler().fit(data[numeric_cols])
        elif self.config.get("scaler") == "quantile":
            self._scaler = QuantileTransformer(output_distribution="normal").fit(
                data[numeric_cols]
            )
        else:
            self._scaler = StandardScaler().fit(data[numeric_cols])

        degree = int(self.config.get("polynomial_degree", 1))
        if degree > 1 or self.config.get("interaction_only"):
            self._poly = PolynomialFeatures(
                degree=degree,
                interaction_only=self.config.get("interaction_only", False),
                include_bias=False,
            ).fit(data[numeric_cols])

        if self.config.get("detect_outliers", True):
            for col in numeric_cols:
                q1, q3 = data[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self._outlier_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        if self._scaler is not None:
            scaled = self._scaler.transform(data[numeric_cols])
            result[numeric_cols] = scaled
        if self._poly is not None:
            poly_features = self._poly.transform(data[numeric_cols])
            names = self._poly.get_feature_names_out(numeric_cols)
            poly_df = pd.DataFrame(poly_features, columns=names, index=data.index)
            poly_df = poly_df.drop(columns=list(numeric_cols), errors="ignore")
            result = result.join(poly_df)
        for col in numeric_cols:
            if col in self._outlier_bounds:
                low, high = self._outlier_bounds[col]
                result[f"{col}_outlier"] = (
                    (data[col] < low) | (data[col] > high)
                ).astype(int)
            if self.config.get("bins"):
                bins = int(self.config["bins"])
                result[f"{col}_bin"] = pd.cut(data[col], bins=bins, labels=False)
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        return self.fit(data).transform(data)
