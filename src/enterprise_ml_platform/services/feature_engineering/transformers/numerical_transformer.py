"""Numerical feature transformer.

This module implements scaling, polynomial features, and basic outlier handling.
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
    """Transform numerical columns according to the supplied configuration."""

    config: dict[str, Any]
    _scaler: StandardScaler | RobustScaler | QuantileTransformer | None = field(
        init=False, default=None
    )
    _poly: PolynomialFeatures | None = field(init=False, default=None)
    _outlier_bounds: dict[str, tuple[float, float]] = field(
        init=False, default_factory=dict
    )

    def fit(self, data: pd.DataFrame) -> NumericalFeatureTransformer:
        """Learn scaling, polynomial, and outlier parameters from ``data``."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        scaler_name = str(self.config.get("scaler", "standard"))
        if scaler_name == "robust":
            self._scaler = RobustScaler().fit(data[numeric_columns])
        elif scaler_name == "quantile":
            self._scaler = QuantileTransformer(output_distribution="normal").fit(
                data[numeric_columns]
            )
        else:
            self._scaler = StandardScaler().fit(data[numeric_columns])

        degree = int(self.config.get("polynomial_degree", 1))
        interaction_only = bool(self.config.get("interaction_only", False))
        self._poly = None
        if degree > 1 or interaction_only:
            self._poly = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=False,
            ).fit(data[numeric_columns])

        self._outlier_bounds.clear()
        if self.config.get("detect_outliers", True):
            for column in numeric_columns:
                quantiles = data[column].quantile([0.25, 0.75])
                lower_quartile = float(quantiles.loc[0.25])
                upper_quartile = float(quantiles.loc[0.75])
                interquartile_range = upper_quartile - lower_quartile
                self._outlier_bounds[column] = (
                    lower_quartile - 1.5 * interquartile_range,
                    upper_quartile + 1.5 * interquartile_range,
                )
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the parameters learned by :meth:`fit`."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        if self._scaler is not None:
            result[numeric_columns] = self._scaler.transform(data[numeric_columns])
        if self._poly is not None:
            polynomial_values = self._poly.transform(data[numeric_columns])
            names = self._poly.get_feature_names_out(numeric_columns)
            polynomial_data = pd.DataFrame(
                polynomial_values, columns=names, index=data.index
            )
            polynomial_data = polynomial_data.drop(
                columns=list(numeric_columns), errors="ignore"
            )
            result = result.join(polynomial_data)
        for column in numeric_columns:
            if column in self._outlier_bounds:
                lower, upper = self._outlier_bounds[column]
                result[f"{column}_outlier"] = (
                    (data[column] < lower) | (data[column] > upper)
                ).astype(int)
            if self.config.get("bins"):
                bins = int(self.config["bins"])
                result[f"{column}_bin"] = pd.cut(data[column], bins=bins, labels=False)
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit this transformer and transform ``data`` in one operation."""
        return self.fit(data).transform(data)
