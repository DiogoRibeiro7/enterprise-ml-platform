"""Feature selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV


@dataclass
class FeatureSelector:
    """Automated feature selection helper."""

    config: dict[str, Any]

    def select(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Return the columns retained by the configured selection strategy."""
        if not self.config.get("enabled", True) or y is None:
            return X

        method = str(self.config.get("method", "univariate"))
        if method == "univariate":
            k = int(self.config.get("k_best", min(10, X.shape[1])))
            selector = SelectKBest(score_func=f_classif, k=k).fit(X, y)
            return cast(pd.DataFrame, X.loc[:, selector.get_support()])

        if method == "mutual_info":
            k = int(self.config.get("k_best", min(10, X.shape[1])))
            selector = SelectKBest(score_func=mutual_info_classif, k=k).fit(X, y)
            return cast(pd.DataFrame, X.loc[:, selector.get_support()])

        if method == "lasso":
            model = LassoCV(cv=5).fit(X, y)
            selected = np.abs(model.coef_) > float(
                self.config.get("lasso_threshold", 1e-5)
            )
            return cast(pd.DataFrame, X.loc[:, selected])

        if method == "tree":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            percentile = float(self.config.get("percentile", 50))
            tree_threshold = np.percentile(model.feature_importances_, 100 - percentile)
            return cast(
                pd.DataFrame,
                X.loc[:, model.feature_importances_ >= tree_threshold],
            )

        if method == "correlation":
            correlation_threshold = float(self.config.get("threshold", 0.9))
            correlation = X.corr().abs()
            upper = correlation.where(
                np.triu(np.ones(correlation.shape), k=1).astype(bool)
            )
            to_drop = [
                column
                for column in upper.columns
                if upper[column].gt(correlation_threshold).any()
            ]
            return X.drop(columns=to_drop)

        return X
