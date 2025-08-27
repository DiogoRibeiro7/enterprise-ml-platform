"""Feature selection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier


@dataclass
class FeatureSelector:
    """Automated feature selection helper."""

    config: Dict[str, Any]

    def select(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self.config.get("enabled", True) or y is None:
            return X
        method = self.config.get("method", "univariate")
        if method == "univariate":
            k = int(self.config.get("k_best", min(10, X.shape[1])))
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)
            cols = X.columns[selector.get_support()]
            return X[cols]
        if method == "mutual_info":
            k = int(self.config.get("k_best", min(10, X.shape[1])))
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selector.fit(X, y)
            cols = X.columns[selector.get_support()]
            return X[cols]
        if method == "lasso":
            model = LassoCV(cv=5).fit(X, y)
            cols = X.columns[np.abs(model.coef_) > float(self.config.get("lasso_threshold", 1e-5))]
            return X[cols]
        if method == "tree":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            thresh = np.percentile(importances, 100 - self.config.get("percentile", 50))
            cols = X.columns[importances >= thresh]
            return X[cols]
        if method == "correlation":
            thresh = float(self.config.get("threshold", 0.9))
            corr = X.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
            return X.drop(columns=to_drop)
        return X
