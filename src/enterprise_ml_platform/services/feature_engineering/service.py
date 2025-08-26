# Enterprise ML Pipeline - Advanced Feature Engineering Service
# File: services/feature_engineering/service.py

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import structlog
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import dask.dataframe as dd
from dask.distributed import Client, as_completed
import feast
from feast import FeatureStore
import pyarrow as pa
import pyarrow.compute as pc
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    LabelEncoder,
    OneHotEncoder,
    TargetEncoder,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import optuna
from category_encoders import (
    BinaryEncoder,
    BaseNEncoder,
    CatBoostEncoder,
    JamesSteinEncoder,
    WOEEncoder,
)
import polars as pl
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")

from core.pipeline_orchestrator import (
    BasePipelineStage,
    ExecutionContext,
    StageResult,
    PipelineStage,
    ExecutionStatus,
)

logger = structlog.get_logger()


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""

    name: str
    type: str  # 'numerical', 'categorical', 'temporal', 'text', 'composite'
    transformations: List[Dict[str, Any]]
    selection_method: Optional[str] = None
    validation_rules: Optional[List[Dict]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class FeatureMetrics:
    """Metrics for feature engineering"""

    features_created: int = 0
    features_selected: int = 0
    transformation_time_ms: float = 0.0
    selection_time_ms: float = 0.0
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)
    data_quality_improvement: float = 0.0


class FeatureTransformer(ABC):
    """Abstract base class for feature transformers"""

    @abstractmethod
    async def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "FeatureTransformer":
        pass

    @abstractmethod
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    async def fit_transform(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass


class NumericalFeatureTransformer(FeatureTransformer):
    """Advanced numerical feature transformer with multiple strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.feature_stats = {}
        self.outlier_bounds = {}
        self.logger = structlog.get_logger().bind(transformer="numerical")

    async def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "NumericalFeatureTransformer":
        """Fit numerical transformers"""

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = data[col].dropna()

            # Compute statistics
            self.feature_stats[col] = {
                "mean": col_data.mean(),
                "std": col_data.std(),
                "skewness": stats.skew(col_data),
                "kurtosis": stats.kurtosis(col_data),
                "percentiles": col_data.quantile(
                    [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
                ).to_dict(),
            }

            # Determine optimal scaling method
            scaling_method = self._select_optimal_scaler(col_data, target)

            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "quantile":
                scaler = QuantileTransformer(output_distribution="normal")
            else:
                scaler = StandardScaler()  # default

            self.scalers[col] = scaler.fit(col_data.values.reshape(-1, 1))

            # Compute outlier bounds using IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            self.outlier_bounds[col] = {
                "lower": q1 - 1.5 * iqr,
                "upper": q3 + 1.5 * iqr,
            }

        return self

    def _select_optimal_scaler(
        self, data: pd.Series, target: Optional[pd.Series]
    ) -> str:
        """Select optimal scaling method based on data distribution"""

        skewness = abs(stats.skew(data))
        has_outliers = self._detect_outliers(data)

        if has_outliers:
            return "robust"
        elif skewness > 2:
            return "quantile"
        elif data.min() >= 0 and data.max() <= 1:
            return "minmax"
        else:
            return "standard"

    def _detect_outliers(self, data: pd.Series) -> bool:
        """Detect outliers using modified Z-score"""
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return (np.abs(modified_z_scores) > 3.5).any()

    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical features"""

        result = data.copy()
        new_features = pd.DataFrame(index=data.index)

        for col in data.select_dtypes(include=[np.number]).columns:
            if col in self.scalers:
                # Basic scaling
                scaled_values = (
                    self.scalers[col]
                    .transform(data[col].values.reshape(-1, 1))
                    .flatten()
                )
                new_features[f"{col}_scaled"] = scaled_values

                # Generate polynomial features if configured
                if self.config.get("polynomial_features", False):
                    degree = self.config.get("polynomial_degree", 2)
                    for d in range(2, degree + 1):
                        new_features[f"{col}_poly_{d}"] = np.power(scaled_values, d)

                # Generate interaction features if configured
                if self.config.get("interaction_features", False):
                    for other_col in data.select_dtypes(include=[np.number]).columns:
                        if other_col != col and other_col in self.scalers:
                            other_scaled = (
                                self.scalers[other_col]
                                .transform(data[other_col].values.reshape(-1, 1))
                                .flatten()
                            )
                            new_features[f"{col}_x_{other_col}"] = (
                                scaled_values * other_scaled
                            )

                # Generate statistical features
                if self.config.get("statistical_features", False):
                    # Rolling statistics
                    window = self.config.get("rolling_window", 30)
                    new_features[f"{col}_rolling_mean"] = (
                        data[col].rolling(window).mean()
                    )
                    new_features[f"{col}_rolling_std"] = data[col].rolling(window).std()
                    new_features[f"{col}_rolling_skew"] = (
                        data[col].rolling(window).skew()
                    )

                    # Lag features
                    for lag in self.config.get("lags", [1, 7, 30]):
                        new_features[f"{col}_lag_{lag}"] = data[col].shift(lag)

                # Outlier indicators
                bounds = self.outlier_bounds.get(col, {})
                if bounds:
                    new_features[f"{col}_is_outlier"] = (
                        (data[col] < bounds["lower"]) | (data[col] > bounds["upper"])
                    ).astype(int)

                # Binning features
                if self.config.get("binning", False):
                    n_bins = self.config.get("n_bins", 10)
                    new_features[f"{col}_bin"] = pd.cut(
                        data[col], bins=n_bins, labels=False
                    )

        return pd.concat([result, new_features], axis=1)

    async def fit_transform(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step"""
        await self.fit(data, target)
        return await self.transform(data)

    def get_feature_names(self) -> List[str]:
        """Get generated feature names"""
        # Would return actual feature names based on configuration
        return []


class CategoricalFeatureTransformer(FeatureTransformer):
    """Advanced categorical feature transformer with multiple encoding strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encoders = {}
        self.cardinality_info = {}
        self.logger = structlog.get_logger().bind(transformer="categorical")

    async def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "CategoricalFeatureTransformer":
        """Fit categorical encoders"""

        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            col_data = data[col].fillna("MISSING")
            unique_values = col_data.nunique()

            self.cardinality_info[col] = {
                "cardinality": unique_values,
                "null_percentage": data[col].isnull().mean(),
                "most_frequent": col_data.mode().iloc[0]
                if len(col_data.mode()) > 0
                else "MISSING",
            }

            # Select optimal encoding strategy
            encoding_method = self._select_optimal_encoder(
                unique_values, target is not None
            )

            if encoding_method == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(col_data.values.reshape(-1, 1))

            elif encoding_method == "target":
                encoder = TargetEncoder()
                if target is not None:
                    encoder.fit(col_data.values.reshape(-1, 1), target)
                else:
                    # Fallback to label encoding if no target
                    encoder = LabelEncoder()
                    encoder.fit(col_data)

            elif encoding_method == "binary":
                encoder = BinaryEncoder()
                encoder.fit(col_data.values.reshape(-1, 1))

            elif encoding_method == "catboost":
                encoder = CatBoostEncoder()
                if target is not None:
                    encoder.fit(col_data.values.reshape(-1, 1), target)
                else:
                    encoder = LabelEncoder()
                    encoder.fit(col_data)

            elif encoding_method == "woe":
                encoder = WOEEncoder()
                if target is not None:
                    encoder.fit(col_data.values.reshape(-1, 1), target)
                else:
                    encoder = LabelEncoder()
                    encoder.fit(col_data)

            else:  # label encoding
                encoder = LabelEncoder()
                encoder.fit(col_data)

            self.encoders[col] = (encoder, encoding_method)

        return self

    def _select_optimal_encoder(self, cardinality: int, has_target: bool) -> str:
        """Select optimal encoding method based on cardinality and target availability"""

        if cardinality <= 10:
            return "onehot"
        elif cardinality <= 50 and has_target:
            return "target"
        elif cardinality <= 100:
            return "binary"
        elif has_target:
            return "catboost"
        else:
            return "label"

    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features"""

        result = data.copy()
        new_features = pd.DataFrame(index=data.index)

        for col in data.select_dtypes(include=["object", "category"]).columns:
            if col in self.encoders:
                encoder, encoding_method = self.encoders[col]
                col_data = data[col].fillna("MISSING")

                try:
                    if encoding_method == "onehot":
                        encoded = encoder.transform(col_data.values.reshape(-1, 1))
                        feature_names = [
                            f"{col}_{cat}" for cat in encoder.categories_[0]
                        ]
                        for i, name in enumerate(feature_names):
                            new_features[name] = encoded[:, i]

                    elif encoding_method in ["binary", "catboost", "woe"]:
                        encoded = encoder.transform(col_data.values.reshape(-1, 1))
                        if encoded.ndim == 1:
                            new_features[f"{col}_encoded"] = encoded
                        else:
                            for i in range(encoded.shape[1]):
                                new_features[f"{col}_encoded_{i}"] = encoded[:, i]

                    else:  # label, target
                        new_features[f"{col}_encoded"] = encoder.transform(col_data)

                    # Generate frequency encoding
                    if self.config.get("frequency_encoding", True):
                        value_counts = col_data.value_counts()
                        new_features[f"{col}_frequency"] = col_data.map(value_counts)

                    # Generate cardinality features
                    if self.config.get("cardinality_features", True):
                        new_features[f"{col}_is_rare"] = (
                            new_features[f"{col}_frequency"]
                            <= self.config.get("rare_threshold", 5)
                        ).astype(int)

                except Exception as e:
                    self.logger.warning(
                        "Encoding failed for column", column=col, error=str(e)
                    )
                    # Fallback to label encoding
                    fallback_encoder = LabelEncoder()
                    new_features[f"{col}_encoded"] = fallback_encoder.fit_transform(
                        col_data
                    )

        return pd.concat([result, new_features], axis=1)

    async def fit_transform(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step"""
        await self.fit(data, target)
        return await self.transform(data)

    def get_feature_names(self) -> List[str]:
        """Get generated feature names"""
        return []


class TemporalFeatureTransformer(FeatureTransformer):
    """Advanced temporal feature transformer"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datetime_cols = []
        self.seasonal_components = {}
        self.logger = structlog.get_logger().bind(transformer="temporal")

    async def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "TemporalFeatureTransformer":
        """Fit temporal transformers"""

        # Detect datetime columns
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(
                data[col]
            ) or self._is_datetime_string(data[col]):
                self.datetime_cols.append(col)

        # Analyze seasonal patterns if configured
        if self.config.get("seasonal_decomposition", False) and target is not None:
            for col in self.datetime_cols:
                try:
                    # Create time series with datetime index
                    ts_data = (
                        data.set_index(col)[target.name]
                        if hasattr(target, "name")
                        else data.set_index(col)[0]
                    )
                    ts_data = (
                        ts_data.sort_index().resample("D").mean()
                    )  # Daily aggregation

                    if (
                        len(ts_data) > 2 * 365
                    ):  # Need at least 2 years for seasonal decomposition
                        decomposition = seasonal_decompose(
                            ts_data.dropna(), model="additive", period=365
                        )
                        self.seasonal_components[col] = decomposition

                except Exception as e:
                    self.logger.warning(
                        "Seasonal decomposition failed", column=col, error=str(e)
                    )

        return self

    def _is_datetime_string(self, series: pd.Series) -> bool:
        """Check if string series contains datetime-like values"""
        if series.dtype != "object":
            return False

        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False

        try:
            pd.to_datetime(sample)
            return True
        except:
            return False

    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform temporal features"""

        result = data.copy()
        new_features = pd.DataFrame(index=data.index)

        for col in self.datetime_cols:
            if col in data.columns:
                # Convert to datetime if needed
                dt_series = pd.to_datetime(data[col], errors="coerce")

                # Basic datetime features
                new_features[f"{col}_year"] = dt_series.dt.year
                new_features[f"{col}_month"] = dt_series.dt.month
                new_features[f"{col}_day"] = dt_series.dt.day
                new_features[f"{col}_hour"] = dt_series.dt.hour
                new_features[f"{col}_minute"] = dt_series.dt.minute
                new_features[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                new_features[f"{col}_dayofyear"] = dt_series.dt.dayofyear
                new_features[f"{col}_quarter"] = dt_series.dt.quarter
                new_features[f"{col}_week"] = dt_series.dt.isocalendar().week

                # Cyclical encoding for better model performance
                if self.config.get("cyclical_encoding", True):
                    new_features[f"{col}_month_sin"] = np.sin(
                        2 * np.pi * dt_series.dt.month / 12
                    )
                    new_features[f"{col}_month_cos"] = np.cos(
                        2 * np.pi * dt_series.dt.month / 12
                    )
                    new_features[f"{col}_day_sin"] = np.sin(
                        2 * np.pi * dt_series.dt.day / 31
                    )
                    new_features[f"{col}_day_cos"] = np.cos(
                        2 * np.pi * dt_series.dt.day / 31
                    )
                    new_features[f"{col}_hour_sin"] = np.sin(
                        2 * np.pi * dt_series.dt.hour / 24
                    )
                    new_features[f"{col}_hour_cos"] = np.cos(
                        2 * np.pi * dt_series.dt.hour / 24
                    )
                    new_features[f"{col}_dayofweek_sin"] = np.sin(
                        2 * np.pi * dt_series.dt.dayofweek / 7
                    )
                    new_features[f"{col}_dayofweek_cos"] = np.cos(
                        2 * np.pi * dt_series.dt.dayofweek / 7
                    )

                # Business logic features
                new_features[f"{col}_is_weekend"] = (
                    dt_series.dt.dayofweek >= 5
                ).astype(int)
                new_features[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(
                    int
                )
                new_features[f"{col}_is_month_start"] = (
                    dt_series.dt.is_month_start.astype(int)
                )
                new_features[f"{col}_is_quarter_end"] = (
                    dt_series.dt.is_quarter_end.astype(int)
                )

                # Holiday features (basic implementation)
                if self.config.get("holiday_features", True):
                    new_features[f"{col}_is_holiday"] = self._get_holiday_indicator(
                        dt_series
                    )

                # Time since epoch (useful for trend analysis)
                new_features[f"{col}_timestamp"] = dt_series.astype(np.int64) // 10**9

                # Age features (time since reference date)
                reference_date = pd.to_datetime(
                    self.config.get("reference_date", datetime.now())
                )
                new_features[f"{col}_days_since_ref"] = (
                    reference_date - dt_series
                ).dt.days

                # Seasonal features from decomposition
                if col in self.seasonal_components:
                    # This would require more sophisticated matching logic
                    # new_features[f"{col}_trend"] = ...
                    # new_features[f"{col}_seasonal"] = ...
                    pass

        return pd.concat([result, new_features], axis=1)

    def _get_holiday_indicator(self, dt_series: pd.Series) -> pd.Series:
        """Get holiday indicator (simplified implementation)"""
        # This is a basic implementation - in production, you'd use a proper holiday library
        holidays = [
            (1, 1),  # New Year
            (7, 4),  # Independence Day
            (12, 25),  # Christmas
        ]

        is_holiday = pd.Series(False, index=dt_series.index)
        for month, day in holidays:
            mask = (dt_series.dt.month == month) & (dt_series.dt.day == day)
            is_holiday |= mask

        return is_holiday.astype(int)

    async def fit_transform(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        await self.fit(data, target)
        return await self.transform(data)

    def get_feature_names(self) -> List[str]:
        return []


class CompositeFeatureTransformer(FeatureTransformer):
    """Advanced composite feature transformer for domain-specific features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.composite_functions = {}
        self.feature_interactions = {}
        self.logger = structlog.get_logger().bind(transformer="composite")

    async def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "CompositeFeatureTransformer":
        """Fit composite feature transformers"""

        # Register domain-specific composite functions
        self._register_composite_functions()

        # Discover feature interactions using mutual information
        if self.config.get("auto_interactions", False) and target is not None:
            await self._discover_feature_interactions(data, target)

        return self

    def _register_composite_functions(self):
        """Register domain-specific composite feature functions"""

        # Financial domain features
        if "financial" in self.config.get("domains", []):
            self.composite_functions.update(
                {
                    "debt_to_income_ratio": lambda df: df.get("debt", 0)
                    / (df.get("income", 1) + 1e-8),
                    "utilization_rate": lambda df: df.get("balance", 0)
                    / (df.get("credit_limit", 1) + 1e-8),
                    "payment_ratio": lambda df: df.get("payment", 0)
                    / (df.get("balance", 1) + 1e-8),
                }
            )

        # E-commerce domain features
        if "ecommerce" in self.config.get("domains", []):
            self.composite_functions.update(
                {
                    "conversion_rate": lambda df: df.get("purchases", 0)
                    / (df.get("visits", 1) + 1e-8),
                    "avg_order_value": lambda df: df.get("revenue", 0)
                    / (df.get("orders", 1) + 1e-8),
                    "customer_lifetime_value": lambda df: (
                        df.get("avg_order_value", 0)
                        * df.get("purchase_frequency", 0)
                        * df.get("customer_lifespan", 1)
                    ),
                }
            )

        # Custom functions from config
        for name, formula in self.config.get("custom_functions", {}).items():
            try:
                # Safe evaluation of custom formulas
                self.composite_functions[name] = eval(f"lambda df: {formula}")
            except Exception as e:
                self.logger.warning(
                    "Failed to register custom function", name=name, error=str(e)
                )

    async def _discover_feature_interactions(
        self, data: pd.DataFrame, target: pd.Series
    ):
        """Discover important feature interactions using mutual information"""

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        top_k = self.config.get("max_interactions", 20)

        interactions = []

        # Calculate pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                try:
                    # Create interaction feature
                    interaction_feature = data[col1] * data[col2]

                    # Calculate mutual information with target
                    mi_score = mutual_info_classif(
                        interaction_feature.values.reshape(-1, 1),
                        target,
                        random_state=42,
                    )[0]

                    interactions.append((col1, col2, mi_score))

                except Exception as e:
                    continue

        # Select top interactions
        interactions.sort(key=lambda x: x[2], reverse=True)
        self.feature_interactions = {
            f"{col1}_x_{col2}": (col1, col2) for col1, col2, _ in interactions[:top_k]
        }

    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform using composite features"""

        result = data.copy()
        new_features = pd.DataFrame(index=data.index)

        # Apply composite functions
        for feature_name, func in self.composite_functions.items():
            try:
                new_features[feature_name] = func(data)
            except Exception as e:
                self.logger.warning(
                    "Composite function failed", feature=feature_name, error=str(e)
                )

        # Apply discovered interactions
        for interaction_name, (col1, col2) in self.feature_interactions.items():
            if col1 in data.columns and col2 in data.columns:
                new_features[interaction_name] = data[col1] * data[col2]

        # Generate ratio features if configured
        if self.config.get("ratio_features", False):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            ratio_pairs = self.config.get("ratio_pairs", [])

            for col1, col2 in ratio_pairs:
                if col1 in numeric_cols and col2 in numeric_cols:
                    new_features[f"{col1}_div_{col2}"] = data[col1] / (
                        data[col2] + 1e-8
                    )

        return pd.concat([result, new_features], axis=1)

    async def fit_transform(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        await self.fit(data, target)
        return await self.transform(data)

    def get_feature_names(self) -> List[str]:
        return list(self.composite_functions.keys()) + list(
            self.feature_interactions.keys()
        )


class FeatureSelector:
    """Advanced feature selection with multiple strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selected_features = []
        self.feature_scores = {}
        self.logger = structlog.get_logger().bind(component="feature_selector")

    async def select_features(
        self, data: pd.DataFrame, target: pd.Series, method: str = "auto"
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select features using specified method"""

        if method == "auto":
            method = self._select_optimal_method(data, target)

        if method == "statistical":
            return await self._statistical_selection(data, target)
        elif method == "mutual_info":
            return await self._mutual_info_selection(data, target)
        elif method == "recursive":
            return await self._recursive_selection(data, target)
        elif method == "lasso":
            return await self._lasso_selection(data, target)
        elif method == "optuna":
            return await self._optuna_selection(data, target)
        else:
            return data, {}

    def _select_optimal_method(self, data: pd.DataFrame, target: pd.Series) -> str:
        """Select optimal feature selection method based on data characteristics"""

        n_features = data.shape[1]
        n_samples = data.shape[0]

        if n_features > n_samples:
            return "lasso"  # High-dimensional data
        elif n_features > 1000:
            return "mutual_info"  # Many features
        elif self.config.get("use_optuna", False):
            return "optuna"  # Optimization-based
        else:
            return "statistical"  # Default

    async def _statistical_selection(
        self, data: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Statistical feature selection using univariate tests"""

        k = self.config.get("k_best", min(50, data.shape[1]))

        # Select appropriate test based on target type
        if target.dtype in ["int64", "bool"]:
            score_func = f_classif
        else:
            from sklearn.feature_selection import f_regression

            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=k)

        # Handle non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        X_selected = selector.fit_transform(numeric_data, target)
        selected_features = numeric_data.columns[selector.get_support()].tolist()

        # Get feature scores
        feature_scores = dict(zip(numeric_data.columns, selector.scores_))

        return data[selected_features], feature_scores

    async def _mutual_info_selection(
        self, data: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Mutual information based feature selection"""

        numeric_data = data.select_dtypes(include=[np.number])

        # Calculate mutual information scores
        if target.dtype in ["int64", "bool"]:
            mi_scores = mutual_info_classif(numeric_data, target, random_state=42)
        else:
            from sklearn.feature_selection import mutual_info_regression

            mi_scores = mutual_info_regression(numeric_data, target, random_state=42)

        # Select top k features
        k = self.config.get("k_best", min(50, len(mi_scores)))
        top_indices = np.argsort(mi_scores)[-k:]

        selected_features = numeric_data.columns[top_indices].tolist()
        feature_scores = dict(zip(numeric_data.columns, mi_scores))

        return data[selected_features], feature_scores

    async def _recursive_selection(
        self, data: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Recursive feature elimination"""

        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        numeric_data = data.select_dtypes(include=[np.number])

        # Select appropriate estimator
        if target.dtype in ["int64", "bool"]:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)

        n_features = self.config.get(
            "n_features_to_select", min(20, numeric_data.shape[1])
        )
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)

        rfe.fit(numeric_data, target)

        selected_features = numeric_data.columns[rfe.support_].tolist()
        feature_scores = dict(zip(numeric_data.columns, rfe.ranking_))

        return data[selected_features], feature_scores

    async def _lasso_selection(
        self, data: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """LASSO-based feature selection"""

        from sklearn.linear_model import LassoCV, LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler

        numeric_data = data.select_dtypes(include=[np.number])

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        # Select appropriate model
        if target.dtype in ["int64", "bool"]:
            model = LogisticRegressionCV(
                penalty="l1", solver="liblinear", cv=5, random_state=42
            )
        else:
            model = LassoCV(cv=5, random_state=42)

        model.fit(X_scaled, target)

        # Get feature coefficients
        coefficients = (
            np.abs(model.coef_).flatten()
            if hasattr(model, "coef_")
            else model.feature_importances_
        )

        # Select features with non-zero coefficients
        threshold = self.config.get("lasso_threshold", 1e-5)
        selected_mask = coefficients > threshold

        selected_features = numeric_data.columns[selected_mask].tolist()
        feature_scores = dict(zip(numeric_data.columns, coefficients))

        return data[selected_features], feature_scores

    async def _optuna_selection(
        self, data: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Optuna-based feature selection optimization"""

        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        numeric_data = data.select_dtypes(include=[np.number])

        def objective(trial):
            # Suggest feature subset
            n_features = trial.suggest_int(
                "n_features", 5, min(50, numeric_data.shape[1])
            )
            feature_indices = []

            for i in range(numeric_data.shape[1]):
                if trial.suggest_categorical(f"feature_{i}", [True, False]):
                    feature_indices.append(i)

            # Ensure minimum number of features
            if len(feature_indices) < n_features:
                feature_indices = list(range(min(n_features, numeric_data.shape[1])))

            X_subset = numeric_data.iloc[:, feature_indices]

            # Select appropriate model
            if target.dtype in ["int64", "bool"]:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                scoring = "accuracy"
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                scoring = "r2"

            # Cross-validation score
            scores = cross_val_score(model, X_subset, target, cv=3, scoring=scoring)
            return scores.mean()

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.get("optuna_trials", 100))

        # Extract best features
        best_params = study.best_params
        selected_indices = [
            i
            for i in range(numeric_data.shape[1])
            if best_params.get(f"feature_{i}", False)
        ]

        selected_features = numeric_data.columns[selected_indices].tolist()

        # Calculate feature importance based on trial history
        feature_scores = {}
        for col in numeric_data.columns:
            feature_scores[col] = 0.0  # Default score

        return data[selected_features], feature_scores


class FeatureEngineeringService:
    """Enterprise feature engineering service with orchestration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transformers = {}
        self.feature_selector = FeatureSelector(config.get("feature_selection", {}))
        self.feature_store = None
        self.dask_client = None
        self.logger = structlog.get_logger().bind(service="feature_engineering")
        self.metrics = FeatureMetrics()

    async def initialize(self):
        """Initialize service components"""

        # Initialize Dask client for distributed processing
        if self.config.get("use_dask", False):
            self.dask_client = Client(
                self.config.get("dask_scheduler", "localhost:8786")
            )

        # Initialize Feast feature store if configured
        if self.config.get("feast_config"):
            self.feature_store = FeatureStore(
                repo_path=self.config["feast_config"]["repo_path"]
            )

        # Initialize transformers
        await self._initialize_transformers()

        self.logger.info("Feature engineering service initialized")

    async def _initialize_transformers(self):
        """Initialize feature transformers based on configuration"""

        transformer_configs = self.config.get("transformers", {})

        if "numerical" in transformer_configs:
            self.transformers["numerical"] = NumericalFeatureTransformer(
                transformer_configs["numerical"]
            )

        if "categorical" in transformer_configs:
            self.transformers["categorical"] = CategoricalFeatureTransformer(
                transformer_configs["categorical"]
            )

        if "temporal" in transformer_configs:
            self.transformers["temporal"] = TemporalFeatureTransformer(
                transformer_configs["temporal"]
            )

        if "composite" in transformer_configs:
            self.transformers["composite"] = CompositeFeatureTransformer(
                transformer_configs["composite"]
            )

    async def engineer_features(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        feature_configs: Optional[List[FeatureConfig]] = None,
    ) -> Tuple[pd.DataFrame, FeatureMetrics]:
        """Main feature engineering pipeline"""

        start_time = datetime.now()
        original_features = data.shape[1]

        # Use Dask for large datasets
        if self.config.get("use_dask", False) and len(data) > self.config.get(
            "dask_threshold", 100000
        ):
            data = dd.from_pandas(
                data, npartitions=self.config.get("dask_partitions", 4)
            )

        result_data = data.copy()

        # Apply transformers in sequence
        for transformer_name, transformer in self.transformers.items():
            self.logger.info("Applying transformer", transformer=transformer_name)

            try:
                result_data = await transformer.fit_transform(result_data, target)
                self.logger.info(
                    "Transformer completed",
                    transformer=transformer_name,
                    features_added=result_data.shape[1]
                    - (
                        result_data.shape[1]
                        if "result_data" in locals()
                        else original_features
                    ),
                )
            except Exception as e:
                self.logger.error(
                    "Transformer failed", transformer=transformer_name, error=str(e)
                )
                continue

        # Convert back from Dask if needed
        if hasattr(result_data, "compute"):
            result_data = result_data.compute()

        # Feature selection
        if target is not None and self.config.get("feature_selection", {}).get(
            "enabled", True
        ):
            selection_start = datetime.now()

            selection_method = self.config.get("feature_selection", {}).get(
                "method", "auto"
            )
            selected_data, feature_scores = await self.feature_selector.select_features(
                result_data, target, selection_method
            )

            selection_time = (datetime.now() - selection_start).total_seconds() * 1000

            self.metrics.features_selected = selected_data.shape[1]
            self.metrics.selection_time_ms = selection_time
            self.metrics.feature_importance_scores = feature_scores

            result_data = selected_data

        # Update metrics
        transformation_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.features_created = result_data.shape[1] - original_features
        self.metrics.transformation_time_ms = transformation_time

        # Store features in feature store if configured
        if self.feature_store and self.config.get("store_features", False):
            await self._store_features(result_data)

        return result_data, self.metrics

    async def _store_features(self, features: pd.DataFrame):
        """Store features in Feast feature store"""

        try:
            # This would require proper Feast configuration and entity definitions
            # Implementation depends on specific Feast setup
            self.logger.info(
                "Storing features in feature store", features=features.shape[1]
            )
        except Exception as e:
            self.logger.error("Failed to store features", error=str(e))

    async def shutdown(self):
        """Shutdown service components"""

        if self.dask_client:
            await self.dask_client.close()

        self.logger.info("Feature engineering service shutdown complete")


# Feature Engineering Pipeline Stage
class FeatureEngineeringStage(BasePipelineStage):
    """Pipeline stage for feature engineering"""

    def __init__(self, service: FeatureEngineeringService):
        super().__init__("feature_engineering", PipelineStage.FEATURE_ENGINEERING)
        self.service = service

    async def _execute_stage(self, context: ExecutionContext) -> StageResult:
        """Execute feature engineering stage"""

        try:
            # Get data from previous stage
            input_data = context.artifacts.get("processed_data")
            if not input_data:
                raise ValueError("No processed data found from previous stages")

            # Load data
            data = pd.read_parquet(input_data)

            # Get target if available
            target = None
            target_column = context.config.get("feature_engineering", {}).get(
                "target_column"
            )
            if target_column and target_column in data.columns:
                target = data[target_column]
                data = data.drop(columns=[target_column])

            # Initialize service
            await self.service.initialize()

            # Engineer features
            engineered_data, metrics = await self.service.engineer_features(
                data, target
            )

            # Store results
            output_path = "/tmp/engineered_features.parquet"
            engineered_data.to_parquet(output_path)

            return StageResult(
                stage=self.stage_type,
                status=ExecutionStatus.SUCCESS,
                output=engineered_data,
                artifacts={"engineered_features": output_path},
                metrics={
                    "features_created": metrics.features_created,
                    "features_selected": metrics.features_selected,
                    "transformation_time_ms": metrics.transformation_time_ms,
                    "selection_time_ms": metrics.selection_time_ms,
                    "data_quality_improvement": metrics.data_quality_improvement,
                },
            )

        except Exception as e:
            self.logger.error("Feature engineering stage failed", error=str(e))
            raise e

    async def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup feature engineering resources"""
        await self.service.shutdown()


# Configuration example
FEATURE_ENGINEERING_CONFIG = {
    "use_dask": True,
    "dask_scheduler": "localhost:8786",
    "dask_threshold": 100000,
    "dask_partitions": 8,
    "transformers": {
        "numerical": {
            "polynomial_features": True,
            "polynomial_degree": 2,
            "interaction_features": True,
            "statistical_features": True,
            "rolling_window": 30,
            "lags": [1, 7, 30],
            "binning": True,
            "n_bins": 10,
        },
        "categorical": {
            "frequency_encoding": True,
            "cardinality_features": True,
            "rare_threshold": 5,
        },
        "temporal": {
            "cyclical_encoding": True,
            "holiday_features": True,
            "reference_date": "2024-01-01",
            "seasonal_decomposition": True,
        },
        "composite": {
            "domains": ["financial", "ecommerce"],
            "auto_interactions": True,
            "max_interactions": 50,
            "ratio_features": True,
            "ratio_pairs": [["revenue", "visits"], ["debt", "income"]],
            "custom_functions": {
                "risk_score": "df['debt'] / df['income'] + df['late_payments'] * 0.1"
            },
        },
    },
    "feature_selection": {
        "enabled": True,
        "method": "auto",  # auto, statistical, mutual_info, recursive, lasso, optuna
        "k_best": 50,
        "lasso_threshold": 1e-5,
        "use_optuna": True,
        "optuna_trials": 100,
    },
    "feast_config": {"repo_path": "./feature_repo"},
    "store_features": True,
}
