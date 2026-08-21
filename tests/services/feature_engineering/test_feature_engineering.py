import asyncio
import pathlib
import sys

import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3] / "src"))

from enterprise_ml_platform.services.feature_engineering import (
    FeatureEngineeringService,
)


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "num2": [2, 4, 6, 8, 10],
            "cat": ["a", "b", "a", "c", "b"],
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        }
    )
    target = pd.Series([0, 1, 0, 1, 0])
    return df, target


def test_service_creates_features(sample_data):
    df, target = sample_data
    service = FeatureEngineeringService(
        {
            "transformers": {
                "numerical": {"polynomial_degree": 2, "bins": 2},
                "categorical": {"one_hot_threshold": 3},
                "temporal": {"reference_date": "2020-01-01"},
                "composite": {"ratios": [("num2", "num1")]},
            },
            "feature_selection": {"method": "univariate", "k_best": 3},
        }
    )
    engineered, metrics = asyncio.run(service.engineer_features(df, target))
    assert metrics.features_created >= 0
    assert metrics.features_selected > 0
