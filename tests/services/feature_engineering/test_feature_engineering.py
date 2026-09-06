import asyncio

import pandas as pd
import pytest

from enterprise_ml_platform.services.feature_engineering import (
    FeatureEngineeringService,
)
from enterprise_ml_platform.services.feature_engineering.transformers import (
    CategoricalFeatureTransformer,
)


@pytest.fixture
def sample_data():
    frame = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "num2": [2, 4, 6, 8, 10],
            "cat": ["a", "b", "a", "c", "b"],
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        }
    )
    target = pd.Series([0, 1, 0, 1, 0])
    return frame, target


def test_service_creates_features(sample_data):
    frame, target = sample_data
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
    engineered, metrics = asyncio.run(service.engineer_features(frame, target))
    assert metrics.features_created >= 0
    assert metrics.features_selected > 0
    assert len(engineered) == len(frame)


def test_categorical_refit_replaces_learned_state() -> None:
    transformer = CategoricalFeatureTransformer({"one_hot_threshold": 2})
    high_cardinality = pd.DataFrame({"city": ["Porto", "Lisbon", "Braga"]})
    low_cardinality = pd.DataFrame({"city": ["Porto", "Porto", "Porto"]})

    assert transformer.fit(high_cardinality) is transformer
    transformer.fit(low_cardinality)

    transformed = transformer.transform(low_cardinality)
    assert transformed.columns.tolist() == ["city_Porto"]


def test_shutdown_supports_synchronous_dask_client() -> None:
    class SynchronousClient:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    service = FeatureEngineeringService({})
    client = SynchronousClient()
    service.client = client

    asyncio.run(service.shutdown())

    assert client.closed
