"""Base protocol definitions for core components.

This module declares the core abstract protocols used throughout the
Enterprise ML Platform. Implementations of these protocols should provide
functional logic for data access, feature transformation and model
training.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataConnector(Protocol):
    """Protocol for data source connectors.

    Implementations handle communication with external data sources
    such as databases, object stores or streaming platforms.
    """

    def connect(self, config: dict[str, Any]) -> None:
        """Establish connection to a data source.

        Args:
            config: Connection parameters specific to the data source.
        """
        ...

    def fetch(self) -> Any:
        """Retrieve data from the connected source.

        Returns:
            Retrieved data object, typically a ``pandas`` DataFrame or similar
            structure.
        """
        ...


@runtime_checkable
class FeatureTransformer(Protocol):
    """Protocol for feature transformation components.

    A feature transformer is responsible for preparing raw data for model
    consumption by applying feature engineering techniques.
    """

    def fit(self, data: Any) -> None:
        """Fit the transformer using input data.

        Args:
            data: Input data used to compute transformation parameters.
        """
        ...

    def transform(self, data: Any) -> Any:
        """Transform the input data into feature representations.

        Args:
            data: Raw input data to transform.

        Returns:
            Transformed feature data.
        """
        ...


@runtime_checkable
class ModelTrainer(Protocol):
    """Protocol for model training components.

    Implementations encapsulate algorithms used for training predictive
    models and evaluating their performance.
    """

    def train(self, features: Any, targets: Any) -> Any:
        """Train a model using the provided features and targets.

        Args:
            features: Input features for training.
            targets: Target values corresponding to the features.

        Returns:
            Trained model instance.
        """
        ...

    def evaluate(self, model: Any, features: Any, targets: Any) -> dict[str, float]:
        """Evaluate a trained model.

        Args:
            model: Trained model to evaluate.
            features: Evaluation features.
            targets: True target values.

        Returns:
            A mapping of metric names to their computed values.
        """
        ...

    def save(self, model: Any, path: str) -> None:
        """Persist a trained model to the given path.

        Args:
            model: Trained model instance.
            path: Destination path where the model should be stored.
        """
        ...
