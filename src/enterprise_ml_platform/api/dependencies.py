"""Dependency injection components for the API layer.

This module provides helpers that can be used with FastAPI's dependency
injection system. A lightweight :class:`ModelRegistry` is provided to manage
loaded models during the lifetime of the application.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import structlog


class ModelRegistry:
    """In-memory registry for loaded ML models.

    The registry is intentionally simple and suited for demonstration and unit
    testing purposes. Models are stored in-memory and can be loaded or unloaded
    at runtime through the API.
    """

    def __init__(self) -> None:
        self._models: Dict[str, object] = {}

    # ------------------------------------------------------------------
    def list_models(self) -> List[str]:
        """Return a list of loaded model names."""

        return list(self._models.keys())

    # ------------------------------------------------------------------
    def get(self, name: str) -> Optional[object]:
        """Retrieve a model by name.

        Args:
            name: Registered model name.

        Returns:
            The model object if loaded, otherwise ``None``.
        """

        return self._models.get(name)

    # ------------------------------------------------------------------
    def info(self, name: str) -> "ModelInfo":
        """Return metadata for a loaded model.

        Args:
            name: Registered model name.

        Raises:
            KeyError: If the model is not loaded.
        """

        from .schemas.models import ModelInfo

        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        return ModelInfo(name=name, version="1.0", description="demo model")

    # ------------------------------------------------------------------
    def load(self, name: str) -> "ModelInfo":
        """Load a demo model into the registry.

        A simple logistic regression model trained on the Iris dataset is used
        for demonstration. In a production system this would interface with a
        model store or registry.
        """

        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from .schemas.models import ModelInfo

        data = load_iris()
        model = LogisticRegression(max_iter=200)
        model.fit(data.data, data.target)
        self._models[name] = model
        return ModelInfo(name=name, version="1.0", description="Iris classifier")

    # ------------------------------------------------------------------
    def unload(self, name: str) -> None:
        """Remove a model from the registry if it exists."""

        self._models.pop(name, None)


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """FastAPI dependency providing the shared :class:`ModelRegistry`."""

    return _registry


def get_logger() -> structlog.BoundLogger:
    """Return a structured logger bound to the current request."""

    return structlog.get_logger()
