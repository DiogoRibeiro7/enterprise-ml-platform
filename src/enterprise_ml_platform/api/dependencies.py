"""Dependency injection components for the API layer.

The serving layer keeps an in-process cache of loaded models. What it may put
in that cache is decided by configuration: with a registry configured, models
are resolved through the MLflow Model Registry by alias, so promoting a new
champion swaps the served model without redeploying the API. Only development
is allowed to fall back to the built-in demo model.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from redis.asyncio import Redis

from ..services.feature_store import (
    FeatureRegistry,
    FeatureStoreConfig,
    FeatureStoreService,
    InMemoryOfflineStore,
    OfflineStore,
    OnlineFeatureStore,
    ParquetOfflineStore,
)
from ..services.model_registry import MLflowModelRegistry, ModelRegistryError
from ..services.monitoring.collectors.metrics_collector import MetricsCollector
from .config import APISettings

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    from .schemas.models import ModelInfo

logger = structlog.get_logger(__name__)

DEMO_VERSION = "demo"


class ModelNotAvailableError(RuntimeError):
    """Raised when a model cannot be loaded for serving."""


@dataclass(frozen=True)
class LoadedModel:
    """A model held in the serving cache.

    Attributes:
        name: Name the model is served under.
        version: Registry version, or ``demo`` for the built-in stand-in.
        model: The object predictions are made with.
        source: ``registry`` or ``demo``.
        n_features: Feature count the model expects, when it advertises one.
    """

    name: str
    version: str
    model: Any
    source: str
    n_features: int | None = None

    def predict(self, data: Any) -> Any:
        """Run inference. Blocking; callers must keep this off the event loop."""
        return self.model.predict(data)


class ModelRegistry:
    """In-process cache of the models this server can serve."""

    def __init__(
        self,
        backend: MLflowModelRegistry | None = None,
        *,
        allow_demo_models: bool = True,
        default_alias: str = "champion",
    ) -> None:
        """Create a cache.

        Args:
            backend: Registry to resolve models through. Without one, only
                demo models can be loaded.
            allow_demo_models: Whether the built-in stand-in may be served.
            default_alias: Alias resolved when a caller names no version.
        """
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()
        self.backend = backend
        self.allow_demo_models = allow_demo_models
        self.default_alias = default_alias

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Return the names of every loaded model."""
        return list(self._models)

    # ------------------------------------------------------------------
    def get(self, name: str) -> LoadedModel | None:
        """Return a loaded model, or ``None`` if it is not in the cache."""
        return self._models.get(name)

    # ------------------------------------------------------------------
    def info(self, name: str) -> ModelInfo:
        """Return metadata for a loaded model.

        Raises:
            KeyError: If the model is not loaded.
        """
        from .schemas.models import ModelInfo

        loaded = self._models.get(name)
        if loaded is None:
            raise KeyError(f"Model '{name}' not found")
        return ModelInfo(
            name=loaded.name,
            version=loaded.version,
            description=f"served from {loaded.source}",
        )

    # ------------------------------------------------------------------
    def load(
        self,
        name: str,
        *,
        alias: str | None = None,
        version: str | None = None,
    ) -> ModelInfo:
        """Load a model into the serving cache.

        Args:
            name: Registered model name.
            alias: Deployment alias to resolve. Defaults to the configured one.
            version: Exact version, taking precedence over ``alias``.

        Raises:
            ModelNotAvailableError: If the model cannot be loaded.
        """
        from .schemas.models import ModelInfo

        loaded = (
            self._load_from_backend(name, alias, version)
            if self.backend is not None
            else self._load_demo(name)
        )
        with self._lock:
            self._models[name] = loaded
        logger.info(
            "model_loaded", model=name, version=loaded.version, source=loaded.source
        )
        return ModelInfo(
            name=loaded.name,
            version=loaded.version,
            description=f"served from {loaded.source}",
        )

    # ------------------------------------------------------------------
    def _load_from_backend(
        self, name: str, alias: str | None, version: str | None
    ) -> LoadedModel:
        assert self.backend is not None
        resolved_alias = alias or self.default_alias
        try:
            model = self.backend.load(name, alias=resolved_alias, version=version)
            info = (
                self.backend.get_version(name, version)
                if version
                else self.backend.get_by_alias(name, resolved_alias)
            )
        except ModelRegistryError as exc:
            raise ModelNotAvailableError(str(exc)) from exc
        return LoadedModel(
            name=name,
            version=info.version,
            model=model,
            source="registry",
            n_features=getattr(model, "n_features_in_", None),
        )

    # ------------------------------------------------------------------
    def _load_demo(self, name: str) -> LoadedModel:
        """Train the built-in stand-in model.

        This exists so the API can be explored without a registry. It is not a
        model store: the model is fitted here and now, and every restart
        produces a new one.
        """
        if not self.allow_demo_models:
            raise ModelNotAvailableError(
                f"no model registry configured, so model '{name}' cannot be served; "
                "set MLP_MODEL_REGISTRY_URI"
            )
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression

        data = load_iris()
        model = LogisticRegression(max_iter=200).fit(data.data, data.target)
        logger.warning("demo_model_loaded", model=name)
        return LoadedModel(
            name=name,
            version=DEMO_VERSION,
            model=model,
            source="demo",
            n_features=int(model.n_features_in_),
        )

    # ------------------------------------------------------------------
    def unload(self, name: str) -> None:
        """Remove a model from the cache if it is there."""
        with self._lock:
            self._models.pop(name, None)


def build_model_registry(settings: APISettings) -> ModelRegistry:
    """Build the serving cache described by ``settings``."""
    backend: MLflowModelRegistry | None = None
    if settings.model_registry_uri:
        backend = MLflowModelRegistry(
            tracking_uri=settings.model_registry_uri,
            registry_uri=settings.model_registry_uri,
        )
    return ModelRegistry(
        backend,
        allow_demo_models=settings.allow_demo_models,
        default_alias=settings.model_alias,
    )


def build_offline_store(
    settings: APISettings, metrics: MetricsCollector
) -> OfflineStore:
    """Build the offline store described by ``settings``."""
    if settings.feature_store_offline_path:
        return ParquetOfflineStore(Path(settings.feature_store_offline_path), metrics)
    logger.warning(
        "offline_feature_store_is_in_memory",
        detail="set MLP_FEATURE_STORE_OFFLINE_PATH for a store that survives restarts",
    )
    return InMemoryOfflineStore(metrics=metrics)


def build_feature_store(settings: APISettings) -> FeatureStoreService:
    """Build the feature store service described by ``settings``."""
    metrics = MetricsCollector()
    online = OnlineFeatureStore(
        Redis.from_url(settings.feature_store_redis_url), metrics=metrics
    )
    return FeatureStoreService(
        FeatureStoreConfig(redis_url=settings.feature_store_redis_url),
        FeatureRegistry(),
        online,
        build_offline_store(settings, metrics),
    )


# ----------------------------------------------------------------------
# FastAPI dependencies
# ----------------------------------------------------------------------
_registry: ModelRegistry | None = None
_feature_store: FeatureStoreService | None = None
_settings: APISettings | None = None


def configure(settings: APISettings) -> None:
    """Bind the shared dependencies to ``settings``.

    Called by the application factory so that each app gets components built
    from its own settings rather than from the ambient environment.
    """
    global _registry, _feature_store, _settings
    _settings = settings
    _registry = build_model_registry(settings)
    _feature_store = None  # rebuilt lazily; it opens a Redis connection


def get_settings() -> APISettings:
    """Return the settings the application was configured with."""
    global _settings
    if _settings is None:
        _settings = APISettings.from_env()
    return _settings


def get_registry() -> ModelRegistry:
    """Return the shared serving cache."""
    global _registry
    if _registry is None:
        _registry = build_model_registry(get_settings())
    return _registry


def get_feature_store() -> FeatureStoreService:
    """Return the shared :class:`FeatureStoreService`."""
    global _feature_store
    if _feature_store is None:
        _feature_store = build_feature_store(get_settings())
    return _feature_store


def get_logger() -> structlog.BoundLogger:
    """Return a structured logger bound to the current request."""
    logger: structlog.BoundLogger = structlog.get_logger()
    return logger
