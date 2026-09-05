"""Runtime configuration for the API layer.

Everything that differs between a laptop and a deployed environment is read
from the environment here, so that no credential or origin policy is baked
into the source.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..core.exceptions import ConfigurationError

ENV_PREFIX = "MLP_"
DEVELOPMENT = "development"


def _env(
    source: Mapping[str, str], name: str, default: str | None = None
) -> str | None:
    value = source.get(f"{ENV_PREFIX}{name}")
    if value is None or not value.strip():
        return default
    return value.strip()


def _int_env(source: Mapping[str, str], name: str, default: int) -> int:
    raw = _env(source, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"{ENV_PREFIX}{name} must be an integer, got {raw!r}"
        ) from exc


def _float_env(source: Mapping[str, str], name: str, default: float) -> float:
    raw = _env(source, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"{ENV_PREFIX}{name} must be a number, got {raw!r}"
        ) from exc


TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _bool_env(source: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _env(source, name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise ConfigurationError(f"{ENV_PREFIX}{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class APISettings:
    """Settings governing how the API server behaves.

    Attributes:
        environment: Deployment environment name. Anything other than
            ``development`` is treated as a real deployment and held to
            stricter rules.
        api_key: Key clients must present in ``X-API-Key``. ``None`` disables
            authentication, which is only permitted in development.
        cors_origins: Exact origins allowed to call the API from a browser.
        rate_limit_per_minute: Requests allowed per client per minute.
        request_timeout_seconds: Seconds before an in-flight request is aborted.
        drift_window_size: Maximum live rows retained for each served version.
        drift_min_samples: Rows required before drift is evaluated.
        drift_threshold: PSI score that marks a feature as drifted.
        host: Interface the server binds to.
        port: Port the server binds to.
    """

    environment: str = DEVELOPMENT
    api_key: str | None = None
    cors_origins: Sequence[str] = field(default_factory=tuple)
    rate_limit_per_minute: int = 120
    request_timeout_seconds: int = 30
    host: str = "127.0.0.1"
    port: int = 8000
    model_registry_uri: str | None = None
    model_alias: str = "champion"
    allow_demo_models: bool = True
    max_batch_size: int = 1000
    feature_store_redis_url: str = "redis://localhost:6379/0"
    feature_store_offline_path: str | None = None
    drift_window_size: int = 256
    drift_min_samples: int = 50
    drift_threshold: float = 0.2

    @property
    def is_development(self) -> bool:
        """Return ``True`` when running outside a real deployment."""
        return self.environment.lower() == DEVELOPMENT

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ConfigurationError(
                f"{ENV_PREFIX}MAX_BATCH_SIZE must be at least 1, "
                f"got {self.max_batch_size}"
            )
        if self.drift_window_size < 2:
            raise ConfigurationError(
                f"{ENV_PREFIX}DRIFT_WINDOW_SIZE must be at least 2, "
                f"got {self.drift_window_size}"
            )
        if not 2 <= self.drift_min_samples <= self.drift_window_size:
            raise ConfigurationError(
                f"{ENV_PREFIX}DRIFT_MIN_SAMPLES must be between 2 and "
                f"{ENV_PREFIX}DRIFT_WINDOW_SIZE, got {self.drift_min_samples}"
            )
        if not math.isfinite(self.drift_threshold) or self.drift_threshold <= 0:
            raise ConfigurationError(
                f"{ENV_PREFIX}DRIFT_THRESHOLD must be positive, "
                f"got {self.drift_threshold}"
            )
        if self.is_development:
            return
        # Rules that only apply to real deployments.
        if not self.api_key:
            raise ConfigurationError(
                f"{ENV_PREFIX}API_KEY must be set when "
                f"{ENV_PREFIX}ENVIRONMENT is {self.environment!r}"
            )
        if "*" in self.cors_origins:
            raise ConfigurationError(
                f"{ENV_PREFIX}CORS_ORIGINS cannot be '*' when "
                f"{ENV_PREFIX}ENVIRONMENT is {self.environment!r}; "
                "list the allowed origins explicitly"
            )
        if self.allow_demo_models:
            raise ConfigurationError(
                f"{ENV_PREFIX}ALLOW_DEMO_MODELS cannot be enabled when "
                f"{ENV_PREFIX}ENVIRONMENT is {self.environment!r}; "
                "a deployment must serve models from the registry"
            )
        if not self.model_registry_uri:
            raise ConfigurationError(
                f"{ENV_PREFIX}MODEL_REGISTRY_URI must be set when "
                f"{ENV_PREFIX}ENVIRONMENT is {self.environment!r}"
            )

    @classmethod
    def from_env(cls, source: Mapping[str, str] | None = None) -> APISettings:
        """Build settings from environment variables.

        Args:
            source: Mapping to read from. Defaults to ``os.environ``.

        Raises:
            ConfigurationError: If a deployed environment is misconfigured.
        """
        source = os.environ if source is None else source
        raw_origins = _env(source, "CORS_ORIGINS", "")
        origins = tuple(o.strip() for o in (raw_origins or "").split(",") if o.strip())
        environment = _env(source, "ENVIRONMENT", DEVELOPMENT) or DEVELOPMENT
        return cls(
            environment=environment,
            api_key=_env(source, "API_KEY"),
            cors_origins=origins,
            rate_limit_per_minute=_int_env(source, "RATE_LIMIT_PER_MINUTE", 120),
            request_timeout_seconds=_int_env(source, "REQUEST_TIMEOUT_SECONDS", 30),
            host=_env(source, "HOST", "127.0.0.1") or "127.0.0.1",
            port=_int_env(source, "PORT", 8000),
            model_registry_uri=_env(source, "MODEL_REGISTRY_URI"),
            model_alias=_env(source, "MODEL_ALIAS", "champion") or "champion",
            allow_demo_models=_bool_env(
                source, "ALLOW_DEMO_MODELS", environment.lower() == DEVELOPMENT
            ),
            max_batch_size=_int_env(source, "MAX_BATCH_SIZE", 1000),
            feature_store_redis_url=(
                _env(source, "FEATURE_STORE_REDIS_URL", "redis://localhost:6379/0")
                or "redis://localhost:6379/0"
            ),
            feature_store_offline_path=_env(source, "FEATURE_STORE_OFFLINE_PATH"),
            drift_window_size=_int_env(source, "DRIFT_WINDOW_SIZE", 256),
            drift_min_samples=_int_env(source, "DRIFT_MIN_SAMPLES", 50),
            drift_threshold=_float_env(source, "DRIFT_THRESHOLD", 0.2),
        )
