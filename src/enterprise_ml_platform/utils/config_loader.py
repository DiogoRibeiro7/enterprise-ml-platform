"""YAML configuration loader with environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Sequence

import yaml

from enterprise_ml_platform.core.exceptions import ConfigurationError


def load_config(
    path: str | Path,
    *,
    env_prefix: str = "EMPLATFORM_",
    required_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load a YAML configuration file and apply environment overrides.

    Environment variables beginning with ``env_prefix`` override nested
    configuration values. Nested keys should be separated by double
    underscores. For example ``EMPLATFORM_database__host=localhost`` would
    override ``{"database": {"host": "localhost"}}``.

    Args:
        path: Path to the YAML configuration file.
        env_prefix: Prefix used to identify environment variables for override.
        required_keys: Keys that must be present in the resulting configuration.

    Returns:
        Loaded and merged configuration dictionary.

    Raises:
        ConfigurationError: If the file cannot be read or validation fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
        config: Dict[str, Any] = yaml.safe_load(raw) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("Failed to load configuration") from exc

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(env_prefix):
            continue
        key_path = env_key[len(env_prefix) :].lower().split("__")
        _apply_env_override(config, key_path, env_value)

    if required_keys:
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {', '.join(missing)}"
            )

    return config


def _apply_env_override(
    config: MutableMapping[str, Any], key_path: Sequence[str], value: str
) -> None:
    """Apply a single environment override to the configuration mapping.

    Args:
        config: Configuration dictionary to update.
        key_path: Sequence representing nested keys.
        value: Environment variable value as a string.
    """
    current: MutableMapping[str, Any] = config
    for key in key_path[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]  # type: ignore[assignment]

    current[key_path[-1]] = yaml.safe_load(value)
