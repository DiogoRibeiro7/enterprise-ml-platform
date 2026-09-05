"""Tests for API configuration.

The application used to be built with ``api_key="secret"`` and
``allow_origins=["*"]`` written into the source, so the credential shipped in
the repository and every origin could call the API.
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from enterprise_ml_platform.api.config import APISettings
from enterprise_ml_platform.api.main import create_app
from enterprise_ml_platform.core.exceptions import ConfigurationError


def _production(
    registry_uri: str = "sqlite:///registry.db", **overrides
) -> APISettings:
    """Settings that satisfy every deployment guardrail, plus overrides.

    ``registry_uri`` defaults to a relative path, which is fine for settings
    that are only inspected. Anything that *builds* the app must pass a
    tmp_path URI: constructing the app opens the registry, and a relative URI
    would create the database in the working directory.
    """
    defaults = {
        "environment": "production",
        "api_key": "k",
        "cors_origins": (),
        "allow_demo_models": False,
        "model_registry_uri": registry_uri,
    }
    return APISettings(**{**defaults, **overrides})


@pytest.fixture
def registry_uri(tmp_path: pathlib.Path) -> str:
    """A throwaway model registry for tests that build the application."""
    return f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"


API_MAIN = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "enterprise_ml_platform"
    / "api"
    / "main.py"
)


# ----------------------------------------------------------------------
# Nothing sensitive in the source
# ----------------------------------------------------------------------
def test_no_credentials_are_hardcoded_in_the_app_factory() -> None:
    source = API_MAIN.read_text(encoding="utf-8")
    assert 'api_key="secret"' not in source
    assert 'allow_origins=["*"]' not in source


# ----------------------------------------------------------------------
# Reading the environment
# ----------------------------------------------------------------------
def test_settings_read_from_environment() -> None:
    settings = APISettings.from_env(
        {
            "MLP_ENVIRONMENT": "production",
            "MLP_API_KEY": "from-the-environment",
            "MLP_CORS_ORIGINS": "https://app.example.com, https://admin.example.com",
            "MLP_RATE_LIMIT_PER_MINUTE": "30",
            "MLP_REQUEST_TIMEOUT_SECONDS": "5",
            "MLP_PORT": "9000",
            "MLP_MODEL_REGISTRY_URI": "sqlite:///registry.db",
            "MLP_ALLOW_DEMO_MODELS": "false",
            "MLP_MAX_BATCH_SIZE": "64",
            "MLP_DRIFT_WINDOW_SIZE": "128",
            "MLP_DRIFT_MIN_SAMPLES": "32",
            "MLP_DRIFT_THRESHOLD": "0.35",
        }
    )

    assert settings.api_key == "from-the-environment"
    assert settings.cors_origins == (
        "https://app.example.com",
        "https://admin.example.com",
    )
    assert settings.rate_limit_per_minute == 30
    assert settings.request_timeout_seconds == 5
    assert settings.port == 9000
    assert settings.model_registry_uri == "sqlite:///registry.db"
    assert settings.allow_demo_models is False
    assert settings.max_batch_size == 64
    assert settings.drift_window_size == 128
    assert settings.drift_min_samples == 32
    assert settings.drift_threshold == pytest.approx(0.35)
    assert not settings.is_development


def test_defaults_are_development_and_unauthenticated() -> None:
    settings = APISettings.from_env({})

    assert settings.is_development
    assert settings.api_key is None
    assert settings.cors_origins == ()


def test_blank_environment_values_fall_back_to_defaults() -> None:
    settings = APISettings.from_env({"MLP_API_KEY": "   ", "MLP_CORS_ORIGINS": " "})

    assert settings.api_key is None
    assert settings.cors_origins == ()


def test_non_integer_setting_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="RATE_LIMIT_PER_MINUTE"):
        APISettings.from_env({"MLP_RATE_LIMIT_PER_MINUTE": "many"})


# ----------------------------------------------------------------------
# Deployment guardrails
# ----------------------------------------------------------------------
def test_deployment_without_an_api_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="MLP_API_KEY must be set"):
        APISettings.from_env({"MLP_ENVIRONMENT": "production"})


def test_deployment_with_wildcard_cors_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="cannot be '\\*'"):
        APISettings.from_env(
            {
                "MLP_ENVIRONMENT": "production",
                "MLP_API_KEY": "k",
                "MLP_CORS_ORIGINS": "*",
            }
        )


def test_deployment_serving_demo_models_is_rejected() -> None:
    """A deployment must serve versioned artifacts, not a model fitted at load."""
    with pytest.raises(ConfigurationError, match="ALLOW_DEMO_MODELS"):
        APISettings.from_env(
            {
                "MLP_ENVIRONMENT": "production",
                "MLP_API_KEY": "k",
                "MLP_ALLOW_DEMO_MODELS": "true",
            }
        )


def test_deployment_without_a_model_registry_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="MODEL_REGISTRY_URI"):
        APISettings.from_env(
            {
                "MLP_ENVIRONMENT": "production",
                "MLP_API_KEY": "k",
                "MLP_ALLOW_DEMO_MODELS": "false",
            }
        )


def test_demo_models_are_allowed_by_default_only_in_development() -> None:
    assert APISettings.from_env({}).allow_demo_models is True


def test_non_boolean_setting_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="ALLOW_DEMO_MODELS"):
        APISettings.from_env({"MLP_ALLOW_DEMO_MODELS": "perhaps"})


def test_zero_batch_size_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="MAX_BATCH_SIZE"):
        APISettings.from_env({"MLP_MAX_BATCH_SIZE": "0"})


@pytest.mark.parametrize(
    ("source", "setting"),
    [
        ({"MLP_DRIFT_WINDOW_SIZE": "1"}, "DRIFT_WINDOW_SIZE"),
        (
            {"MLP_DRIFT_WINDOW_SIZE": "10", "MLP_DRIFT_MIN_SAMPLES": "11"},
            "DRIFT_MIN_SAMPLES",
        ),
        ({"MLP_DRIFT_THRESHOLD": "0"}, "DRIFT_THRESHOLD"),
        ({"MLP_DRIFT_THRESHOLD": "nan"}, "DRIFT_THRESHOLD"),
        ({"MLP_DRIFT_THRESHOLD": "inf"}, "DRIFT_THRESHOLD"),
    ],
)
def test_invalid_drift_settings_are_rejected(
    source: dict[str, str], setting: str
) -> None:
    with pytest.raises(ConfigurationError, match=setting):
        APISettings.from_env(source)


def test_development_may_run_open() -> None:
    """Local work must not require credentials to be provisioned first."""
    settings = APISettings.from_env({"MLP_ENVIRONMENT": "development"})
    assert settings.api_key is None


# ----------------------------------------------------------------------
# The app honours the settings
# ----------------------------------------------------------------------
def test_app_enforces_the_configured_api_key(registry_uri: str) -> None:
    app = create_app(_production(registry_uri, api_key="configured-key"))
    client = TestClient(app)

    assert client.get("/api/v1/health").status_code == 401
    assert (
        client.get("/api/v1/health", headers={"X-API-Key": "secret"}).status_code == 401
    )
    assert (
        client.get(
            "/api/v1/health", headers={"X-API-Key": "configured-key"}
        ).status_code
        == 200
    )


def test_app_restricts_cors_to_configured_origins(registry_uri: str) -> None:
    app = create_app(
        _production(registry_uri, cors_origins=("https://app.example.com",))
    )
    client = TestClient(app)
    headers = {"X-API-Key": "k"}

    allowed = client.get(
        "/api/v1/health", headers={**headers, "Origin": "https://app.example.com"}
    )
    denied = client.get(
        "/api/v1/health", headers={**headers, "Origin": "https://evil.example.com"}
    )

    assert (
        allowed.headers.get("access-control-allow-origin") == "https://app.example.com"
    )
    assert "access-control-allow-origin" not in denied.headers
