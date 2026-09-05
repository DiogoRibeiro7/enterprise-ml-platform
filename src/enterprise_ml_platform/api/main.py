"""Entry point for the FastAPI application."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import dependencies
from .config import APISettings
from .middleware import (
    AuthMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    TimeoutMiddleware,
)
from .routers import ab_testing, feature_store, health, models, predictions

logger = structlog.get_logger(__name__)


async def _request_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    """Return serialisable validation details without echoing request values."""
    if not isinstance(exc, RequestValidationError):  # pragma: no cover - framework API
        raise exc
    details = [
        {key: value for key, value in error.items() if key != "input"}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(details)})


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Explicit settings. Defaults to reading the environment.
    """

    settings = settings or APISettings.from_env()

    dependencies.configure(settings)

    app = FastAPI(title="Enterprise ML Platform API", version="1.0.0")
    app.state.settings = settings
    app.add_exception_handler(RequestValidationError, _request_validation_error)

    # Routers
    app.include_router(predictions.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(feature_store.router, prefix="/api/v1")
    app.include_router(ab_testing.router, prefix="/api/v1")

    # Middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(TimeoutMiddleware, timeout=settings.request_timeout_seconds)
    app.add_middleware(
        RateLimitMiddleware, max_per_minute=settings.rate_limit_per_minute
    )
    app.add_middleware(AuthMiddleware, api_key=settings.api_key)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    if not settings.api_key:
        logger.warning(
            "api_authentication_disabled",
            environment=settings.environment,
            detail="MLP_API_KEY is unset; every request is accepted",
        )

    return app


def start_server() -> None:
    """Run the API with uvicorn. Backs the ``mlp-server`` console script."""

    import uvicorn

    settings = APISettings.from_env()
    uvicorn.run(
        "enterprise_ml_platform.api.main:app",
        host=settings.host,
        port=settings.port,
        factory=False,
    )


app = create_app()
