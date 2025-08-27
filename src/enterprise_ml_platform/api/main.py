"""Entry point for the FastAPI application."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware import AuthMiddleware, LoggingMiddleware, RateLimitMiddleware, TimeoutMiddleware
from .routers import health, models, predictions


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="Enterprise ML Platform API", version="1.0.0")

    # Routers
    app.include_router(predictions.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")

    # Middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(TimeoutMiddleware, timeout=30)
    app.add_middleware(RateLimitMiddleware, max_per_minute=120)
    app.add_middleware(AuthMiddleware, api_key="secret")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()
