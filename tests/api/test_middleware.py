"""Regression tests for the middleware rejection paths.

Starlette runs ``BaseHTTPMiddleware`` outside the routing layer, where
FastAPI's exception handlers never see a raised ``HTTPException``. Every
middleware here used to raise one, so a rejected request escaped as an
unhandled error rather than the status code it advertised -- meaning API-key
authentication never actually returned 401.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from enterprise_ml_platform.api.middleware import (
    AuthMiddleware,
    RateLimitMiddleware,
    TimeoutMiddleware,
)


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        return {"pong": True}

    @app.get("/slow")
    async def slow() -> dict:
        await asyncio.sleep(1.0)
        return {"pong": True}

    return app


# ----------------------------------------------------------------------
def test_missing_api_key_returns_401() -> None:
    app = _app()
    app.add_middleware(AuthMiddleware, api_key="the-key")

    response = TestClient(app).get("/ping")

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_wrong_api_key_returns_401() -> None:
    app = _app()
    app.add_middleware(AuthMiddleware, api_key="the-key")

    response = TestClient(app).get("/ping", headers={"X-API-Key": "not-the-key"})

    assert response.status_code == 401


def test_correct_api_key_passes_through() -> None:
    app = _app()
    app.add_middleware(AuthMiddleware, api_key="the-key")

    response = TestClient(app).get("/ping", headers={"X-API-Key": "the-key"})

    assert response.status_code == 200
    assert response.json() == {"pong": True}


def test_no_configured_key_disables_authentication() -> None:
    app = _app()
    app.add_middleware(AuthMiddleware, api_key=None)

    assert TestClient(app).get("/ping").status_code == 200


# ----------------------------------------------------------------------
def test_exceeding_the_rate_limit_returns_429() -> None:
    app = _app()
    app.add_middleware(RateLimitMiddleware, max_per_minute=2)
    client = TestClient(app)

    statuses = [client.get("/ping").status_code for _ in range(4)]

    assert statuses[:2] == [200, 200]
    assert statuses[2:] == [429, 429]


def test_rate_limited_response_carries_a_reason() -> None:
    app = _app()
    app.add_middleware(RateLimitMiddleware, max_per_minute=1)
    client = TestClient(app)
    client.get("/ping")

    response = client.get("/ping")

    assert response.json() == {"detail": "Rate limit exceeded"}


# ----------------------------------------------------------------------
def test_slow_request_returns_504() -> None:
    app = _app()
    app.add_middleware(TimeoutMiddleware, timeout=0.05)

    response = TestClient(app).get("/slow")

    assert response.status_code == 504
    assert response.json() == {"detail": "Request timeout"}
