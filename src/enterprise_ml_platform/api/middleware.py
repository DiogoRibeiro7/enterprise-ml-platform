"""Custom middleware implementations for the FastAPI application."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable

import structlog
from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "Request duration", ["method", "path"]
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Attach correlation IDs and log requests/responses."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        logger = structlog.get_logger().bind(correlation_id=correlation_id)
        logger.info("request.start", method=request.method, path=request.url.path)
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
        REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
        logger.info(
            "request.end",
            status_code=response.status_code,
            duration=duration,
        )
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Very small API-key based authentication middleware."""

    def __init__(self, app, api_key: str | None = None) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        key = request.headers.get("X-API-Key")
        if self.api_key and key != self.api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Naive in-memory rate limiter."""

    def __init__(self, app, max_per_minute: int = 60) -> None:
        super().__init__(app)
        self.max_per_minute = max_per_minute
        self.calls: dict[str, tuple[int, int]] = {}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        now = int(time.time())
        window = now // 60
        ident = request.client.host if request.client else "global"
        count, last_window = self.calls.get(ident, (0, window))
        if last_window != window:
            count = 0
        count += 1
        self.calls[ident] = (count, window)
        if count > self.max_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Abort requests taking longer than the configured timeout."""

    def __init__(self, app, timeout: int = 30) -> None:
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError as exc:  # pragma: no cover - network issues
            raise HTTPException(status_code=504, detail="Request timeout") from exc
