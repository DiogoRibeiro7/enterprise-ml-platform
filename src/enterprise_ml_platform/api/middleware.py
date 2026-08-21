"""Custom middleware implementations for the FastAPI application."""

from __future__ import annotations

import asyncio
import time
import uuid
from hmac import compare_digest

import structlog
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "Request duration", ["method", "path"]
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Attach correlation IDs and log requests/responses."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        logger = structlog.get_logger().bind(correlation_id=correlation_id)
        logger.info("request.start", method=request.method, path=request.url.path)
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
        REQUEST_COUNT.labels(
            request.method, request.url.path, response.status_code
        ).inc()
        logger.info(
            "request.end",
            status_code=response.status_code,
            duration=duration,
        )
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Very small API-key based authentication middleware."""

    def __init__(self, app: ASGIApp, api_key: str | None = None) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        key = request.headers.get("X-API-Key")
        if self.api_key and not compare_digest(key or "", self.api_key):
            # Middleware runs outside the routing layer, so a raised
            # HTTPException is never turned into a response -- it escapes as an
            # unhandled error. Rejections have to be returned, not raised.
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Naive in-memory rate limiter."""

    def __init__(self, app: ASGIApp, max_per_minute: int = 60) -> None:
        super().__init__(app)
        self.max_per_minute = max_per_minute
        self.calls: dict[str, tuple[int, int]] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        now = int(time.time())
        window = now // 60
        ident = request.client.host if request.client else "global"
        count, last_window = self.calls.get(ident, (0, window))
        if last_window != window:
            count = 0
        count += 1
        self.calls[ident] = (count, window)
        if count > self.max_per_minute:
            # Returned, not raised: see AuthMiddleware.dispatch.
            return JSONResponse(
                status_code=429, content={"detail": "Rate limit exceeded"}
            )
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Abort requests taking longer than the configured timeout."""

    def __init__(self, app: ASGIApp, timeout: float = 30) -> None:
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except TimeoutError:
            # Returned, not raised: see AuthMiddleware.dispatch.
            return JSONResponse(status_code=504, content={"detail": "Request timeout"})
