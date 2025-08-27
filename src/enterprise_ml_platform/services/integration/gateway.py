"""Main API gateway orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .auth.auth_manager import AuthManager
from .routing.request_router import RequestRouter
from .rate_limiting.rate_limiter import RateLimiter
from .transformation.data_transformer import DataTransformer
from .caching.cache_manager import CacheManager
from .connectors.enterprise_connectors import EnterpriseConnectors
from .monitoring.gateway_monitor import GatewayMonitor
from .policies.policy_engine import PolicyEngine


@dataclass
class APIGateway:
    """Coordinates authentication, routing and system connectivity.

    The gateway is intentionally minimal but captures the core concepts needed
    for an enterprise integration layer.  It supports HTTP style requests, as
    well as GraphQL and WebSocket style interactions via convenience methods.
    """

    auth: AuthManager = field(default_factory=AuthManager)
    router: RequestRouter = field(default_factory=RequestRouter)
    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(rate=100, per=60))
    transformer: DataTransformer = field(default_factory=DataTransformer)
    cache: CacheManager = field(default_factory=CacheManager)
    connectors: EnterpriseConnectors = field(default_factory=EnterpriseConnectors)
    monitor: GatewayMonitor = field(default_factory=GatewayMonitor)
    policies: PolicyEngine = field(default_factory=PolicyEngine)

    def handle_http(self, request: Dict[str, Any]) -> Any:
        """Process a generic HTTP request represented as a mapping."""

        user = self.auth.authenticate(request.get("headers", {}))
        if not user or not self.auth.authorize(user, request.get("method", "GET")):
            self.monitor.record_error("auth")
            raise PermissionError("unauthorized")

        if not self.limiter.allow(user):
            self.monitor.record_error("rate_limit")
            raise RuntimeError("rate limit exceeded")

        if not self.policies.evaluate(request):
            self.monitor.record_error("policy")
            raise PermissionError("policy check failed")

        request = self.transformer.transform_request(request)
        cache_key = request.get("cache_key")
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.monitor.record_request("cache")
                return cached

        route = self.router.route(request)
        self.monitor.record_request(route)
        response = self.connectors.call(route, request)
        response = self.transformer.transform_response(response)

        if cache_key:
            self.cache.set(cache_key, response)
        return response

    # Convenience wrappers -------------------------------------------
    def handle_graphql(self, query: str, **kwargs: Any) -> Any:
        request = {"path": "/graphql", "body": query, **kwargs}
        return self.handle_http(request)

    def handle_websocket(self, message: Dict[str, Any]) -> Any:
        message.setdefault("path", "/ws")
        return self.handle_http(message)
