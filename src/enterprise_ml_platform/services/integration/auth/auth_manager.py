"""Authentication and authorization helpers for the API gateway."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AuthManager:
    """Manages API keys and bearer tokens.

    The implementation is intentionally lightweight.  Real deployments would
    integrate with OAuth providers, LDAP, or other identity services.  The
    helper covers enough surface area for tests and documentation examples.
    """

    api_keys: dict[str, str] = field(default_factory=dict)
    tokens: dict[str, tuple[str, float]] = field(default_factory=dict)

    def register_api_key(self, user: str, key: str) -> None:
        self.api_keys[key] = user

    def issue_token(self, user: str, token: str, ttl: int = 3600) -> str:
        self.tokens[token] = (user, time.time() + ttl)
        return token

    def authenticate(self, headers: dict[str, str]) -> str | None:
        """Return the user identified by ``headers`` if any."""

        token = headers.get("Authorization")
        if token and token in self.tokens:
            user, expiry = self.tokens[token]
            if expiry > time.time():
                return user
        api_key = headers.get("X-API-Key")
        if api_key and api_key in self.api_keys:
            return self.api_keys[api_key]
        return None

    def authorize(self, user: str, action: str) -> bool:
        """Check whether ``user`` may perform ``action``.

        For demonstration purposes every authenticated user is authorised.  The
        method exists so that RBAC/ABAC logic can be slotted in later.
        """

        return True
