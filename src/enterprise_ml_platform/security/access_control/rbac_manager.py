"""Role Based Access Control (RBAC) utilities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RBACManager:
    """Simple in‑memory RBAC manager.

    The goal of this class is to provide a very small façade that mimics a
    persistent RBAC backend.  Roles map to permissions and users are
    associated with a set of roles.  The class is deliberately tiny but is
    feature complete enough for unit tests and examples.
    """

    role_permissions: dict[str, set[str]] = field(default_factory=dict)
    user_roles: dict[str, set[str]] = field(default_factory=dict)

    def add_role(self, role: str, permissions: list[str]) -> None:
        self.role_permissions.setdefault(role, set()).update(permissions)

    def assign_role(self, user: str, role: str) -> None:
        self.user_roles.setdefault(user, set()).add(role)

    def check_access(self, user: str, permission: str) -> bool:
        """Return ``True`` if ``user`` has ``permission``."""

        roles = self.user_roles.get(user, set())
        for role in roles:
            if permission in self.role_permissions.get(role, set()):
                return True
        return False

    def revoke_role(self, user: str, role: str) -> None:
        self.user_roles.get(user, set()).discard(role)
