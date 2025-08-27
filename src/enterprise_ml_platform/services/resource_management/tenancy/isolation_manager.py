from __future__ import annotations

"""Namespace isolation stub for multi-tenancy."""


class IsolationManager:
    def __init__(self) -> None:
        self.namespaces: dict[str, dict[str, str]] = {}

    def assign_namespace(self, tenant: str, namespace: str) -> None:
        self.namespaces[tenant] = {"namespace": namespace}

    def get_namespace(self, tenant: str) -> str | None:
        ns = self.namespaces.get(tenant)
        if ns:
            return ns["namespace"]
        return None
