"""Enforce simple resource quotas per tenant."""

from __future__ import annotations


class QuotaManager:
    def __init__(self) -> None:
        self.quotas: dict[str, dict[str, int | None]] = {}

    def set_quota(
        self, tenant: str, cpu: int | None = None, gpu: int | None = None
    ) -> None:
        self.quotas[tenant] = {"cpu": cpu, "gpu": gpu}

    def check(self, tenant: str, cpu: int = 0, gpu: int = 0) -> bool:
        quota = self.quotas.get(tenant)
        if not quota:
            return True
        return not (
            (quota["cpu"] is not None and cpu > quota["cpu"])
            or (quota["gpu"] is not None and gpu > quota["gpu"])
        )
