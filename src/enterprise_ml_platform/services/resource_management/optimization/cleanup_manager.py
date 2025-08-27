from __future__ import annotations

"""Automated cleanup of unused resources."""


class CleanupManager:
    def __init__(self) -> None:
        self.cleaned: list[str] = []

    def cleanup(self, resource_id: str) -> None:
        self.cleaned.append(resource_id)
