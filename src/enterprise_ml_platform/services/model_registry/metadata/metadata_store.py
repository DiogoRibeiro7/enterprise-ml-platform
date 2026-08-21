"""In-memory metadata persistence for models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetadataStore:
    """Simple dictionary based metadata store."""

    store: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def save(self, name: str, version: str, metadata: dict[str, Any]) -> None:
        """Persist metadata for ``name`` and ``version``."""

        self.store[(name, version)] = metadata

    def get(self, name: str, version: str) -> dict[str, Any]:
        """Retrieve metadata for a model version."""

        return self.store.get((name, version), {})
