from __future__ import annotations

"""In-memory metadata persistence for models."""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any


@dataclass
class MetadataStore:
    """Simple dictionary based metadata store."""

    store: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)

    def save(self, name: str, version: str, metadata: Dict[str, Any]) -> None:
        """Persist metadata for ``name`` and ``version``."""

        self.store[(name, version)] = metadata

    def get(self, name: str, version: str) -> Dict[str, Any]:
        """Retrieve metadata for a model version."""

        return self.store.get((name, version), {})
