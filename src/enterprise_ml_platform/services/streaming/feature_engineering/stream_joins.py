"""Join streaming records with static tables or feature store."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class StreamJoiner:
    """Perform simple key-based joins for enrichment."""

    def __init__(self, key: str, table: Mapping[str, dict[str, Any]]) -> None:
        self.key = key
        self.table = table

    async def join(self, features: dict[str, Any]) -> dict[str, Any]:
        join_value = features.get(self.key)
        if join_value is None:
            return features
        extra = self.table.get(str(join_value))
        if not extra:
            return features
        enriched = features.copy()
        enriched.update(extra)
        return enriched
