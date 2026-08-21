"""Search utilities for models in the registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSearch:
    """Very small metadata based search engine."""

    def search(
        self,
        metadata: dict[tuple[str, str], dict[str, str]],
        query: str,
    ) -> list[tuple[str, str]]:
        """Return model versions whose metadata contains ``query``."""

        results: list[tuple[str, str]] = []
        for key, meta in metadata.items():
            haystack = " ".join(f"{k}:{v}" for k, v in meta.items()).lower()
            if query.lower() in haystack:
                results.append(key)
        return results
