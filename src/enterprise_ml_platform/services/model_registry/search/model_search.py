from __future__ import annotations

"""Search utilities for models in the registry."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ModelSearch:
    """Very small metadata based search engine."""

    def search(
        self,
        metadata: Dict[Tuple[str, str], Dict[str, str]],
        query: str,
    ) -> List[Tuple[str, str]]:
        """Return model versions whose metadata contains ``query``."""

        results: List[Tuple[str, str]] = []
        for key, meta in metadata.items():
            haystack = " ".join(f"{k}:{v}" for k, v in meta.items()).lower()
            if query.lower() in haystack:
                results.append(key)
        return results
