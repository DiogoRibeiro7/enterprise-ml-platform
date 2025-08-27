from __future__ import annotations

"""Model exporting utilities."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelExporter:
    """Convert models into different serialisation formats.

    The exporter only simulates conversion by returning a string that indicates
    where a converted artifact would be stored.
    """

    def export(self, model: Any, fmt: str) -> str:
        """"Export`` model to format ``fmt`` and return a fake path."""

        return f"/tmp/exported_model.{fmt}"
