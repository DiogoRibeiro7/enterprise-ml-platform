"""Minimal in‑memory annotation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class AnnotationTools:
    """Stores annotations for images and supports simple queries."""

    annotations: List[Dict] = field(default_factory=list)

    def add(self, image_id: str, label: str, bbox: Tuple[int, int, int, int] | None = None) -> None:
        self.annotations.append({"image_id": image_id, "label": label, "bbox": bbox})

    def list(self) -> List[Dict]:
        return list(self.annotations)
