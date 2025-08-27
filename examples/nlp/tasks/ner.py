"""Named entity recognition task implementation."""
from __future__ import annotations

from typing import List

from ..models.transformer_models import TransformerModels


class NamedEntityRecognizer:
    """Identify entities in text using transformer-based NER."""

    def __init__(self, model_name: str | None = None) -> None:
        self.models = TransformerModels()
        self.pipeline = self.models.get_ner_pipeline(model_name)

    def predict(self, texts: List[str]) -> List[List[dict]]:
        """Return entities for each text input."""
        return [self.pipeline(t) for t in texts]


__all__ = ["NamedEntityRecognizer"]
