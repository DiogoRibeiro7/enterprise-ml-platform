"""Text classification task implementation."""
from __future__ import annotations

from typing import List

from ..models.transformer_models import TransformerModels


class TextClassifier:
    """Perform multi-class or multi-label classification using transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.models = TransformerModels()
        self.pipeline = self.models.get_classification_pipeline(model_name)

    def predict(self, texts: List[str]) -> List[dict]:
        """Return model predictions for a list of texts."""
        return self.pipeline(texts)


__all__ = ["TextClassifier"]
