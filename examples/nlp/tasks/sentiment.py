"""Sentiment analysis task implementation."""
from __future__ import annotations

from typing import List

from ..models.transformer_models import TransformerModels


class SentimentAnalyzer:
    """Predict sentiment polarity for text inputs."""

    def __init__(self, model_name: str | None = None) -> None:
        self.models = TransformerModels()
        self.pipeline = self.models.get_sentiment_pipeline(model_name)

    def predict(self, texts: List[str]) -> List[dict]:
        """Return sentiment predictions."""
        return self.pipeline(texts)


__all__ = ["SentimentAnalyzer"]
