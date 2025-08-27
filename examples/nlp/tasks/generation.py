"""Text generation task implementation."""
from __future__ import annotations

from typing import List

from ..models.transformer_models import TransformerModels


class TextGenerator:
    """Generate text continuations using causal language models."""

    def __init__(self, model_name: str | None = None, max_new_tokens: int = 20) -> None:
        self.models = TransformerModels()
        self.pipeline = self.models.get_generation_pipeline(model_name)
        self.max_new_tokens = max_new_tokens

    def predict(self, prompts: List[str]) -> List[str]:
        """Generate text for each prompt."""
        outputs = self.pipeline(prompts, max_new_tokens=self.max_new_tokens)
        return [o[0]["generated_text"] for o in outputs]


__all__ = ["TextGenerator"]
