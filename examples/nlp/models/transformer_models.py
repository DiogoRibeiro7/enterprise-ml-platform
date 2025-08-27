"""Wrappers around HuggingFace transformer models for multiple NLP tasks."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, pipeline


class TransformerModels:
    """Utility class to load transformer pipelines for various tasks.

    Defaults to tiny models for fast initialization in example environments.
    """

    DEFAULT_CLASSIFICATION_MODEL = "hf-internal-testing/tiny-random-bert"
    DEFAULT_NER_MODEL = "hf-internal-testing/tiny-bert-for-token-classification"
    DEFAULT_SENTIMENT_MODEL = "hf-internal-testing/tiny-random-bert"
    DEFAULT_GENERATION_MODEL = "hf-internal-testing/tiny-random-gpt2"

    def __init__(self) -> None:
        pass

    @lru_cache(maxsize=None)
    def get_classification_pipeline(self, model_name: Optional[str] = None) -> Any:
        model_name = model_name or self.DEFAULT_CLASSIFICATION_MODEL
        return pipeline("text-classification", model=model_name, tokenizer=model_name)

    @lru_cache(maxsize=None)
    def get_ner_pipeline(self, model_name: Optional[str] = None) -> Any:
        model_name = model_name or self.DEFAULT_NER_MODEL
        return pipeline("ner", model=model_name, tokenizer=model_name, grouped_entities=True)

    @lru_cache(maxsize=None)
    def get_sentiment_pipeline(self, model_name: Optional[str] = None) -> Any:
        model_name = model_name or self.DEFAULT_SENTIMENT_MODEL
        return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    @lru_cache(maxsize=None)
    def get_generation_pipeline(self, model_name: Optional[str] = None) -> Any:
        model_name = model_name or self.DEFAULT_GENERATION_MODEL
        return pipeline("text-generation", model=model_name, tokenizer=model_name)


__all__ = ["TransformerModels"]
