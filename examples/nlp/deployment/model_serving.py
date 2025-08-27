"""Simple FastAPI service exposing NLP tasks for inference."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import FastAPI

from ..preprocessing.text_processor import TextProcessor
from ..tasks.classification import TextClassifier
from ..tasks.generation import TextGenerator
from ..tasks.ner import NamedEntityRecognizer
from ..tasks.sentiment import SentimentAnalyzer

app = FastAPI(title="NLP Model Serving")


@lru_cache(maxsize=1)
def _processor() -> TextProcessor:
    return TextProcessor()


@lru_cache(maxsize=1)
def _classifier() -> TextClassifier:
    return TextClassifier()


@lru_cache(maxsize=1)
def _ner() -> NamedEntityRecognizer:
    return NamedEntityRecognizer()


@lru_cache(maxsize=1)
def _sentiment() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@lru_cache(maxsize=1)
def _generator() -> TextGenerator:
    return TextGenerator()


@app.post("/classify")
async def classify(texts: List[str]) -> List[dict]:
    processed = [_processor().process(t).text for t in texts]
    return _classifier().predict(processed)


@app.post("/ner")
async def ner(texts: List[str]) -> List[List[dict]]:
    processed = [_processor().process(t).text for t in texts]
    return _ner().predict(processed)


@app.post("/sentiment")
async def sentiment(texts: List[str]) -> List[dict]:
    processed = [_processor().process(t).text for t in texts]
    return _sentiment().predict(processed)


@app.post("/generate")
async def generate(prompts: List[str]) -> List[str]:
    processed = [_processor().process(p).text for p in prompts]
    return _generator().predict(processed)


__all__ = ["app"]
