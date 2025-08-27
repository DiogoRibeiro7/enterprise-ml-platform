"""End-to-end NLP pipeline example using transformer models."""
from __future__ import annotations

from typing import List

from .preprocessing.text_processor import TextProcessor
from .models.transformer_models import TransformerModels
from .tasks.classification import TextClassifier
from .tasks.generation import TextGenerator
from .tasks.ner import NamedEntityRecognizer
from .tasks.sentiment import SentimentAnalyzer


class NLPPipeline:
    """Coordinate preprocessing and modelling for NLP tasks."""

    def __init__(self) -> None:
        self.processor = TextProcessor()
        self.models = TransformerModels()
        self.classifier = TextClassifier()
        self.ner = NamedEntityRecognizer()
        self.sentiment = SentimentAnalyzer()
        self.generator = TextGenerator()

    def run_all(self, texts: List[str]) -> dict:
        processed = [self.processor.process(t) for t in texts]
        cleaned = [p.text for p in processed]

        classification = self.classifier.predict(cleaned)
        ner = self.ner.predict(cleaned)
        sentiment = self.sentiment.predict(cleaned)
        generation = self.generator.predict(cleaned)

        return {
            "classification": classification,
            "ner": ner,
            "sentiment": sentiment,
            "generation": generation,
        }


def main() -> None:
    pipeline = NLPPipeline()
    texts = ["<p>Hello world! My email is test@example.com</p>"]
    results = pipeline.run_all(texts)
    print(results)


if __name__ == "__main__":
    main()
