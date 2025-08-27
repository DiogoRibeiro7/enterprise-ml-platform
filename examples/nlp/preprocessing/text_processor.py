"""Advanced text preprocessing utilities for NLP tasks.

This module provides a :class:`TextProcessor` that performs:
- HTML stripping and normalization
- Language detection
- Tokenization using HuggingFace tokenizers
- Duplicate removal
- Simple anonymization of emails and numbers

The processor is designed for multilingual input and privacy preserving
pipelines.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable, List

from langdetect import detect
from transformers import AutoTokenizer


@dataclass
class ProcessedText:
    """Container for processed text outputs."""

    text: str
    language: str
    tokens: List[str]


class TextProcessor:
    """Perform text cleaning, normalization and tokenization."""

    def __init__(self, model_name: str = "hf-internal-testing/tiny-random-bert") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags and unescape entities."""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        return html.unescape(cleaned)

    @staticmethod
    def _anonymize(text: str) -> str:
        """Anonymize emails and numbers for privacy."""
        text = re.sub(r"[\w.-]+@[\w.-]+", "<email>", text)
        text = re.sub(r"\b\d+\b", "<num>", text)
        return text

    @staticmethod
    def _deduplicate(lines: Iterable[str]) -> List[str]:
        seen = set()
        unique = []
        for line in lines:
            if line not in seen:
                unique.append(line)
                seen.add(line)
        return unique

    def process(self, text: str) -> ProcessedText:
        """Run the full preprocessing pipeline on ``text``."""
        clean = self._strip_html(text)
        clean = self._anonymize(clean)
        language = detect(clean)
        tokens = self.tokenizer.tokenize(clean)
        return ProcessedText(text=clean, language=language, tokens=tokens)


__all__ = ["TextProcessor", "ProcessedText"]
