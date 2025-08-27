"""PII detection and anonymisation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


@dataclass
class PiiDetector:
    patterns: Iterable[Tuple[str, re.Pattern]] = (
        ("email", EMAIL_RE),
        ("phone", PHONE_RE),
        ("ssn", SSN_RE),
    )

    def detect(self, text: str) -> List[Tuple[str, str]]:
        """Return list of ``(type, value)`` for PII found in ``text``."""

        findings: List[Tuple[str, str]] = []
        for name, pattern in self.patterns:
            for match in pattern.findall(text):
                findings.append((name, match))
        return findings

    def anonymize(self, text: str) -> str:
        """Replace PII in ``text`` with redacted placeholders."""

        redacted = text
        for name, pattern in self.patterns:
            redacted = pattern.sub(f"<{name}>", redacted)
        return redacted

