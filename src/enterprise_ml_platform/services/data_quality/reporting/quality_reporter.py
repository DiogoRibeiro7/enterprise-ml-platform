"""Generate human readable data quality reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityReporter:
    """Produce simple text summaries of validation results."""

    def generate(self, results: dict[str, list[str]], score: float) -> str:
        lines = [f"Quality score: {score:.2f}"]
        for section, issues in results.items():
            if not issues:
                continue
            lines.append(f"\n{section} issues:")
            lines.extend(f"- {msg}" for msg in issues)
        return "\n".join(lines)
