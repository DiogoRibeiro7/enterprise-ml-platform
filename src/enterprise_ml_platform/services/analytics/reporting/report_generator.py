"""Automated report generation utilities."""

from __future__ import annotations

from collections.abc import Iterable


class ReportGenerator:
    """Create simple textual reports from metrics and insights."""

    def generate(self, metrics: dict[str, float], insights: Iterable[str]) -> str:
        lines = ["Analytics Report", "================", ""]
        lines.append("Key Metrics:")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value:.3f}")
        if insights:
            lines.append("")
            lines.append("Insights:")
            for insight in insights:
                lines.append(f"- {insight}")
        return "\n".join(lines)
