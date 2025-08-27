"""Generate Grafana dashboard configs for experiments."""
from __future__ import annotations

from typing import Dict, Any


def generate_dashboard(experiment: str) -> Dict[str, Any]:
    """Return a minimal Grafana dashboard config for the experiment."""
    return {
        "title": f"AB Test - {experiment}",
        "panels": [
            {
                "type": "graph",
                "title": "Conversions",
                "targets": [
                    {"expr": f"sum(ab_test_conversions{{experiment='{experiment}'}}) by (variant)"}
                ],
            }
        ],
    }
