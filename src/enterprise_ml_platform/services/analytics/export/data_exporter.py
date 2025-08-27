"""Data export helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


class DataExporter:
    """Export analytics data in common formats."""

    def export(self, data: Iterable[Dict[str, float]], path: str | Path, fmt: str = "csv") -> None:
        p = Path(path)
        if fmt == "json":
            with p.open("w", encoding="utf-8") as f:
                json.dump(list(data), f)
        else:  # default to CSV
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in data for k in row}))
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
