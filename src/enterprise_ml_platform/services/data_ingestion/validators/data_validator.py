"""Data quality validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog


@dataclass
class ValidationReport:
    """Outcome of a validation run."""

    passed: bool
    errors: List[str]
    stats: Dict[str, Any]


class DataValidator:
    """Rule based data validator.

    The validator executes a set of quality rules on a ``pandas`` data frame
    and optionally infers a reference schema for subsequent batches.
    """

    def __init__(self, schema: Optional[Dict[str, str]] = None) -> None:
        self.schema = schema or {}
        self._log = structlog.get_logger().bind(component="data_validator")

    def infer_schema(self, frame: pd.DataFrame) -> Dict[str, str]:
        """Infer a schema mapping column names to dtypes."""
        self.schema = {col: str(dtype) for col, dtype in frame.dtypes.items()}
        return self.schema

    async def validate(
        self, frame: pd.DataFrame, rules: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """Validate ``frame`` according to ``rules``.

        Supported rule types:

        ``schema``: Ensures that the data frame matches the inferred schema.
        ``completeness``: Checks for missing values, optionally fills them.
        ``uniqueness``: Removes duplicate rows.
        ``custom``: Executes a user provided callable.
        """

        errors: List[str] = []

        if not self.schema:
            self.infer_schema(frame)
        else:
            missing = [c for c in self.schema if c not in frame.columns]
            if missing:
                errors.append(f"missing columns: {missing}")
            for col, dtype in self.schema.items():
                if col in frame.columns and str(frame[col].dtype) != dtype:
                    errors.append(
                        f"column {col} expected {dtype} got {frame[col].dtype}"
                    )

        for rule in rules or []:
            if rule.get("type") == "completeness":
                threshold = rule.get("threshold", 0.0)
                ratio = frame.isna().mean().mean()
                if ratio > threshold:
                    errors.append(
                        f"missing value ratio {ratio:.2%} exceeds {threshold:.2%}"
                    )
                if rule.get("impute"):
                    frame = frame.fillna(rule.get("impute_value", 0))
            elif rule.get("type") == "uniqueness":
                before = len(frame)
                frame = frame.drop_duplicates()
                if before != len(frame):
                    self._log.info("duplicates removed", count=before - len(frame))
            elif rule.get("type") == "custom":
                func = rule["func"]
                if not func(frame):
                    errors.append(rule.get("message", "custom rule failed"))

        report = ValidationReport(
            passed=not errors,
            errors=errors,
            stats={"rows": len(frame)},
        )
        if not report.passed:
            self._log.warning("validation failed", errors=errors)
        return frame
