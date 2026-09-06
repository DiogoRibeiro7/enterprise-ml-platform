"""Data quality validation utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import structlog


@dataclass(frozen=True)
class ValidationReport:
    """Outcome and summary statistics from a validation run."""

    passed: bool
    errors: tuple[str, ...]
    stats: dict[str, int | float]


class DataValidator:
    """Apply schema, completeness, uniqueness, and custom rules to a frame."""

    def __init__(self, schema: Mapping[str, str] | None = None) -> None:
        self.schema = dict(schema or {})
        self.last_report: ValidationReport | None = None
        self._log = structlog.get_logger().bind(component="data_validator")

    def infer_schema(self, frame: pd.DataFrame) -> dict[str, str]:
        """Infer and store a mapping of column names to pandas dtypes."""
        self.schema = {
            str(column): str(dtype) for column, dtype in frame.dtypes.items()
        }
        return self.schema.copy()

    async def validate(
        self,
        frame: pd.DataFrame,
        rules: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Apply configured rules and store their report in :attr:`last_report`."""
        errors: list[str] = []

        if not self.schema:
            self.infer_schema(frame)
        else:
            missing = [column for column in self.schema if column not in frame.columns]
            if missing:
                errors.append(f"missing columns: {missing}")
            for column, dtype in self.schema.items():
                if column in frame.columns and str(frame[column].dtype) != dtype:
                    errors.append(
                        f"column {column} expected {dtype} got {frame[column].dtype}"
                    )

        for rule in rules or []:
            rule_type = str(rule.get("type", ""))
            if rule_type == "completeness":
                threshold = float(rule.get("threshold", 0.0))
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError("completeness threshold must be between 0 and 1")
                missing_ratio = float(frame.isna().mean().mean())
                if missing_ratio > threshold:
                    errors.append(
                        f"missing value ratio {missing_ratio:.2%} "
                        f"exceeds {threshold:.2%}"
                    )
                if rule.get("impute"):
                    frame = frame.fillna(rule.get("impute_value", 0))
            elif rule_type == "uniqueness":
                rows_before = len(frame)
                frame = frame.drop_duplicates()
                duplicates = rows_before - len(frame)
                if duplicates:
                    self._log.info("duplicates removed", count=duplicates)
            elif rule_type == "custom":
                predicate = rule.get("func")
                if not callable(predicate):
                    raise TypeError("custom validation rule requires a callable func")
                typed_predicate = cast(Callable[[pd.DataFrame], bool], predicate)
                if not typed_predicate(frame):
                    errors.append(str(rule.get("message", "custom rule failed")))
            else:
                raise ValueError(f"Unsupported validation rule type {rule_type!r}")

        self.last_report = ValidationReport(
            passed=not errors,
            errors=tuple(errors),
            stats={
                "rows": len(frame),
                "columns": frame.shape[1],
                "errors": len(errors),
            },
        )
        if errors:
            self._log.warning("validation failed", errors=errors)
        return frame
