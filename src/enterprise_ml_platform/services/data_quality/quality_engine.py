"""High level orchestrator for data quality validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .validators.schema_validator import SchemaValidator
from .validators.business_rule_validator import BusinessRuleValidator
from .profiling.data_profiler import DataProfiler
from .anomaly.anomaly_detector import AnomalyDetector
from .monitoring.quality_monitor import QualityMonitor
from .rules.rule_engine import RuleEngine
from .reporting.quality_reporter import QualityReporter
from .remediation.data_remediation import DataRemediation


@dataclass
class QualityEngine:
    """Coordinate validation, profiling and remediation of datasets."""

    schema_validator: Optional[SchemaValidator] = None
    business_validator: BusinessRuleValidator = field(default_factory=BusinessRuleValidator)
    profiler: DataProfiler = field(default_factory=DataProfiler)
    anomaly_detector: AnomalyDetector = field(default_factory=AnomalyDetector)
    monitor: QualityMonitor = field(default_factory=QualityMonitor)
    rules: RuleEngine = field(default_factory=RuleEngine)
    reporter: QualityReporter = field(default_factory=QualityReporter)
    remediation: DataRemediation = field(default_factory=DataRemediation)

    def run(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Execute all quality checks and return issues by category."""

        results: Dict[str, List[str]] = {}
        if self.schema_validator:
            results["schema"] = self.schema_validator.validate(df)
        if self.business_validator.rules:
            results["business"] = self.business_validator.validate(df)
        rule_results = self.rules.run(df)
        results.update({f"rule:{k}": v for k, v in rule_results.items()})
        anomalies = self.anomaly_detector.detect(df)
        results["anomalies"] = [f"{col}:{idx}" for col, idx in anomalies.items()]

        # Compute simple quality score: proportion of checks without issues
        total_sections = len(results)
        failed = sum(1 for issues in results.values() if issues)
        score = 1 - failed / total_sections if total_sections else 1.0
        self.monitor.record(score)

        # Attach report string for convenience
        results["report"] = [self.reporter.generate(results, score)]
        return results

    def remediate(self, df: pd.DataFrame, results: Dict[str, List[str]]) -> pd.DataFrame:
        """Apply automatic remediation based on ``results``."""

        issues = [msg for msgs in results.values() for msg in msgs]
        return self.remediation.remediate(df, issues)
