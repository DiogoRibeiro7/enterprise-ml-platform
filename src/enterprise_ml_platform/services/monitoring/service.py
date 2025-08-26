# Enterprise ML Pipeline - Advanced Monitoring and Complete Orchestration
# File: services/monitoring/service.py

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import asyncpg
import numpy as np
import pandas as pd
import structlog
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod

from redis import asyncio as aioredis

# Monitoring and observability
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    start_http_server,
    CollectorRegistry,
)
import grafana_api
from elasticsearch import AsyncElasticsearch
import mlflow
from mlflow.tracking import MlflowClient

# Advanced monitoring libraries
from evidently.model_monitoring import (
    CombinedDriftDetector,
    DataDriftDetector,
    TargetDriftDetector,
)
from evidently.model_monitoring.monitors import (
    DataDriftMonitor,
    TargetDriftMonitor,
    RegressionPerformanceMonitor,
)
from evidently.pipeline.column_mapping import ColumnMapping
import alibi_detect
from alibi_detect.drift import KSDrift, MMDDrift, TabularDrift
from river import drift
import deepchecks
from deepchecks.core import DatasetKind
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

# Statistical libraries
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import permutation_test_score

# Infrastructure monitoring
import psutil
import GPUtil
import kubernetes
from kubernetes import client, config as k8s_config

# Alerting
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import slack_sdk
from slack_sdk.webhook import WebhookClient
import pagerduty
from twilio.rest import Client as TwilioClient

from core.pipeline_orchestrator import (
    BasePipelineStage,
    ExecutionContext,
    StageResult,
    PipelineStage,
    ExecutionStatus,
)

logger = structlog.get_logger()

# Monitoring metrics
PREDICTION_COUNTER = Counter(
    "ml_predictions_total", "Total predictions made", ["model_name", "version"]
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds", "Prediction latency", ["model_name"]
)
PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total", "Prediction errors", ["model_name", "error_type"]
)
DATA_DRIFT_SCORE = Gauge(
    "ml_data_drift_score", "Data drift detection score", ["feature_name"]
)
TARGET_DRIFT_SCORE = Gauge(
    "ml_target_drift_score", "Target drift detection score", ["model_name"]
)
MODEL_ACCURACY = Gauge(
    "ml_model_accuracy", "Current model accuracy", ["model_name", "version"]
)
FEATURE_IMPORTANCE_DRIFT = Gauge(
    "ml_feature_importance_drift", "Feature importance drift", ["feature_name"]
)
SYSTEM_RESOURCE_USAGE = Gauge(
    "ml_system_resource_usage", "System resource usage", ["resource_type"]
)


@dataclass
class AlertConfig:
    """Configuration for alerts"""

    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'critical', 'warning', 'info'
    channels: List[str]  # email, slack, pagerduty, sms
    cooldown_minutes: int = 30
    conditions: Optional[Dict[str, Any]] = None


@dataclass
class MonitoringMetrics:
    """Comprehensive monitoring metrics"""

    model_performance: Dict[str, float] = field(default_factory=dict)
    data_drift: Dict[str, float] = field(default_factory=dict)
    target_drift: float = 0.0
    prediction_latency: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    system_resources: Dict[str, float] = field(default_factory=dict)
    feature_importance_drift: Dict[str, float] = field(default_factory=dict)
    data_quality: Dict[str, float] = field(default_factory=dict)


class DriftDetector(ABC):
    """Abstract base class for drift detection"""

    @abstractmethod
    async def fit(self, reference_data: pd.DataFrame) -> "DriftDetector":
        pass

    @abstractmethod
    async def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        pass


class AdvancedDriftDetector(DriftDetector):
    """Advanced drift detector using multiple algorithms"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reference_data = None
        self.detectors = {}
        self.column_mapping = None
        self.logger = structlog.get_logger().bind(component="drift_detector")

    async def fit(self, reference_data: pd.DataFrame) -> "AdvancedDriftDetector":
        """Fit drift detectors on reference data"""

        self.reference_data = reference_data.copy()

        # Initialize different drift detection algorithms
        numeric_features = reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_features = reference_data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Kolmogorov-Smirnov drift detector for numerical features
        if numeric_features and self.config.get("ks_drift", {}).get("enabled", True):
            self.detectors["ks"] = {}
            for feature in numeric_features:
                detector = KSDrift(
                    reference_data[feature].values,
                    p_val=self.config.get("ks_drift", {}).get("p_val", 0.05),
                )
                self.detectors["ks"][feature] = detector

        # Maximum Mean Discrepancy for multivariate drift
        if self.config.get("mmd_drift", {}).get("enabled", True):
            try:
                self.detectors["mmd"] = MMDDrift(
                    reference_data[numeric_features].values,
                    p_val=self.config.get("mmd_drift", {}).get("p_val", 0.05),
                    n_permutations=self.config.get("mmd_drift", {}).get(
                        "n_permutations", 100
                    ),
                )
            except Exception as e:
                self.logger.warning(
                    "MMD drift detector initialization failed", error=str(e)
                )

        # Tabular drift detector for mixed data types
        if self.config.get("tabular_drift", {}).get("enabled", True):
            try:
                self.detectors["tabular"] = TabularDrift(
                    reference_data.values,
                    p_val=self.config.get("tabular_drift", {}).get("p_val", 0.05),
                    categories_per_feature=self._get_categories_per_feature(
                        reference_data
                    ),
                )
            except Exception as e:
                self.logger.warning(
                    "Tabular drift detector initialization failed", error=str(e)
                )

        # River-based drift detectors for streaming data
        if self.config.get("river_drift", {}).get("enabled", False):
            self.detectors["river"] = {}
            for feature in numeric_features:
                detector = drift.ADWIN(
                    delta=self.config.get("river_drift", {}).get("delta", 0.002)
                )
                self.detectors["river"][feature] = detector

        self.logger.info(
            "Drift detectors fitted", detectors=list(self.detectors.keys())
        )
        return self

    def _get_categories_per_feature(
        self, data: pd.DataFrame
    ) -> Dict[int, Optional[int]]:
        """Get categories per feature for tabular drift detector"""
        categories_per_feature = {}

        for i, col in enumerate(data.columns):
            if data[col].dtype in ["object", "category"]:
                categories_per_feature[i] = data[col].nunique()
            else:
                categories_per_feature[i] = None

        return categories_per_feature

    async def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in current data"""

        drift_results = {
            "overall_drift": False,
            "feature_drift": {},
            "drift_scores": {},
            "p_values": {},
        }

        # KS drift detection for individual features
        if "ks" in self.detectors:
            for feature, detector in self.detectors["ks"].items():
                if feature in current_data.columns:
                    result = detector.predict(current_data[feature].values)

                    drift_results["feature_drift"][feature] = bool(
                        result["data"]["is_drift"]
                    )
                    drift_results["drift_scores"][f"{feature}_ks"] = float(
                        result["data"]["distance"]
                    )
                    drift_results["p_values"][f"{feature}_ks"] = float(
                        result["data"]["p_val"]
                    )

        # MMD drift detection for multivariate drift
        if "mmd" in self.detectors:
            try:
                numeric_features = current_data.select_dtypes(
                    include=[np.number]
                ).columns
                result = self.detectors["mmd"].predict(
                    current_data[numeric_features].values
                )

                drift_results["multivariate_drift"] = bool(result["data"]["is_drift"])
                drift_results["drift_scores"]["mmd"] = float(result["data"]["distance"])
                drift_results["p_values"]["mmd"] = float(result["data"]["p_val"])

            except Exception as e:
                self.logger.warning("MMD drift detection failed", error=str(e))

        # Tabular drift detection
        if "tabular" in self.detectors:
            try:
                result = self.detectors["tabular"].predict(current_data.values)

                drift_results["tabular_drift"] = bool(result["data"]["is_drift"])
                drift_results["drift_scores"]["tabular"] = float(
                    result["data"]["distance"]
                )
                drift_results["p_values"]["tabular"] = float(result["data"]["p_val"])

            except Exception as e:
                self.logger.warning("Tabular drift detection failed", error=str(e))

        # River drift detection for streaming
        if "river" in self.detectors:
            for feature, detector in self.detectors["river"].items():
                if feature in current_data.columns:
                    for value in current_data[feature]:
                        if not pd.isna(value):
                            change_detected = detector.update(value)
                            if change_detected:
                                drift_results["feature_drift"][f"{feature}_river"] = (
                                    True
                                )
                                break

        # Overall drift assessment
        feature_drifts = [
            v for v in drift_results["feature_drift"].values() if isinstance(v, bool)
        ]
        multivariate_drift = drift_results.get("multivariate_drift", False)
        tabular_drift = drift_results.get("tabular_drift", False)

        drift_results["overall_drift"] = (
            any(feature_drifts) or multivariate_drift or tabular_drift
        )

        # Update Prometheus metrics
        for feature, score in drift_results["drift_scores"].items():
            DATA_DRIFT_SCORE.labels(feature_name=feature).set(score)

        return drift_results


class PerformanceMonitor:
    """Advanced model performance monitoring"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reference_performance = {}
        self.performance_history = []
        self.logger = structlog.get_logger().bind(component="performance_monitor")

    async def set_baseline_performance(
        self, model_name: str, baseline_metrics: Dict[str, float]
    ):
        """Set baseline performance metrics"""
        self.reference_performance[model_name] = baseline_metrics
        self.logger.info(
            "Baseline performance set", model=model_name, metrics=baseline_metrics
        )

    async def monitor_performance(
        self,
        model_name: str,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Monitor model performance"""

        performance_metrics = {
            "timestamp": datetime.now(),
            "model_name": model_name,
            "prediction_count": len(predictions),
        }

        if actuals is not None:
            # Calculate performance metrics
            if len(np.unique(actuals)) <= 10:  # Classification
                performance_metrics.update(
                    {
                        "accuracy": accuracy_score(actuals, predictions),
                        "precision": precision_score(
                            actuals, predictions, average="weighted"
                        ),
                        "recall": recall_score(
                            actuals, predictions, average="weighted"
                        ),
                        "f1_score": f1_score(actuals, predictions, average="weighted"),
                    }
                )

                if probabilities is not None and len(np.unique(actuals)) == 2:
                    performance_metrics["auc_roc"] = roc_auc_score(
                        actuals, probabilities[:, 1]
                    )

            else:  # Regression
                from sklearn.metrics import (
                    mean_squared_error,
                    mean_absolute_error,
                    r2_score,
                )

                performance_metrics.update(
                    {
                        "mse": mean_squared_error(actuals, predictions),
                        "mae": mean_absolute_error(actuals, predictions),
                        "r2_score": r2_score(actuals, predictions),
                    }
                )

            # Performance degradation detection
            if model_name in self.reference_performance:
                degradation = await self._detect_performance_degradation(
                    model_name, performance_metrics
                )
                performance_metrics["performance_degradation"] = degradation

            # Update Prometheus metrics
            main_metric = performance_metrics.get(
                "accuracy"
            ) or performance_metrics.get("r2_score", 0)
            MODEL_ACCURACY.labels(model_name=model_name, version="current").set(
                main_metric
            )

        # Store performance history
        self.performance_history.append(performance_metrics)

        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(days=30)
        self.performance_history = [
            p for p in self.performance_history if p["timestamp"] > cutoff_time
        ]

        return performance_metrics

    async def _detect_performance_degradation(
        self, model_name: str, current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Detect performance degradation"""

        baseline = self.reference_performance[model_name]
        degradation_results = {}

        # Define critical metrics and thresholds
        critical_metrics = {
            "accuracy": 0.05,  # 5% degradation threshold
            "f1_score": 0.05,
            "auc_roc": 0.03,
            "r2_score": 0.1,
        }

        for metric, threshold in critical_metrics.items():
            if metric in baseline and metric in current_metrics:
                baseline_value = baseline[metric]
                current_value = current_metrics[metric]

                # Calculate relative degradation
                if baseline_value > 0:
                    relative_degradation = (
                        baseline_value - current_value
                    ) / baseline_value

                    degradation_results[metric] = {
                        "baseline": baseline_value,
                        "current": current_value,
                        "relative_degradation": relative_degradation,
                        "is_degraded": relative_degradation > threshold,
                    }

        return degradation_results


class DataQualityMonitor:
    """Comprehensive data quality monitoring"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_rules = config.get("quality_rules", [])
        self.logger = structlog.get_logger().bind(component="data_quality_monitor")

    async def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality comprehensively"""

        quality_results = {
            "overall_score": 0.0,
            "completeness": {},
            "validity": {},
            "consistency": {},
            "uniqueness": {},
            "timeliness": {},
        }

        # Completeness assessment
        quality_results["completeness"] = await self._assess_completeness(data)

        # Validity assessment
        quality_results["validity"] = await self._assess_validity(data)

        # Consistency assessment
        quality_results["consistency"] = await self._assess_consistency(data)

        # Uniqueness assessment
        quality_results["uniqueness"] = await self._assess_uniqueness(data)

        # Timeliness assessment (if timestamp columns exist)
        quality_results["timeliness"] = await self._assess_timeliness(data)

        # Calculate overall quality score
        dimension_scores = []
        for dimension, results in quality_results.items():
            if dimension != "overall_score" and isinstance(results, dict):
                if "score" in results:
                    dimension_scores.append(results["score"])

        quality_results["overall_score"] = (
            np.mean(dimension_scores) if dimension_scores else 0.0
        )

        return quality_results

    async def _assess_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness"""

        missing_counts = data.isnull().sum()
        total_records = len(data)

        completeness_scores = {}
        for col in data.columns:
            completeness_scores[col] = 1.0 - (missing_counts[col] / total_records)

        overall_completeness = np.mean(list(completeness_scores.values()))

        return {
            "score": overall_completeness,
            "by_column": completeness_scores.to_dict(),
            "total_missing": missing_counts.sum(),
            "missing_percentage": missing_counts.sum()
            / (total_records * len(data.columns)),
        }

    async def _assess_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity based on defined rules"""

        validity_results = {"score": 1.0, "violations": {}, "rule_results": {}}

        for rule in self.quality_rules:
            if rule["type"] == "range":
                column = rule["column"]
                min_val, max_val = rule["range"]

                if column in data.columns:
                    violations = (
                        (data[column] < min_val) | (data[column] > max_val)
                    ).sum()
                    violation_rate = violations / len(data)

                    validity_results["violations"][f"{column}_range"] = violations
                    validity_results["rule_results"][f"{column}_range"] = {
                        "violations": violations,
                        "violation_rate": violation_rate,
                        "passed": violation_rate < rule.get("threshold", 0.05),
                    }

            elif rule["type"] == "categorical":
                column = rule["column"]
                allowed_values = set(rule["allowed_values"])

                if column in data.columns:
                    invalid_mask = ~data[column].isin(allowed_values)
                    violations = invalid_mask.sum()
                    violation_rate = violations / len(data)

                    validity_results["violations"][f"{column}_categorical"] = violations
                    validity_results["rule_results"][f"{column}_categorical"] = {
                        "violations": violations,
                        "violation_rate": violation_rate,
                        "passed": violation_rate < rule.get("threshold", 0.05),
                    }

        # Calculate overall validity score
        if validity_results["rule_results"]:
            passed_rules = sum(
                1 for r in validity_results["rule_results"].values() if r["passed"]
            )
            validity_results["score"] = passed_rules / len(
                validity_results["rule_results"]
            )

        return validity_results

    async def _assess_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency"""

        consistency_results = {"score": 1.0, "inconsistencies": {}}

        # Check for inconsistent data types within columns
        for col in data.columns:
            if data[col].dtype == "object":
                # Check for mixed types in object columns
                non_null_data = data[col].dropna()
                if len(non_null_data) > 0:
                    type_counts = non_null_data.apply(
                        lambda x: type(x).__name__
                    ).value_counts()
                    if len(type_counts) > 1:
                        consistency_results["inconsistencies"][f"{col}_mixed_types"] = (
                            type_counts.to_dict()
                        )

        return consistency_results

    async def _assess_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness"""

        duplicate_count = data.duplicated().sum()
        uniqueness_score = 1.0 - (duplicate_count / len(data))

        # Check uniqueness for columns that should be unique
        unique_violations = {}
        for rule in self.quality_rules:
            if rule["type"] == "unique" and rule["column"] in data.columns:
                column = rule["column"]
                duplicates = data[column].duplicated().sum()
                unique_violations[column] = duplicates

        return {
            "score": uniqueness_score,
            "duplicate_rows": duplicate_count,
            "unique_violations": unique_violations,
        }

    async def _assess_timeliness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data timeliness"""

        timeliness_results = {"score": 1.0, "stale_data": {}}

        # Look for datetime columns
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns

        for col in datetime_cols:
            latest_timestamp = data[col].max()
            current_time = pd.Timestamp.now()

            if pd.notna(latest_timestamp):
                age_hours = (current_time - latest_timestamp).total_seconds() / 3600

                # Define staleness threshold (configurable)
                staleness_threshold = self.config.get("staleness_threshold_hours", 24)

                timeliness_results["stale_data"][col] = {
                    "latest_timestamp": latest_timestamp,
                    "age_hours": age_hours,
                    "is_stale": age_hours > staleness_threshold,
                }

        return timeliness_results


class AlertManager:
    """Advanced alerting system with multiple channels"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = {}
        self.notification_clients = {}
        self.logger = structlog.get_logger().bind(component="alert_manager")

        # Initialize notification clients
        self._initialize_notification_clients()

    def _initialize_notification_clients(self):
        """Initialize notification clients for different channels"""

        # Email client
        if self.config.get("email", {}).get("enabled", False):
            self.notification_clients["email"] = {
                "smtp_server": self.config["email"]["smtp_server"],
                "smtp_port": self.config["email"]["smtp_port"],
                "username": self.config["email"]["username"],
                "password": self.config["email"]["password"],
                "from_email": self.config["email"]["from_email"],
            }

        # Slack client
        if self.config.get("slack", {}).get("enabled", False):
            self.notification_clients["slack"] = WebhookClient(
                url=self.config["slack"]["webhook_url"]
            )

        # PagerDuty client
        if self.config.get("pagerduty", {}).get("enabled", False):
            self.notification_clients["pagerduty"] = pagerduty.Events(
                integration_key=self.config["pagerduty"]["integration_key"]
            )

        # SMS client (Twilio)
        if self.config.get("sms", {}).get("enabled", False):
            self.notification_clients["sms"] = TwilioClient(
                self.config["sms"]["account_sid"], self.config["sms"]["auth_token"]
            )

    async def evaluate_alerts(self, metrics: MonitoringMetrics) -> List[Dict[str, Any]]:
        """Evaluate all configured alerts against current metrics"""

        triggered_alerts = []
        alert_configs = self.config.get("alerts", [])

        for alert_config in alert_configs:
            alert = AlertConfig(**alert_config)

            if await self._should_trigger_alert(alert, metrics):
                triggered_alerts.append(await self._trigger_alert(alert, metrics))

        return triggered_alerts

    async def _should_trigger_alert(
        self, alert: AlertConfig, metrics: MonitoringMetrics
    ) -> bool:
        """Check if alert should be triggered"""

        # Get metric value
        metric_value = await self._get_metric_value(alert.metric, metrics)

        if metric_value is None:
            return False

        # Check cooldown period
        alert_key = f"{alert.name}_{alert.metric}"
        if alert_key in self.alert_history:
            last_triggered = self.alert_history[alert_key]["last_triggered"]
            if datetime.now() - last_triggered < timedelta(
                minutes=alert.cooldown_minutes
            ):
                return False

        # Evaluate threshold condition
        if alert.comparison == "gt":
            condition_met = metric_value > alert.threshold
        elif alert.comparison == "lt":
            condition_met = metric_value < alert.threshold
        elif alert.comparison == "eq":
            condition_met = metric_value == alert.threshold
        else:
            return False

        # Evaluate additional conditions if specified
        if alert.conditions:
            for condition_key, condition_value in alert.conditions.items():
                context_value = await self._get_metric_value(condition_key, metrics)
                if context_value != condition_value:
                    condition_met = False
                    break

        return condition_met

    async def _get_metric_value(
        self, metric_path: str, metrics: MonitoringMetrics
    ) -> Optional[float]:
        """Extract metric value from monitoring metrics using dot notation"""

        parts = metric_path.split(".")
        value = metrics

        try:
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None

            return float(value) if isinstance(value, (int, float)) else None

        except (AttributeError, KeyError, ValueError, TypeError):
            return None

    async def _trigger_alert(
        self, alert: AlertConfig, metrics: MonitoringMetrics
    ) -> Dict[str, Any]:
        """Trigger alert through configured channels"""

        alert_data = {
            "alert_name": alert.name,
            "metric": alert.metric,
            "threshold": alert.threshold,
            "current_value": await self._get_metric_value(alert.metric, metrics),
            "severity": alert.severity,
            "timestamp": datetime.now(),
            "comparison": alert.comparison,
        }

        # Update alert history
        alert_key = f"{alert.name}_{alert.metric}"
        self.alert_history[alert_key] = {
            "last_triggered": datetime.now(),
            "trigger_count": self.alert_history.get(alert_key, {}).get(
                "trigger_count", 0
            )
            + 1,
        }

        # Send notifications through configured channels
        notification_results = {}

        for channel in alert.channels:
            try:
                if channel == "email":
                    result = await self._send_email_alert(alert_data)
                elif channel == "slack":
                    result = await self._send_slack_alert(alert_data)
                elif channel == "pagerduty":
                    result = await self._send_pagerduty_alert(alert_data)
                elif channel == "sms":
                    result = await self._send_sms_alert(alert_data)
                else:
                    result = {"status": "unsupported_channel"}

                notification_results[channel] = result

            except Exception as e:
                notification_results[channel] = {"status": "failed", "error": str(e)}
                self.logger.error(
                    "Alert notification failed", channel=channel, error=str(e)
                )

        alert_data["notification_results"] = notification_results

        self.logger.info(
            "Alert triggered",
            alert=alert.name,
            severity=alert.severity,
            channels=alert.channels,
        )

        return alert_data

    async def _send_email_alert(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """Send email alert"""

        if "email" not in self.notification_clients:
            return {"status": "not_configured"}

        email_config = self.notification_clients["email"]

        subject = f"[{alert_data['severity'].upper()}] ML Pipeline Alert: {alert_data['alert_name']}"

        body = f"""
        Alert: {alert_data["alert_name"]}
        Severity: {alert_data["severity"]}
        Metric: {alert_data["metric"]}
        Current Value: {alert_data["current_value"]}
        Threshold: {alert_data["threshold"]} ({alert_data["comparison"]})
        Timestamp: {alert_data["timestamp"]}
        """

        # Create message
        msg = MimeMultipart()
        msg["From"] = email_config["from_email"]
        msg["To"] = ", ".join(self.config.get("email", {}).get("recipients", []))
        msg["Subject"] = subject
        msg.attach(MimeText(body, "plain"))

        # Send email
        try:
            with smtplib.SMTP(
                email_config["smtp_server"], email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(email_config["username"], email_config["password"])
                server.send_message(msg)

            return {"status": "sent"}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _send_slack_alert(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """Send Slack alert"""

        if "slack" not in self.notification_clients:
            return {"status": "not_configured"}

        slack_client = self.notification_clients["slack"]

        # Determine color based on severity
        colors = {"critical": "#FF0000", "warning": "#FFA500", "info": "#00FF00"}

        color = colors.get(alert_data["severity"], "#808080")

        message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"ML Pipeline Alert: {alert_data['alert_name']}",
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert_data["severity"].upper(),
                            "short": True,
                        },
                        {
                            "title": "Metric",
                            "value": alert_data["metric"],
                            "short": True,
                        },
                        {
                            "title": "Current Value",
                            "value": str(alert_data["current_value"]),
                            "short": True,
                        },
                        {
                            "title": "Threshold",
                            "value": f"{alert_data['threshold']} ({alert_data['comparison']})",
                            "short": True,
                        },
                    ],
                    "timestamp": int(alert_data["timestamp"].timestamp()),
                }
            ]
        }

        try:
            response = slack_client.send(**message)
            return {"status": "sent", "response": str(response.status_code)}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _send_pagerduty_alert(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """Send PagerDuty alert"""

        if "pagerduty" not in self.notification_clients:
            return {"status": "not_configured"}

        pd_client = self.notification_clients["pagerduty"]

        try:
            response = pd_client.trigger(
                summary=f"ML Pipeline Alert: {alert_data['alert_name']}",
                severity=alert_data["severity"],
                source="ml-pipeline",
                custom_details={
                    "metric": alert_data["metric"],
                    "current_value": alert_data["current_value"],
                    "threshold": alert_data["threshold"],
                    "comparison": alert_data["comparison"],
                },
            )

            return {"status": "sent", "incident_key": response.get("incident_key")}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _send_sms_alert(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """Send SMS alert"""

        if "sms" not in self.notification_clients:
            return {"status": "not_configured"}

        twilio_client = self.notification_clients["sms"]

        message_body = (
            f"ML Alert: {alert_data['alert_name']} "
            f"({alert_data['severity']}) - "
            f"{alert_data['metric']}: {alert_data['current_value']} "
            f"(threshold: {alert_data['threshold']})"
        )

        try:
            recipients = self.config.get("sms", {}).get("recipients", [])

            for recipient in recipients:
                message = twilio_client.messages.create(
                    body=message_body,
                    from_=self.config["sms"]["from_number"],
                    to=recipient,
                )

            return {"status": "sent", "message_count": len(recipients)}

        except Exception as e:
            return {"status": "failed", "error": str(e)}


class MonitoringService:
    """Comprehensive monitoring service orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.drift_detector = AdvancedDriftDetector(config.get("drift_detection", {}))
        self.performance_monitor = PerformanceMonitor(
            config.get("performance_monitoring", {})
        )
        self.data_quality_monitor = DataQualityMonitor(config.get("data_quality", {}))
        self.alert_manager = AlertManager(config.get("alerting", {}))

        # Data storage clients
        self.storage_clients = {}
        self.logger = structlog.get_logger().bind(service="monitoring")

        # Initialize storage and messaging clients
        self._initialize_clients()

        # Start Prometheus metrics server
        if config.get("prometheus", {}).get("enabled", True):
            self.registry = CollectorRegistry()
            self.metrics_port = config.get("prometheus", {}).get("port", 8081)
            start_http_server(self.metrics_port, registry=self.registry)

    def _initialize_clients(self):
        """Initialize storage and messaging clients"""

        # Redis for caching and fast lookups
        redis_config = self.config.get("redis", {})
        if redis_config.get("enabled", False):
            self.storage_clients["redis"] = aioredis.from_url(
                redis_config["url"], decode_responses=True
            )

        # PostgreSQL for structured monitoring data
        postgres_config = self.config.get("postgres", {})
        if postgres_config.get("enabled", False):
            self.storage_clients["postgres"] = postgres_config

        # Elasticsearch for log analytics
        elasticsearch_config = self.config.get("elasticsearch", {})
        if elasticsearch_config.get("enabled", False):
            self.storage_clients["elasticsearch"] = AsyncElasticsearch(
                elasticsearch_config["hosts"]
            )

        # Kafka for streaming monitoring events
        kafka_config = self.config.get("kafka", {})
        if kafka_config.get("enabled", False):
            self.storage_clients["kafka_producer"] = AIOKafkaProducer(
                bootstrap_servers=kafka_config["bootstrap_servers"]
            )

    async def initialize(self):
        """Initialize monitoring service"""

        # Start Kafka producer if configured
        if "kafka_producer" in self.storage_clients:
            await self.storage_clients["kafka_producer"].start()

        self.logger.info("Monitoring service initialized")

    async def shutdown(self):
        """Shutdown monitoring service"""

        # Close storage clients
        if "redis" in self.storage_clients:
            await self.storage_clients["redis"].close()

        if "elasticsearch" in self.storage_clients:
            await self.storage_clients["elasticsearch"].close()

        if "kafka_producer" in self.storage_clients:
            await self.storage_clients["kafka_producer"].stop()

        self.logger.info("Monitoring service shutdown complete")

    async def monitor_prediction_batch(
        self,
        model_name: str,
        input_data: pd.DataFrame,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonitoringMetrics:
        """Monitor a batch of predictions comprehensively"""

        start_time = datetime.now()

        # Initialize monitoring metrics
        metrics = MonitoringMetrics()

        try:
            # Update Prometheus counters
            PREDICTION_COUNTER.labels(model_name=model_name, version="current").inc(
                len(predictions)
            )

            # Data drift detection
            if (
                hasattr(self.drift_detector, "reference_data")
                and self.drift_detector.reference_data is not None
            ):
                drift_results = await self.drift_detector.detect_drift(input_data)
                metrics.data_drift = drift_results["drift_scores"]

                # Log drift detection results
                await self._log_drift_detection(model_name, drift_results)

            # Performance monitoring
            if actuals is not None:
                performance_results = (
                    await self.performance_monitor.monitor_performance(
                        model_name, predictions, actuals, probabilities
                    )
                )

                metrics.model_performance = {
                    k: v
                    for k, v in performance_results.items()
                    if isinstance(v, (int, float))
                }

            # Data quality assessment
            quality_results = await self.data_quality_monitor.assess_data_quality(
                input_data
            )
            metrics.data_quality = {
                "overall_score": quality_results["overall_score"],
                "completeness_score": quality_results["completeness"]["score"],
                "validity_score": quality_results["validity"]["score"],
                "uniqueness_score": quality_results["uniqueness"]["score"],
            }

            # System resource monitoring
            metrics.system_resources = await self._collect_system_metrics()

            # Update system resource Prometheus metrics
            for resource_type, value in metrics.system_resources.items():
                SYSTEM_RESOURCE_USAGE.labels(resource_type=resource_type).set(value)

            # Calculate prediction latency
            processing_time = (datetime.now() - start_time).total_seconds()
            metrics.prediction_latency = (
                processing_time / len(predictions)
                if len(predictions) > 0
                else processing_time
            )

            # Update Prometheus latency metric
            PREDICTION_LATENCY.labels(model_name=model_name).observe(
                metrics.prediction_latency
            )

            # Store monitoring data
            await self._store_monitoring_data(model_name, metrics, metadata)

            # Evaluate alerts
            triggered_alerts = await self.alert_manager.evaluate_alerts(metrics)

            if triggered_alerts:
                self.logger.warning(
                    "Alerts triggered", count=len(triggered_alerts), model=model_name
                )

            self.logger.info(
                "Monitoring batch completed",
                model=model_name,
                batch_size=len(predictions),
                processing_time=processing_time,
                alerts=len(triggered_alerts),
            )

            return metrics

        except Exception as e:
            self.logger.error("Monitoring batch failed", model=model_name, error=str(e))

            # Update error counter
            PREDICTION_ERRORS.labels(
                model_name=model_name, error_type=type(e).__name__
            ).inc()

            raise e

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics"""

        metrics = {}

        # CPU usage
        metrics["cpu_usage_percent"] = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        metrics["memory_usage_percent"] = memory.percent
        metrics["memory_available_gb"] = memory.available / (1024**3)

        # Disk usage
        disk = psutil.disk_usage("/")
        metrics["disk_usage_percent"] = (disk.used / disk.total) * 100
        metrics["disk_free_gb"] = disk.free / (1024**3)

        # GPU usage (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    metrics[f"gpu_{i}_usage_percent"] = gpu.load * 100
                    metrics[f"gpu_{i}_memory_percent"] = gpu.memoryUtil * 100
                    metrics[f"gpu_{i}_temperature"] = gpu.temperature
        except:
            pass  # GPU monitoring not available

        # Network I/O
        network = psutil.net_io_counters()
        metrics["network_bytes_sent"] = network.bytes_sent
        metrics["network_bytes_recv"] = network.bytes_recv

        return metrics

    async def _log_drift_detection(
        self, model_name: str, drift_results: Dict[str, Any]
    ):
        """Log drift detection results"""

        drift_event = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "event_type": "data_drift",
            "overall_drift": drift_results["overall_drift"],
            "drift_scores": drift_results["drift_scores"],
            "p_values": drift_results["p_values"],
            "feature_drift": drift_results["feature_drift"],
        }

        # Store in Elasticsearch if available
        if "elasticsearch" in self.storage_clients:
            try:
                await self.storage_clients["elasticsearch"].index(
                    index=f"ml-monitoring-drift-{datetime.now().strftime('%Y-%m')}",
                    body=drift_event,
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to log drift to Elasticsearch", error=str(e)
                )

        # Send to Kafka if available
        if "kafka_producer" in self.storage_clients:
            try:
                await self.storage_clients["kafka_producer"].send(
                    "ml-monitoring-events", value=json.dumps(drift_event).encode()
                )
            except Exception as e:
                self.logger.warning("Failed to send drift event to Kafka", error=str(e))

    async def _store_monitoring_data(
        self,
        model_name: str,
        metrics: MonitoringMetrics,
        metadata: Optional[Dict[str, Any]],
    ):
        """Store monitoring data in configured storage systems"""

        monitoring_record = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "metrics": {
                "model_performance": metrics.model_performance,
                "data_quality": metrics.data_quality,
                "system_resources": metrics.system_resources,
                "prediction_latency": metrics.prediction_latency,
            },
            "metadata": metadata or {},
        }

        # Store in PostgreSQL
        if "postgres" in self.storage_clients:
            await self._store_in_postgres(monitoring_record)

        # Store in Redis for fast access
        if "redis" in self.storage_clients:
            await self._store_in_redis(model_name, monitoring_record)

        # Store in Elasticsearch for analytics
        if "elasticsearch" in self.storage_clients:
            await self._store_in_elasticsearch(monitoring_record)

    async def _store_in_postgres(self, record: Dict[str, Any]):
        """Store monitoring data in PostgreSQL"""

        try:
            postgres_config = self.storage_clients["postgres"]

            async with asyncpg.connect(**postgres_config) as conn:
                await conn.execute(
                    """
                    INSERT INTO ml_monitoring (timestamp, model_name, metrics, metadata)
                    VALUES ($1, $2, $3, $4)
                    """,
                    record["timestamp"],
                    record["model_name"],
                    json.dumps(record["metrics"]),
                    json.dumps(record["metadata"]),
                )

        except Exception as e:
            self.logger.warning("Failed to store in PostgreSQL", error=str(e))

    async def _store_in_redis(self, model_name: str, record: Dict[str, Any]):
        """Store monitoring data in Redis"""

        try:
            redis_client = self.storage_clients["redis"]

            # Store latest metrics
            await redis_client.hset(
                f"ml:monitoring:{model_name}:latest",
                mapping={
                    "timestamp": record["timestamp"],
                    "metrics": json.dumps(record["metrics"]),
                },
            )

            # Store in time series (keep last 1000 records)
            await redis_client.lpush(
                f"ml:monitoring:{model_name}:timeseries", json.dumps(record)
            )
            await redis_client.ltrim(f"ml:monitoring:{model_name}:timeseries", 0, 999)

        except Exception as e:
            self.logger.warning("Failed to store in Redis", error=str(e))

    async def _store_in_elasticsearch(self, record: Dict[str, Any]):
        """Store monitoring data in Elasticsearch"""

        try:
            elasticsearch_client = self.storage_clients["elasticsearch"]

            index_name = f"ml-monitoring-{datetime.now().strftime('%Y-%m')}"

            await elasticsearch_client.index(index=index_name, body=record)

        except Exception as e:
            self.logger.warning("Failed to store in Elasticsearch", error=str(e))

    async def get_monitoring_dashboard_data(
        self, model_name: str, hours: int = 24
    ) -> Dict[str, Any]:
        """Get monitoring data for dashboard visualization"""

        dashboard_data = {
            "model_name": model_name,
            "time_range_hours": hours,
            "current_metrics": {},
            "time_series": {},
            "alerts": [],
            "system_health": {},
        }

        # Get current metrics from Redis
        if "redis" in self.storage_clients:
            try:
                redis_client = self.storage_clients["redis"]

                latest_data = await redis_client.hgetall(
                    f"ml:monitoring:{model_name}:latest"
                )
                if latest_data:
                    dashboard_data["current_metrics"] = json.loads(
                        latest_data.get("metrics", "{}")
                    )

                # Get time series data
                timeseries_data = await redis_client.lrange(
                    f"ml:monitoring:{model_name}:timeseries", 0, -1
                )

                if timeseries_data:
                    parsed_data = [json.loads(data) for data in timeseries_data]
                    # Filter by time range
                    cutoff_time = datetime.now() - timedelta(hours=hours)

                    filtered_data = [
                        data
                        for data in parsed_data
                        if datetime.fromisoformat(data["timestamp"]) > cutoff_time
                    ]

                    dashboard_data["time_series"] = filtered_data

            except Exception as e:
                self.logger.warning(
                    "Failed to get dashboard data from Redis", error=str(e)
                )

        # Get recent alerts
        dashboard_data["alerts"] = list(self.alert_manager.alert_history.values())[-10:]

        # System health summary
        dashboard_data["system_health"] = await self._collect_system_metrics()

        return dashboard_data


class MonitoringStage(BasePipelineStage):
    """Pipeline stage for monitoring setup"""

    def __init__(self, service: MonitoringService):
        super().__init__("monitoring", PipelineStage.MONITORING)
        self.service = service

    async def _execute_stage(self, context: ExecutionContext) -> StageResult:
        """Execute monitoring setup stage"""

        try:
            # Initialize monitoring service
            await self.service.initialize()

            # Set up baseline performance if model was trained
            if "best_model_name" in context.artifacts:
                model_name = context.artifacts["best_model_name"]

                # Get training metrics from context
                training_metrics = context.metrics
                baseline_performance = {
                    "accuracy": training_metrics.get("test_accuracy", 0.0),
                    "cross_val_mean": training_metrics.get("cross_val_mean", 0.0),
                    "cross_val_std": training_metrics.get("cross_val_std", 0.0),
                }

                await self.service.performance_monitor.set_baseline_performance(
                    model_name, baseline_performance
                )

            # Set up drift detection baseline if training data is available
            training_data_path = context.artifacts.get("processed_data")
            if training_data_path:
                try:
                    reference_data = pd.read_parquet(training_data_path)
                    # Remove target column if present
                    target_column = context.config.get("model_training", {}).get(
                        "target_column"
                    )
                    if target_column and target_column in reference_data.columns:
                        reference_data = reference_data.drop(columns=[target_column])

                    await self.service.drift_detector.fit(reference_data)

                except Exception as e:
                    self.logger.warning(
                        "Failed to set up drift detection baseline", error=str(e)
                    )

            return StageResult(
                stage=self.stage_type,
                status=ExecutionStatus.SUCCESS,
                output={"monitoring_service": self.service},
                artifacts={
                    "monitoring_endpoint": f"http://localhost:{self.service.metrics_port}/metrics",
                    "dashboard_config": json.dumps(
                        {
                            "prometheus_port": self.service.metrics_port,
                            "enabled_features": list(
                                self.service.storage_clients.keys()
                            ),
                        }
                    ),
                },
                metrics={
                    "monitoring_initialized": 1,
                    "drift_detection_ready": 1
                    if hasattr(self.service.drift_detector, "reference_data")
                    else 0,
                    "storage_clients": len(self.service.storage_clients),
                },
            )

        except Exception as e:
            self.logger.error("Monitoring stage failed", error=str(e))
            raise e

    async def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup monitoring resources"""
        await self.service.shutdown()


# Complete Pipeline Orchestration with all stages
class CompleteMLPipelineOrchestrator:
    """Complete ML pipeline orchestrator with all enterprise features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}
        self.logger = structlog.get_logger().bind(component="complete_orchestrator")

        # Initialize all services
        self._initialize_services()

    def _initialize_services(self):
        """Initialize all pipeline services"""

        from services.data_ingestion.service import DataIngestionFactory
        from services.feature_engineering.service import FeatureEngineeringService
        from services.model_training.service import (
            ModelTrainingService,
            ModelDeploymentService,
        )

        # Data ingestion service
        self.services["data_ingestion"] = DataIngestionFactory.create_stage(
            self.config.get("data_ingestion", {})
        )

        # Feature engineering service
        self.services["feature_engineering"] = FeatureEngineeringService(
            self.config.get("feature_engineering", {})
        )

        # Model training service
        self.services["model_training"] = ModelTrainingService(
            self.config.get("model_training", {})
        )

        # Model deployment service
        self.services["model_deployment"] = ModelDeploymentService(
            self.config.get("model_deployment", {})
        )

        # Monitoring service
        self.services["monitoring"] = MonitoringService(
            self.config.get("monitoring", {})
        )

    async def execute_complete_pipeline(
        self, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the complete ML pipeline end-to-end"""

        run_id = run_id or f"run-{int(datetime.now().timestamp())}"
        experiment_id = self.config.get("experiment_id", "default")

        self.logger.info("Starting complete ML pipeline execution", run_id=run_id)

        # Create execution context
        context = ExecutionContext(
            run_id=run_id,
            experiment_id=experiment_id,
            environment=self.config.get("environment", "production"),
            config=self.config,
        )

        # Execute pipeline stages in order
        from core.pipeline_orchestrator import PipelineOrchestrator
        from services.data_ingestion.service import DataIngestionStage
        from services.feature_engineering.service import FeatureEngineeringStage
        from services.model_training.service import (
            ModelTrainingStage,
            ModelDeploymentStage,
        )

        stages = [
            DataIngestionStage(
                self.services["data_ingestion"].service,
                self.services["data_ingestion"].data_sources,
            ),
            FeatureEngineeringStage(self.services["feature_engineering"]),
            ModelTrainingStage(self.services["model_training"]),
            ModelDeploymentStage(self.services["model_deployment"]),
            MonitoringStage(self.services["monitoring"]),
        ]

        orchestrator = PipelineOrchestrator(
            stages=stages,
            max_parallel=self.config.get("orchestration", {}).get("max_parallel", 1),
            retry_policy=self.config.get("orchestration", {}).get("retry_policy", {}),
        )

        # Execute pipeline
        results = await orchestrator.execute_pipeline(context)

        # Generate pipeline summary
        summary = {
            "run_id": run_id,
            "execution_time": datetime.now().isoformat(),
            "results": {
                stage.value: {
                    "status": result.status.value,
                    "duration": result.duration_seconds,
                    "metrics": result.metrics,
                    "artifacts": result.artifacts,
                }
                for stage, result in results.items()
            },
            "overall_success": all(
                result.status == ExecutionStatus.SUCCESS for result in results.values()
            ),
        }

        self.logger.info(
            "Complete ML pipeline execution finished",
            run_id=run_id,
            success=summary["overall_success"],
            total_stages=len(results),
        )

        return summary


# Configuration example for complete pipeline
COMPLETE_PIPELINE_CONFIG = {
    "environment": "production",
    "experiment_id": "fraud-detection-v2",
    "orchestration": {
        "max_parallel": 1,
        "retry_policy": {"max_retries": 2, "backoff_factor": 2},
    },
    # Include all previous service configurations
    "data_ingestion": {
        # From previous data ingestion config
    },
    "feature_engineering": {
        # From previous feature engineering config
    },
    "model_training": {
        # From previous model training config
    },
    "model_deployment": {
        # From previous deployment config
    },
    "monitoring": {
        "prometheus": {"enabled": True, "port": 8081},
        "drift_detection": {
            "ks_drift": {"enabled": True, "p_val": 0.05},
            "mmd_drift": {"enabled": True, "p_val": 0.05, "n_permutations": 100},
            "tabular_drift": {"enabled": True, "p_val": 0.05},
        },
        "performance_monitoring": {"enabled": True},
        "data_quality": {
            "quality_rules": [
                {"type": "completeness", "threshold": 0.05},
                {
                    "type": "range",
                    "column": "amount",
                    "range": [0, 1000000],
                    "threshold": 0.01,
                },
            ],
            "staleness_threshold_hours": 24,
        },
        "alerting": {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "alerts@company.com",
                "password": "app_password",
                "from_email": "ml-pipeline@company.com",
                "recipients": ["team@company.com"],
            },
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/services/...",
            },
            "alerts": [
                {
                    "name": "model_accuracy_degradation",
                    "metric": "model_performance.accuracy",
                    "threshold": 0.8,
                    "comparison": "lt",
                    "severity": "critical",
                    "channels": ["email", "slack"],
                    "cooldown_minutes": 60,
                },
                {
                    "name": "data_drift_detected",
                    "metric": "data_drift.overall_drift",
                    "threshold": 1,
                    "comparison": "eq",
                    "severity": "warning",
                    "channels": ["slack"],
                    "cooldown_minutes": 30,
                },
                {
                    "name": "high_prediction_latency",
                    "metric": "prediction_latency",
                    "threshold": 1.0,
                    "comparison": "gt",
                    "severity": "warning",
                    "channels": ["email"],
                    "cooldown_minutes": 15,
                },
            ],
        },
        "storage": {
            "redis": {"enabled": True, "url": "redis://localhost:6379"},
            "postgres": {
                "enabled": True,
                "host": "localhost",
                "port": 5432,
                "database": "ml_monitoring",
                "user": "ml_user",
                "password": "secure_password",
            },
            "elasticsearch": {"enabled": True, "hosts": ["http://localhost:9200"]},
            "kafka": {"enabled": True, "bootstrap_servers": ["localhost:9092"]},
        },
    },
}


# Example usage
async def main():
    """Example of running the complete enterprise ML pipeline"""

    # Initialize and run complete pipeline
    pipeline = CompleteMLPipelineOrchestrator(COMPLETE_PIPELINE_CONFIG)

    # Execute pipeline
    results = await pipeline.execute_complete_pipeline("production-run-001")

    print(f"Pipeline execution completed: {results['overall_success']}")
    print(f"Results: {json.dumps(results, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
