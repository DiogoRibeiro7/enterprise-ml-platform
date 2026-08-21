import asyncio

import numpy as np

from enterprise_ml_platform.services.monitoring.alerting.rules_engine import Alert
from enterprise_ml_platform.services.monitoring.automated_response import (
    AutomatedResponder,
)
from enterprise_ml_platform.services.monitoring.drift_detection import DriftAnalyzer


def test_advanced_drift_detection_numeric_and_categorical():
    np.random.seed(0)
    reference = {"num": np.random.normal(0, 1, 100), "cat": ["a", "b", "a", "b"]}
    analyzer = DriftAnalyzer()
    analyzer.fit(reference)
    current = {"num": np.random.normal(1, 1, 100), "cat": ["a", "a", "a", "a"]}
    scores = analyzer.check(current)
    assert scores["num"] > 0
    assert scores["cat"] > 0


def test_concept_drift_detection():
    analyzer = DriftAnalyzer()
    reference = {"f": [0.1, 0.2, 0.3]}
    analyzer.fit(reference, confidences=[0.9, 0.95, 0.92])
    current = {"f": [0.1, 0.2, 0.3]}
    scores = analyzer.check(current, confidences=[0.5, 0.55, 0.52])
    assert "concept" in scores and scores["concept"] > 0


async def _collect_actions(alerts):
    retrained = []
    rolled_back = []

    async def retrain(alert):
        retrained.append(alert.name)

    async def rollback(alert):
        rolled_back.append(alert.name)

    responder = AutomatedResponder(retrain=retrain, rollback=rollback)
    await responder.handle(alerts)
    return retrained, rolled_back


def test_automated_responder_triggers():
    alerts = [
        Alert(name="feature_drift", severity="warning", message="drift"),
        Alert(name="performance_drop", severity="critical", message="oops"),
    ]
    retrained, rolled_back = asyncio.run(_collect_actions(alerts))
    assert "feature_drift" in retrained
    assert "performance_drop" in rolled_back
