"""Classification metrics have to survive class imbalance.

``evaluate`` reported accuracy alone. On a problem with a 9 percent positive
rate a model that answers "no" to everything scores 0.91, which reads like
success and would win a promotion against a model that actually separates the
classes.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from enterprise_ml_platform.services.model_training.trainers.ensemble_trainer import (
    EnsembleTrainer,
)


@pytest.fixture
def imbalanced() -> tuple[np.ndarray, np.ndarray]:
    """A separable problem with a 10 percent positive rate."""
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.uniform(size=n) < 0.10).astype(int)
    X = rng.normal(size=(n, 3)) + y[:, None] * 2.0
    return X, y


def _trainer(estimator, *, voting: str = "soft") -> EnsembleTrainer:
    """A trainer for one estimator.

    Soft voting by default: a hard-voting ensemble exposes only labels, so
    there are no scores to compute a ROC AUC from.
    """
    return EnsembleTrainer(
        estimators=[("m", estimator)], task="classification", params={"voting": voting}
    )


def test_accuracy_alone_cannot_tell_the_two_models_apart(imbalanced) -> None:
    """The premise: this is why the other metrics are needed."""
    X, y = imbalanced
    majority = _trainer(DummyClassifier(strategy="most_frequent"))
    real = _trainer(LogisticRegression(max_iter=400))

    majority_metrics = majority.evaluate(majority.train(X, y), X, y)
    real_metrics = real.evaluate(real.train(X, y), X, y)

    assert majority_metrics["accuracy"] > 0.85
    assert real_metrics["accuracy"] > 0.85


def test_recall_exposes_the_model_that_never_predicts_the_minority(imbalanced) -> None:
    X, y = imbalanced
    majority = _trainer(DummyClassifier(strategy="most_frequent"))
    real = _trainer(LogisticRegression(max_iter=400))

    majority_metrics = majority.evaluate(majority.train(X, y), X, y)
    real_metrics = real.evaluate(real.train(X, y), X, y)

    assert majority_metrics["recall"] == 0.0
    assert real_metrics["recall"] > 0.5


def test_roc_auc_separates_them(imbalanced) -> None:
    X, y = imbalanced
    majority = _trainer(DummyClassifier(strategy="most_frequent"))
    real = _trainer(LogisticRegression(max_iter=400))

    majority_metrics = majority.evaluate(majority.train(X, y), X, y)
    real_metrics = real.evaluate(real.train(X, y), X, y)

    assert majority_metrics["roc_auc"] == pytest.approx(0.5, abs=0.05)
    assert real_metrics["roc_auc"] > 0.9


def test_majority_class_rate_gives_accuracy_its_baseline(imbalanced) -> None:
    """Accuracy is only readable next to the rate it has to beat."""
    X, y = imbalanced
    trainer = _trainer(DummyClassifier(strategy="most_frequent"))

    metrics = trainer.evaluate(trainer.train(X, y), X, y)

    assert metrics["majority_class_rate"] == pytest.approx(1 - y.mean(), abs=1e-9)
    assert metrics["accuracy"] == pytest.approx(metrics["majority_class_rate"])


def test_every_classification_metric_is_reported(imbalanced) -> None:
    X, y = imbalanced
    trainer = _trainer(LogisticRegression(max_iter=400))

    metrics = trainer.evaluate(trainer.train(X, y), X, y)

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "majority_class_rate",
        "roc_auc",
    }
    assert all(isinstance(v, float) for v in metrics.values())


def test_roc_auc_is_omitted_when_only_one_class_is_present() -> None:
    """It is undefined there, and a fabricated value would be worse."""
    X = np.random.default_rng(1).normal(size=(40, 3))
    y = np.zeros(40, dtype=int)
    trainer = _trainer(DummyClassifier(strategy="most_frequent"))

    metrics = trainer.evaluate(trainer.train(X, y), X, y)

    assert "roc_auc" not in metrics
    assert metrics["majority_class_rate"] == 1.0


def test_hard_voting_reports_no_roc_auc(imbalanced) -> None:
    """A hard-voting ensemble returns labels, so the score is undefined.

    Reporting a number derived from labels would look like a ranking metric
    without being one.
    """
    X, y = imbalanced
    trainer = _trainer(LogisticRegression(max_iter=400), voting="hard")

    metrics = trainer.evaluate(trainer.train(X, y), X, y)

    assert "roc_auc" not in metrics
    assert metrics["recall"] > 0.5


def test_regression_still_reports_r2() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 2))
    y = X[:, 0] * 3.0 + rng.normal(scale=0.1, size=120)
    from sklearn.linear_model import LinearRegression

    trainer = EnsembleTrainer(estimators=[("lr", LinearRegression())], task="regression")

    metrics = trainer.evaluate(trainer.train(X, y), X, y)

    assert set(metrics) == {"r2"}
    assert metrics["r2"] > 0.9
