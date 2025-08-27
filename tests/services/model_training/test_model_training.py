import asyncio
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from enterprise_ml_platform.services.model_training import (
    ModelConfig,
    ModelTrainingService,
)


def test_training_service_with_voting_ensemble():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    config = ModelConfig(
        algorithm="ensemble",
        ensemble={
            "estimators": [
                ("lr", LogisticRegression(max_iter=100)),
                ("dt", DecisionTreeClassifier(max_depth=3)),
            ],
            "task": "classification",
            "method": "voting",
        },
    )
    service = ModelTrainingService()
    model, metrics = asyncio.run(service.train(X, y))
    assert metrics["accuracy"] > 0
