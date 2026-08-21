"""Model training service package with lazy imports."""

__all__ = [
    "ModelTrainingService",
    "ModelConfig",
    "XGBoostTrainer",
    "LightGBMTrainer",
    "EnsembleTrainer",
    "HyperparameterOptimizer",
    "ModelExplainer",
]


def __getattr__(name: str):  # pragma: no cover - thin wrapper
    if name in {"ModelTrainingService", "ModelConfig"}:
        from .service import ModelConfig, ModelTrainingService

        return {
            "ModelTrainingService": ModelTrainingService,
            "ModelConfig": ModelConfig,
        }[name]
    if name == "XGBoostTrainer":
        from .trainers.xgboost_trainer import XGBoostTrainer

        return XGBoostTrainer
    if name == "LightGBMTrainer":
        from .trainers.lightgbm_trainer import LightGBMTrainer

        return LightGBMTrainer
    if name == "EnsembleTrainer":
        from .trainers.ensemble_trainer import EnsembleTrainer

        return EnsembleTrainer
    if name == "HyperparameterOptimizer":
        from .optimization.hyperparameter_optimizer import HyperparameterOptimizer

        return HyperparameterOptimizer
    if name == "ModelExplainer":
        from .explainability.model_explainer import ModelExplainer

        return ModelExplainer
    raise AttributeError(name)
