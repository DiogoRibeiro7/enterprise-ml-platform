"""Trainer implementations for the model training service."""

__all__ = ["XGBoostTrainer", "LightGBMTrainer", "EnsembleTrainer"]


def __getattr__(name: str):  # pragma: no cover - thin wrapper
    if name == "XGBoostTrainer":
        from .xgboost_trainer import XGBoostTrainer

        return XGBoostTrainer
    if name == "LightGBMTrainer":
        from .lightgbm_trainer import LightGBMTrainer

        return LightGBMTrainer
    if name == "EnsembleTrainer":
        from .ensemble_trainer import EnsembleTrainer

        return EnsembleTrainer
    raise AttributeError(name)
