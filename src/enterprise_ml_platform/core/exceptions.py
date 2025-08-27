"""Custom exception hierarchy for the Enterprise ML Platform."""

from __future__ import annotations

from typing import Optional


class MLPlatformError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None) -> None:
        """Initialize the error.

        Args:
            message: Human readable error message.
            cause: Optional original exception that triggered this error.
        """
        super().__init__(message)
        self.cause = cause


class ConfigurationError(MLPlatformError):
    """Raised when configuration loading or validation fails."""


class ServiceError(MLPlatformError):
    """Base class for service-related errors."""


class DataIngestionError(ServiceError):
    """Raised when data ingestion fails."""


class FeatureEngineeringError(ServiceError):
    """Raised when feature engineering operations fail."""


class ModelTrainingError(ServiceError):
    """Raised when model training or evaluation fails."""
