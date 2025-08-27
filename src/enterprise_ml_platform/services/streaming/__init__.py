"""Real-time streaming ML pipeline components."""

from .stream_processor import StreamProcessor, StreamConfig
from .feature_engineering import (
    StreamFeatureEngine,
    TimeWindowAggregator,
    CountWindowAggregator,
    StreamJoiner,
)
from .continuous_learning import (
    OnlineLearner,
    IncrementalTrainer,
    DriftAdapter,
    ModelWarmer,
)

__all__ = [
    "StreamProcessor",
    "StreamConfig",
    "StreamFeatureEngine",
    "TimeWindowAggregator",
    "CountWindowAggregator",
    "StreamJoiner",
    "OnlineLearner",
    "IncrementalTrainer",
    "DriftAdapter",
    "ModelWarmer",
]
