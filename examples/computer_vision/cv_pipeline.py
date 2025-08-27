"""End‑to‑end computer vision pipeline example."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .data.image_processor import ImageProcessor
from .models.vision_models import VisionModel
from .training.distributed_trainer import DistributedTrainer
from .inference.batch_inference import BatchInference
from .inference.real_time_inference import RealTimeInference
from .deployment.edge_deployment import EdgeDeployment
from .evaluation.model_evaluator import ModelEvaluator
from .annotation.annotation_tools import AnnotationTools


@dataclass
class ComputerVisionPipeline:
    """Co-ordinates image processing, training, inference and deployment."""

    processor: ImageProcessor = field(default_factory=ImageProcessor)
    model: VisionModel = field(default_factory=VisionModel)
    trainer: DistributedTrainer | None = None
    batch_infer: BatchInference | None = None
    rt_infer: RealTimeInference | None = None
    evaluator: ModelEvaluator = field(default_factory=ModelEvaluator)
    edge: EdgeDeployment | None = None
    annotations: AnnotationTools = field(default_factory=AnnotationTools)

    def __post_init__(self) -> None:
        self.trainer = DistributedTrainer(self.model)
        self.batch_infer = BatchInference(self.model)
        self.rt_infer = RealTimeInference(self.model)
        self.edge = EdgeDeployment(self.model)

    # Training -----------------------------------------------------------------
    def train(self, images: List[np.ndarray], labels: np.ndarray) -> None:
        X = self.processor.batch(images)
        self.trainer.train(X, labels)

    # Evaluation ---------------------------------------------------------------
    def evaluate(self, images: List[np.ndarray], labels: np.ndarray) -> Dict[str, float]:
        X = self.processor.batch(images)
        preds = self.model.predict(X)
        return self.evaluator.classification_metrics(labels, preds)

    # Inference ----------------------------------------------------------------
    def infer_batch(self, images: List[np.ndarray]) -> np.ndarray:
        X = self.processor.batch(images)
        return self.batch_infer.run(X)

    def submit_real_time(self, image: np.ndarray) -> None:
        self.rt_infer.submit(self.processor.preprocess(image))

    def process_next(self) -> np.ndarray | None:
        return self.rt_infer.process_next()

    # Deployment ---------------------------------------------------------------
    def deploy_to_edge(self, device: str) -> str:
        self.edge.quantize()
        return self.edge.deploy(device)
