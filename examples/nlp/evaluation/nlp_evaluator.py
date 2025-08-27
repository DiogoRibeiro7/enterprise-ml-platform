"""Evaluation utilities for NLP tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class EvaluationResult:
    metric: str
    score: float


class NLPEvaluator:
    """Compute basic evaluation metrics for NLP tasks."""

    @staticmethod
    def classification(true_labels: List[str], pred_labels: List[str]) -> EvaluationResult:
        return EvaluationResult(metric="accuracy", score=accuracy_score(true_labels, pred_labels))

    @staticmethod
    def token_classification(true: List[List[str]], pred: List[List[str]]) -> EvaluationResult:
        y_true = [t for seq in true for t in seq]
        y_pred = [p for seq in pred for p in seq]
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        return EvaluationResult(metric="f1", score=f1)


__all__ = ["NLPEvaluator", "EvaluationResult"]
