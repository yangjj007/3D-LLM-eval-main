"""
Classification metrics for 3D object recognition benchmarks.

Supports: Accuracy, Top-k Accuracy, per-class Accuracy, Confusion Matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute exact-match accuracy."""
    if not predictions:
        return 0.0
    correct = sum(p.strip().lower() == g.strip().lower() for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


def top_k_accuracy(
    prediction_lists: List[List[str]],
    ground_truths: List[str],
    k: int = 5,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        prediction_lists: For each sample, a ranked list of predicted labels.
        ground_truths: Ground truth labels.
        k: Consider the top-k predictions.
    """
    if not prediction_lists:
        return 0.0
    correct = 0
    for preds, gt in zip(prediction_lists, ground_truths):
        top_k_preds = [p.strip().lower() for p in preds[:k]]
        if gt.strip().lower() in top_k_preds:
            correct += 1
    return correct / len(prediction_lists)


def per_class_accuracy(
    predictions: List[str], ground_truths: List[str]
) -> Dict[str, float]:
    """Compute accuracy per unique class."""
    class_correct: Dict[str, int] = {}
    class_total: Dict[str, int] = {}
    for p, g in zip(predictions, ground_truths):
        g_lower = g.strip().lower()
        class_total[g_lower] = class_total.get(g_lower, 0) + 1
        if p.strip().lower() == g_lower:
            class_correct[g_lower] = class_correct.get(g_lower, 0) + 1
    return {
        cls: class_correct.get(cls, 0) / total
        for cls, total in class_total.items()
    }


def confusion_matrix(
    predictions: List[str], ground_truths: List[str]
) -> Dict[str, Dict[str, int]]:
    """Build a confusion matrix as nested dict: gt_label → pred_label → count."""
    matrix: Dict[str, Dict[str, int]] = {}
    for p, g in zip(predictions, ground_truths):
        g_lower = g.strip().lower()
        p_lower = p.strip().lower()
        if g_lower not in matrix:
            matrix[g_lower] = {}
        matrix[g_lower][p_lower] = matrix[g_lower].get(p_lower, 0) + 1
    return matrix


class ClassificationMetrics:
    """Unified interface for classification metrics."""

    @staticmethod
    def compute(
        predictions: List[str],
        ground_truths: List[str],
        metric_names: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        results = {}
        if "accuracy" in metric_names:
            results["accuracy"] = accuracy(predictions, ground_truths)
        if "top_k_accuracy" in metric_names:
            k = kwargs.get("k", 5)
            pred_lists = kwargs.get("prediction_lists", [[p] for p in predictions])
            results[f"top_{k}_accuracy"] = top_k_accuracy(pred_lists, ground_truths, k)
        return results
