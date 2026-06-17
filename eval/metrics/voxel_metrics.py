"""
Voxel-based metrics for VQVAE reconstruction and generation evaluation.

All inputs are binary tensors of shape (..., D, H, W) where 1=occupied, 0=empty.
"""

from __future__ import annotations

from typing import Dict, List

import torch


def voxel_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Intersection over Union of two binary voxel grids.

    Args:
        pred: Predicted binary voxel grid.
        target: Ground-truth binary voxel grid.

    Returns:
        IoU score in [0, 1].
    """
    pred_bool = pred.bool().flatten()
    target_bool = target.bool().flatten()
    intersection = (pred_bool & target_bool).sum().float()
    union = (pred_bool | target_bool).sum().float()
    return (intersection / union).item() if union > 0 else 1.0


def voxel_precision(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of predicted occupied voxels that are correct."""
    pred_bool = pred.bool().flatten()
    target_bool = target.bool().flatten()
    tp = (pred_bool & target_bool).sum().float()
    fp = (pred_bool & ~target_bool).sum().float()
    return (tp / (tp + fp)).item() if (tp + fp) > 0 else 1.0


def voxel_recall(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of ground-truth occupied voxels that are recovered."""
    pred_bool = pred.bool().flatten()
    target_bool = target.bool().flatten()
    tp = (pred_bool & target_bool).sum().float()
    fn = (~pred_bool & target_bool).sum().float()
    return (tp / (tp + fn)).item() if (tp + fn) > 0 else 1.0


def voxel_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Harmonic mean of precision and recall."""
    p = voxel_precision(pred, target)
    r = voxel_recall(pred, target)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class VoxelMetrics:
    """Unified interface for voxel-based metrics."""

    @staticmethod
    def compute(
        results: List[Dict],
        metric_names: List[str],
    ) -> Dict[str, float]:
        """
        Compute voxel metrics over a list of results.

        Each result dict must contain 'original_voxel' and 'reconstructed_voxel'
        as torch.Tensor.

        Returns:
            Dict of metric_name → mean score across all samples.
        """
        metric_fns = {
            "voxel_iou": voxel_iou,
            "voxel_precision": voxel_precision,
            "voxel_recall": voxel_recall,
            "voxel_f1": voxel_f1,
        }

        accumulators = {name: [] for name in metric_names if name in metric_fns}

        for r in results:
            pred = r["reconstructed_voxel"]
            target = r["original_voxel"]
            for name in accumulators:
                accumulators[name].append(metric_fns[name](pred, target))

        return {
            name: sum(vals) / len(vals) if vals else 0.0
            for name, vals in accumulators.items()
        }

    @staticmethod
    def compute_per_sample(
        results: List[Dict],
        metric_names: List[str],
    ) -> List[Dict[str, float]]:
        """Compute voxel metrics per sample."""
        metric_fns = {
            "voxel_iou": voxel_iou,
            "voxel_precision": voxel_precision,
            "voxel_recall": voxel_recall,
            "voxel_f1": voxel_f1,
        }

        per_sample = []
        for r in results:
            pred = r["reconstructed_voxel"]
            target = r["original_voxel"]
            sample_metrics = {}
            for name in metric_names:
                if name in metric_fns:
                    sample_metrics[name] = metric_fns[name](pred, target)
            per_sample.append(sample_metrics)
        return per_sample
