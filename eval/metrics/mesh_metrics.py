"""
Mesh-based metrics for 3D generation evaluation.

Supports: Chamfer Distance, Earth Mover's Distance, F-Score.
All metrics operate on point clouds sampled from meshes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch


def _sample_points_from_mesh(
    mesh_path: str, num_points: int = 8192
) -> np.ndarray:
    """Sample points uniformly from a mesh surface."""
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def hausdorff_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """Symmetric Hausdorff distance (max of directed HD)."""
    a = torch.from_numpy(points_a).float()
    b = torch.from_numpy(points_b).float()
    dist = torch.cdist(a, b, p=2)
    h_ab = dist.min(dim=1).values.max().item()
    h_ba = dist.min(dim=0).values.max().item()
    return float(max(h_ab, h_ba))


def chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    bidirectional: bool = True,
) -> float:
    """
    Compute Chamfer Distance between two point clouds.

    CD = (1/|A|) * sum_{a in A} min_{b in B} ||a - b||^2
       + (1/|B|) * sum_{b in B} min_{a in A} ||b - a||^2

    Args:
        points_a: shape (N, 3)
        points_b: shape (M, 3)
        bidirectional: If True, compute both directions and sum.

    Returns:
        Chamfer Distance (L2 squared, averaged).
    """
    a = torch.from_numpy(points_a).float()
    b = torch.from_numpy(points_b).float()

    # (N, M) pairwise distances
    dist_matrix = torch.cdist(a, b, p=2)  # L2 distance

    # A → B
    min_a_to_b = dist_matrix.min(dim=1).values.pow(2).mean().item()

    if bidirectional:
        # B → A
        min_b_to_a = dist_matrix.min(dim=0).values.pow(2).mean().item()
        return min_a_to_b + min_b_to_a
    return min_a_to_b


def earth_mover_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """
    Approximate Earth Mover's Distance using scipy's linear_sum_assignment.

    Both point clouds must have the same number of points.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
    except ImportError:
        warnings.warn("scipy not available for EMD computation.")
        return -1.0

    n = min(len(points_a), len(points_b))
    a = points_a[:n]
    b = points_b[:n]

    cost_matrix = cdist(a, b, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].mean()


def f_score(
    points_a: np.ndarray,
    points_b: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Compute F-Score at a given distance threshold.

    Returns:
        Tuple of (f_score, precision, recall).
    """
    a = torch.from_numpy(points_a).float()
    b = torch.from_numpy(points_b).float()

    dist_matrix = torch.cdist(a, b, p=2)

    # A → B: for each point in A, nearest distance in B
    min_a = dist_matrix.min(dim=1).values
    # B → A
    min_b = dist_matrix.min(dim=0).values

    precision = (min_a < threshold).float().mean().item()
    recall = (min_b < threshold).float().mean().item()
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f, precision, recall


class MeshMetrics:
    """Unified interface for mesh-based metrics."""

    @staticmethod
    def compute_reconstruction(
        results: List[Dict],
        metric_names: List[str],
        num_sample_points: int = 8192,
        f_score_threshold: float = 0.01,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Per-sample mesh reconstruction: each result must have ``pred_mesh`` and ``gt_mesh``
        (trimesh.Trimesh) or paths in ``pred_mesh_path`` / ``gt_mesh_path``.
        Returns (aggregate dict, list of per-sample metric dicts).
        """
        per_sample: List[Dict[str, float]] = []
        accum: Dict[str, List[float]] = {}

        for r in results:
            pm = r.get("pred_mesh")
            gm = r.get("gt_mesh")
            if pm is None or gm is None:
                continue
            import trimesh

            if not isinstance(pm, trimesh.Trimesh):
                continue
            if not isinstance(gm, trimesh.Trimesh):
                continue
            pp, _ = trimesh.sample.sample_surface(pm, num_sample_points)
            gp, _ = trimesh.sample.sample_surface(gm, num_sample_points)
            row: Dict[str, float] = {}
            if "chamfer_distance" in metric_names:
                cd = chamfer_distance(pp, gp)
                row["chamfer_distance"] = cd
                accum.setdefault("chamfer_distance", []).append(cd)
            if "hausdorff_distance" in metric_names:
                hd = hausdorff_distance(pp, gp)
                row["hausdorff_distance"] = hd
                accum.setdefault("hausdorff_distance", []).append(hd)
            if "f_score" in metric_names:
                f, prec, rec = f_score(pp, gp, threshold=f_score_threshold)
                row["f_score"] = f
                row["f_score_precision"] = prec
                row["f_score_recall"] = rec
                accum.setdefault("f_score", []).append(f)
            per_sample.append(row)

        agg = {
            k: float(sum(v) / len(v)) if v else 0.0
            for k, v in accum.items()
        }
        return agg, per_sample

    @staticmethod
    def compute(
        results: List[Dict],
        metric_names: List[str],
        num_sample_points: int = 8192,
        f_score_thresholds: List[float] = [0.005, 0.01, 0.02],
    ) -> Dict[str, float]:
        """
        Compute mesh metrics over generation results that have reference meshes.

        Each result dict should contain:
            - 'reference_mesh_path': path to ground truth mesh
            - One of: 'generated_mesh_path' or 'voxel_grid' (for point extraction)

        Returns:
            Dict of metric_name → mean score.
        """
        accumulators: Dict[str, List[float]] = {}

        for r in results:
            ref_path = r.get("reference_mesh_path")
            if ref_path is None:
                continue

            ref_points = _sample_points_from_mesh(ref_path, num_sample_points)

            if "generated_mesh_path" in r and r["generated_mesh_path"] is not None:
                pred_points = _sample_points_from_mesh(
                    r["generated_mesh_path"], num_sample_points
                )
            elif "voxel_grid" in r and r["voxel_grid"] is not None:
                indices = torch.nonzero(r["voxel_grid"][0] == 1).float()
                pred_points = ((indices + 0.5) / 64 - 0.5).numpy()
            else:
                continue

            if "chamfer_distance" in metric_names:
                cd = chamfer_distance(pred_points, ref_points)
                accumulators.setdefault("chamfer_distance", []).append(cd)

            if "emd" in metric_names:
                emd = earth_mover_distance(pred_points, ref_points)
                accumulators.setdefault("emd", []).append(emd)

            if "f_score" in metric_names:
                for tau in f_score_thresholds:
                    f, p, rec = f_score(pred_points, ref_points, threshold=tau)
                    accumulators.setdefault(f"f_score@{tau}", []).append(f)

        return {
            name: sum(vals) / len(vals) if vals else 0.0
            for name, vals in accumulators.items()
        }
