"""Morton + FPS mesh token string from batched VAE encoding_indices (no debug prints)."""

from __future__ import annotations

from typing import Any, List

import numpy as np


def variable_length_sequence_to_mesh_token_string(indices: np.ndarray) -> str:
    if len(indices) == 0:
        return "<mesh_start><mesh_end>"
    vals = indices.tolist()
    return "<mesh_start>" + "".join(f"<mesh_{v}>" for v in vals) + "<mesh_end>"


def indices_to_sorted_token_ids(
    encoding_indices: Any,
    coords: Any,
    batch_idx: int,
    max_safe_length: int,
    coord_max: int,
) -> np.ndarray:
    from eval.sparse_backend.variable_length_3d import (
        fps_downsample_indices,
        morton_sort_indices,
    )

    mask = coords[:, 0] == batch_idx
    if not mask.any():
        return np.zeros((0,), dtype=np.int64)
    idx_b = encoding_indices[mask].long().squeeze(-1)
    xyz_b = coords[mask][:, 1:4]
    n_raw = int(idx_b.shape[0])
    if n_raw > max_safe_length:
        fps_idx = fps_downsample_indices(xyz_b.float(), max_safe_length)
        idx_b = idx_b[fps_idx]
        xyz_b = xyz_b[fps_idx]
    idx_np = idx_b.detach().cpu().numpy().astype(np.int64)
    xyz_np = xyz_b.detach().cpu().numpy().astype(np.int64)
    order = morton_sort_indices(xyz_np, coord_max=coord_max)
    return idx_np[order]


def batch_to_mesh_strings(
    encoding_indices_obj: Any,
    batch_size: int,
    max_safe_length: int,
    coord_max: int,
) -> List[str]:
    feats = encoding_indices_obj.feats.squeeze(-1).long()
    coords = encoding_indices_obj.coords
    out: List[str] = []
    for b in range(batch_size):
        arr = indices_to_sorted_token_ids(feats, coords, b, max_safe_length, coord_max)
        out.append(variable_length_sequence_to_mesh_token_string(arr))
    return out
