"""
Variable-length 3D tokenization (Morton + FPS). Vendored for eval; no Med repo import.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

DEFAULT_MAX_SAFE_LENGTH = 15000


def morton_sort_indices(coords_xyz: np.ndarray, coord_max: int = 512) -> np.ndarray:
    n = coords_xyz.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    bits = max(1, int(np.ceil(np.log2(coord_max + 1))))
    x = coords_xyz[:, 0].astype(np.int64)
    y = coords_xyz[:, 1].astype(np.int64)
    z = coords_xyz[:, 2].astype(np.int64)
    bit_idx = np.arange(bits, dtype=np.int64)
    x_bits = ((x[:, None] >> bit_idx) & 1) << (3 * bit_idx)
    y_bits = ((y[:, None] >> bit_idx) & 1) << (3 * bit_idx + 1)
    z_bits = ((z[:, None] >> bit_idx) & 1) << (3 * bit_idx + 2)
    codes = x_bits.sum(axis=1) + y_bits.sum(axis=1) + z_bits.sum(axis=1)
    return np.argsort(codes)


def fps_downsample_indices(
    coords_xyz: torch.Tensor,
    num_sample: int,
    start_idx: Optional[int] = None,
) -> torch.Tensor:
    N = coords_xyz.shape[0]
    if N <= num_sample:
        return torch.arange(N, device=coords_xyz.device, dtype=torch.long)
    if start_idx is None:
        start_idx = 0
    pts = coords_xyz.float()
    selected = torch.zeros(num_sample, dtype=torch.long, device=pts.device)
    selected[0] = start_idx
    min_dists = torch.full((N,), float("inf"), device=pts.device)
    for i in range(1, num_sample):
        last_pt = pts[selected[i - 1]].unsqueeze(0)
        dists_to_last = ((pts - last_pt) ** 2).sum(dim=1)
        min_dists = torch.minimum(min_dists, dists_to_last)
        selected[i] = min_dists.argmax()
    return selected
