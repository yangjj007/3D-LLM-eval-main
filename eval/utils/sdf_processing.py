"""Sparse SDF I/O and mesh→SDF for SparseSDFVQVAE evaluation.

Uses vendored ``trellis.utils.mesh_utils.mesh2sparse_sdf`` (signed SDF in [-1,1],
same convention as Med-3D-LLM preprocessing).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def load_sparse_npz(npz_path: str) -> Dict[str, torch.Tensor]:
    data = np.load(npz_path)
    return {
        "sparse_sdf": torch.from_numpy(data["sparse_sdf"]).float(),
        "sparse_index": torch.from_numpy(data["sparse_index"]).long(),
    }


def mesh_to_sparse_sdf_tensors(
    mesh_path: str,
    resolution: int = 512,
    threshold_factor: float = 0.5,
    watertight: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load mesh file and compute sparse SDF (same pipeline as sdf_voxelize)."""
    import trimesh
    from trellis.utils.mesh_utils import mesh2sparse_sdf

    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "geometry") and len(mesh.geometry) > 0:
            mesh = list(mesh.geometry.values())[0]
        else:
            raise ValueError(f"Invalid mesh: {mesh_path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Empty mesh: {mesh_path}")
    sdf_data = mesh2sparse_sdf(
        mesh,
        resolution=resolution,
        threshold_factor=threshold_factor,
        normalize=True,
        scale=0.95,
        watertight=watertight,
    )
    sparse_sdf = torch.from_numpy(sdf_data["sparse_sdf"]).float()
    sparse_index = torch.from_numpy(sdf_data["sparse_index"]).long()
    if sparse_sdf.ndim == 1:
        sparse_sdf = sparse_sdf.unsqueeze(-1)
    return {"sparse_sdf": sparse_sdf, "sparse_index": sparse_index}


def sparse_dict_to_inputs_3d(
    sparse: Dict[str, torch.Tensor], batch_idx: int = 0
) -> Dict[str, torch.Tensor]:
    n = sparse["sparse_sdf"].shape[0]
    bi = torch.full((n,), batch_idx, dtype=torch.long)
    return {
        "sparse_sdf": sparse["sparse_sdf"],
        "sparse_index": sparse["sparse_index"],
        "batch_idx": bi,
    }


def collate_inputs_3d(batch_samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Merge ``inputs_3d`` for batched VAE Encode (batch_idx = 0..B-1 per point)."""
    sparse_sdfs: List[torch.Tensor] = []
    sparse_indices: List[torch.Tensor] = []
    batch_indices: List[torch.Tensor] = []
    for i, b in enumerate(batch_samples):
        d = b["inputs_3d"]
        n = d["sparse_sdf"].shape[0]
        sparse_sdfs.append(d["sparse_sdf"])
        sparse_indices.append(d["sparse_index"])
        batch_indices.append(torch.full((n,), i, dtype=torch.long))
    return {
        "sparse_sdf": torch.cat(sparse_sdfs, dim=0),
        "sparse_index": torch.cat(sparse_indices, dim=0),
        "batch_idx": torch.cat(batch_indices, dim=0),
    }


def get_or_build_sdf_for_sample(
    mesh_path: str,
    sdf_path: Optional[str],
    cache_dir: Optional[str],
    resolution: int,
    threshold_factor: float,
    sample_id: Optional[str] = None,
    watertight: bool = False,
) -> Dict[str, torch.Tensor]:
    if sdf_path and os.path.isfile(sdf_path):
        return load_sparse_npz(sdf_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Match Med-3D-LLM dataset_toolkits/sdf_voxelize.py:
        # output file is {sha256}_r{resolution}.npz. In eval metadata mode,
        # sample_id/file stem is the sha256 identifier.
        sid = str(sample_id or Path(mesh_path).stem)
        cand = Path(cache_dir) / f"{sid}_r{resolution}.npz"
        if cand.is_file():
            return load_sparse_npz(str(cand))
        try:
            t = mesh_to_sparse_sdf_tensors(
                mesh_path,
                resolution,
                threshold_factor,
                watertight=watertight,
            )
        except Exception as exc:
            print(
                f"[sdf_processing] SDF failed: sample_id={sample_id}, "
                f"mesh_path={mesh_path}, resolution={resolution}, error={exc}",
                flush=True,
            )
            raise
        np.savez_compressed(
            str(cand),
            sparse_sdf=t["sparse_sdf"].numpy(),
            sparse_index=t["sparse_index"].numpy(),
            resolution=np.array(resolution),
        )
        return t
    try:
        return mesh_to_sparse_sdf_tensors(
            mesh_path,
            resolution,
            threshold_factor,
            watertight=watertight,
        )
    except Exception as exc:
        print(
            f"[sdf_processing] SDF failed: sample_id={sample_id}, "
            f"mesh_path={mesh_path}, resolution={resolution}, error={exc}",
            flush=True,
        )
        raise
