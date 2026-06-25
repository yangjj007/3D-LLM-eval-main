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


DEFAULT_SDF_RESOLUTION = 256
DEFAULT_SDF_THRESHOLD_FACTOR = 4.0
DEFAULT_SDF_SCALE = 0.95
DEFAULT_SDF_COMPUTE_EDGE_MASK = True
DEFAULT_SDF_SHARP_GRAD_DEV_THRESH = 0.5


def load_sparse_npz(npz_path: str) -> Dict[str, torch.Tensor]:
    data = np.load(npz_path)
    out = {
        "sparse_sdf": torch.from_numpy(data["sparse_sdf"]).float(),
        "sparse_index": torch.from_numpy(data["sparse_index"]).long(),
    }
    if "edge_mask" in data:
        out["edge_mask"] = torch.from_numpy(data["edge_mask"]).bool()
    if "resolution" in data:
        out["resolution"] = torch.tensor(int(np.asarray(data["resolution"]).reshape(-1)[0]))
    if "extra_band_factor" in data:
        out["extra_band_factor"] = torch.tensor(
            float(np.asarray(data["extra_band_factor"]).reshape(-1)[0])
        )
    return out


def _cache_metadata_matches(
    npz_path: str,
    *,
    resolution: int,
    threshold_factor: float,
    require_edge_mask: bool,
) -> bool:
    try:
        with np.load(npz_path, mmap_mode="r") as data:
            if "resolution" not in data or "extra_band_factor" not in data:
                return False
            stored_res = int(np.asarray(data["resolution"]).reshape(-1)[0])
            stored_band = float(np.asarray(data["extra_band_factor"]).reshape(-1)[0])
            if stored_res != int(resolution):
                return False
            if abs(stored_band - float(threshold_factor)) > 1e-6:
                return False
            if require_edge_mask and "edge_mask" not in data:
                return False
            return True
    except Exception:
        return False


def mesh_to_sparse_sdf_tensors(
    mesh_path: str,
    resolution: int = DEFAULT_SDF_RESOLUTION,
    threshold_factor: float = DEFAULT_SDF_THRESHOLD_FACTOR,
    watertight: bool = False,
    compute_edge_mask: bool = DEFAULT_SDF_COMPUTE_EDGE_MASK,
    sharp_grad_dev_thresh: float = DEFAULT_SDF_SHARP_GRAD_DEV_THRESH,
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
        scale=DEFAULT_SDF_SCALE,
        watertight=watertight,
        compute_edge_mask=compute_edge_mask,
        sharp_grad_dev_thresh=sharp_grad_dev_thresh,
    )
    sparse_sdf = torch.from_numpy(sdf_data["sparse_sdf"]).float()
    sparse_index = torch.from_numpy(sdf_data["sparse_index"]).long()
    if sparse_sdf.ndim == 1:
        sparse_sdf = sparse_sdf.unsqueeze(-1)
    out = {
        "sparse_sdf": sparse_sdf,
        "sparse_index": sparse_index,
        "resolution": torch.tensor(int(sdf_data.get("resolution", resolution))),
        "extra_band_factor": torch.tensor(
            float(sdf_data.get("extra_band_factor", threshold_factor))
        ),
    }
    if "edge_mask" in sdf_data:
        out["edge_mask"] = torch.from_numpy(sdf_data["edge_mask"]).bool()
    return out


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
    compute_edge_mask: bool = DEFAULT_SDF_COMPUTE_EDGE_MASK,
    sharp_grad_dev_thresh: float = DEFAULT_SDF_SHARP_GRAD_DEV_THRESH,
) -> Dict[str, torch.Tensor]:
    def _build() -> Dict[str, torch.Tensor]:
        try:
            return mesh_to_sparse_sdf_tensors(
                mesh_path,
                resolution,
                threshold_factor,
                watertight=watertight,
                compute_edge_mask=compute_edge_mask,
                sharp_grad_dev_thresh=sharp_grad_dev_thresh,
            )
        except Exception as exc:
            print(
                f"[sdf_processing] SDF failed: sample_id={sample_id}, "
                f"mesh_path={mesh_path}, resolution={resolution}, "
                f"threshold_factor={threshold_factor}, error={exc}",
                flush=True,
            )
            raise

    def _save(path: Path, tensors: Dict[str, torch.Tensor]) -> None:
        payload = {
            "sparse_sdf": tensors["sparse_sdf"].detach().cpu().numpy(),
            "sparse_index": tensors["sparse_index"].detach().cpu().numpy(),
            "resolution": np.array(int(resolution), dtype=np.int32),
            "extra_band_factor": np.array(float(threshold_factor), dtype=np.float32),
        }
        if "edge_mask" in tensors:
            payload["edge_mask"] = tensors["edge_mask"].detach().cpu().numpy().astype(np.bool_)
        np.savez_compressed(str(path), **payload)

    if sdf_path and os.path.isfile(sdf_path) and _cache_metadata_matches(
        sdf_path,
        resolution=resolution,
        threshold_factor=threshold_factor,
        require_edge_mask=bool(compute_edge_mask),
    ):
        return load_sparse_npz(sdf_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Match Med-3D-LLM dataset_toolkits/sdf_voxelize.py:
        # output file is {sha256}_r{resolution}.npz. In eval metadata mode,
        # sample_id/file stem is the sha256 identifier.
        sid = str(sample_id or Path(mesh_path).stem)
        cand = Path(cache_dir) / f"{sid}_r{resolution}.npz"
        if cand.is_file() and _cache_metadata_matches(
            str(cand),
            resolution=resolution,
            threshold_factor=threshold_factor,
            require_edge_mask=bool(compute_edge_mask),
        ):
            return load_sparse_npz(str(cand))
        t = _build()
        _save(cand, t)
        return t
    return _build()
