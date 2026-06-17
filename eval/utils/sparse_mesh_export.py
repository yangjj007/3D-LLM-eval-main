"""Direct3D-style sparse SDF to white mesh conversion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import trimesh
from skimage import measure

from eval.utils.white_mesh_postprocess import apply_postprocess_from_cfg


def sparse_sdf_to_meshes(
    reconst_x: Any,
    *,
    voxel_resolution: int = 512,
    mc_threshold: float = 0.0,
    postprocess_cfg: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> List[Optional[trimesh.Trimesh]]:
    """Convert decoded sparse signed-SDF tensors to normalized meshes.

    The sparse SDF values are signed and normalised to [-1, 1]:
      negative → inside the surface, positive / background (1.0) → outside.
    Marching Cubes at iso-level 0.0 (zero-crossing) extracts a smooth surface.

    Optional ``postprocess_cfg`` (``model.white_mesh_postprocess``) cleans the mesh
    (largest component, voxel remesh, etc.). Timing breakdown is stored in
    ``mesh.metadata[\"white_mesh_postprocess\"]`` for logging / JSON export.
    """
    sparse_sdf = reconst_x.feats.float()
    sparse_index = reconst_x.coords
    if sparse_index.numel() == 0:
        return []

    batch_size = int(sparse_index[..., 0].max().detach().cpu().item() + 1)
    meshes: List[Optional[trimesh.Trimesh]] = []
    for i in range(batch_size):
        idx = sparse_index[..., 0] == i
        sparse_sdf_i = sparse_sdf[idx].squeeze(-1).detach().cpu()
        sparse_index_i = sparse_index[idx][..., 1:4].detach().cpu().long()
        if sparse_sdf_i.numel() == 0:
            meshes.append(None)
            continue

        valid = ((sparse_index_i >= 0) & (sparse_index_i < voxel_resolution)).all(dim=1)
        if not bool(valid.all()):
            sparse_sdf_i = sparse_sdf_i[valid]
            sparse_index_i = sparse_index_i[valid]
        if sparse_sdf_i.numel() == 0:
            meshes.append(None)
            continue

        sdf = torch.ones((voxel_resolution, voxel_resolution, voxel_resolution), dtype=torch.float32)
        sdf[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = sparse_sdf_i
        try:
            vertices, faces, _, _ = measure.marching_cubes(
                sdf.numpy(),
                level=float(mc_threshold),
                method="lewiner",
            )
        except ValueError:
            meshes.append(None)
            continue
        vertices = vertices / float(voxel_resolution) * 2.0 - 1.0
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        if postprocess_cfg is not None:
            mesh, dbg = apply_postprocess_from_cfg(
                mesh,
                postprocess_cfg,
                device=device,
                log_prefix=f"batch[{i}]",
            )
            mesh.metadata = dict(getattr(mesh, "metadata", None) or {})
            mesh.metadata["white_mesh_postprocess"] = dbg
        meshes.append(mesh)

    return meshes
