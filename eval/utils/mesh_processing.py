"""
Mesh processing utilities for the evaluation framework.

Consolidates mesh loading, normalization, voxelization, and rotation logic
that was previously duplicated across app.py, main.py, and test_3dvqvae.py.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
import open3d as o3d


def convert_trimesh_to_open3d(trimesh_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(trimesh_mesh.vertices, dtype=np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(trimesh_mesh.faces, dtype=np.int32)
    )
    return o3d_mesh


def rotate_points(
    points: np.ndarray, axis: str = "x", angle_deg: float = 90
) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    if axis == "x":
        R = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])[:3, :3]
    elif axis == "y":
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])[:3, :3]
    elif axis == "z":
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])[:3, :3]
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    return points @ R.T


def load_vertices(filepath: str) -> np.ndarray:
    """
    Load a mesh file, normalize to [-0.5, 0.5], voxelize at 64^3 resolution,
    and apply the ShapeLLM-specific X-axis 90-degree rotation.

    Returns:
        np.ndarray: Voxel center positions, shape (N, 3).
    """
    mesh = trimesh.load(filepath, force="mesh")
    mesh_o3d = convert_trimesh_to_open3d(mesh)
    vertices = np.asarray(mesh_o3d.vertices)

    # Global min/max normalization (matches original ShapeLLM preprocessing)
    min_v = vertices.min()
    max_v = vertices.max()
    vertices_normalized = (vertices - min_v) / (max_v - min_v)
    vertices_centered = vertices_normalized * 1.0 - 0.5
    vertices_centered = np.clip(vertices_centered, -0.5 + 1e-6, 0.5 - 1e-6)

    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices_centered)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh_o3d,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    grid_indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    assert np.all(grid_indices >= 0) and np.all(grid_indices < 64), (
        "Some vertices are out of bounds"
    )

    voxel_centers = (grid_indices + 0.5) / 64 - 0.5
    voxel_rotated = rotate_points(voxel_centers, axis="x", angle_deg=90)
    return voxel_rotated


def voxelize_mesh(filepath: str, resolution: int = 64) -> torch.Tensor:
    """
    Convert a mesh file into a binary voxel grid tensor.

    Returns:
        torch.Tensor: Binary voxel grid, shape (1, 1, R, R, R) where R=resolution.
    """
    voxel_positions = load_vertices(filepath)
    return positions_to_voxel_tensor(voxel_positions, resolution)


def positions_to_voxel_tensor(
    positions: np.ndarray, resolution: int = 64
) -> torch.Tensor:
    """
    Convert voxel center positions (N, 3) in [-0.5, 0.5] to a binary voxel grid.

    Returns:
        torch.Tensor: shape (1, 1, R, R, R).
    """
    coords = ((torch.from_numpy(positions) + 0.5) * resolution).int().contiguous()
    coords = torch.clamp(coords, 0, resolution - 1)
    ss = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float32)
    ss[:, :, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


def mesh_to_voxel_tensor(filepath: str, resolution: int = 64) -> torch.Tensor:
    """Alias for voxelize_mesh."""
    return voxelize_mesh(filepath, resolution)
