"""
Unified ModelAdapter interface for pluggable 3D evaluation backends.

Each adapter implements task-specific batch methods. Runners call these
with list[MeshInput] / list[str] and expect aligned list[Result].
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh


@dataclass
class MeshInput:
    """One evaluation sample that may reference a mesh and/or sparse SDF."""

    sample_id: str
    mesh_path: str
    sdf_path: Optional[str] = None
    reference_mesh_path: Optional[str] = None
    prompt: Optional[str] = None
    ground_truth: Optional[str] = None
    ground_truths: Optional[List[str]] = None

    @classmethod
    def from_sample_dict(cls, d: Dict[str, Any]) -> "MeshInput":
        gts = d.get("ground_truths")
        if gts is None and d.get("ground_truth") is not None:
            gts = [d["ground_truth"]]
        mp = d.get("mesh_path") or ""
        return cls(
            sample_id=str(d.get("sample_id", "")),
            mesh_path=str(mp),
            sdf_path=d.get("sdf_path"),
            reference_mesh_path=d.get("reference_mesh_path"),
            prompt=d.get("prompt"),
            ground_truth=d.get("ground_truth"),
            ground_truths=gts,
        )


@dataclass
class TokenSeq:
    """Discrete 3D tokens + optional coords for sparse VQVAE decode."""

    mesh_token_string: str
    token_ids: List[int] = field(default_factory=list)
    coords_xyz: Optional[Any] = None  # np.ndarray [N,3] int — for decode only
    num_tokens: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenResult:
    """Text-to-3D generation output."""

    raw_response: str
    mesh_token_ids: List[int] = field(default_factory=list)
    pred_mesh: Optional["trimesh.Trimesh"] = None
    voxel_grid: Optional[Any] = None
    num_occupied_voxels: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class ModelAdapter(abc.ABC):
    """Abstract adapter: one implementation per model family (ShapeLLM, Sparse-SDF, …)."""

    name: str = "base"
    supported_tasks: frozenset = frozenset()

    capabilities: Dict[str, Any] = {
        "batched_understanding": False,
        "batched_vqvae_recon": False,
        "batched_generation": False,
        "generation_produces_mesh": True,
    }

    @abc.abstractmethod
    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        """Load weights onto *device* (torch.device)."""

    def unload(self) -> None:
        """Release GPU memory if needed."""
        pass

    def encode_shape_to_tokens(
        self, batch: List[MeshInput], cfg: Dict[str, Any]
    ) -> List[TokenSeq]:
        """Optional: mesh/SDF → discrete token string (+ coords if applicable)."""
        raise NotImplementedError

    def caption_from_shape(
        self, batch: List[MeshInput], cfg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Understanding: for each MeshInput return dict with at least:
        prediction, raw_response (optional), num_tokens, mesh_token_string (optional).
        """
        raise NotImplementedError

    def reconstruct_mesh(
        self, batch: List[MeshInput], cfg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        VQVAE recon: return dict per sample with keys:
        pred_mesh (trimesh.Trimesh | None), gt_mesh (trimesh | None),
        num_tokens, mesh_path, encoding_extra (optional).
        """
        raise NotImplementedError

    def generate_from_text(
        self, prompts: List[str], sample_ids: List[str], cfg: Dict[str, Any]
    ) -> List[GenResult]:
        """Generation task."""
        raise NotImplementedError
