"""ShapeLLM-Omni adapter: thin wrapper over existing model_loader + inference engines."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .base import GenResult, MeshInput, ModelAdapter, TokenSeq


def _voxel_tensor_to_trimesh(voxel_binary: torch.Tensor) -> Any:
    """Convert (1,64,64,64) or (64,64,64) binary occupancy to watertight-ish mesh."""
    import trimesh

    if voxel_binary.dim() == 4:
        vol = voxel_binary[0].squeeze(0).cpu().numpy()
    else:
        vol = voxel_binary.cpu().numpy()
    vol = (vol > 0).astype(np.float32)
    try:
        from skimage.measure import marching_cubes

        verts, faces, _, _ = marching_cubes(vol, level=0.5)
        verts = verts / 64.0 - 0.5 + 0.5 / 64.0
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception:
        idx = np.argwhere(vol > 0.5)
        if len(idx) == 0:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
        pts = (idx.astype(np.float64) + 0.5) / 64.0 - 0.5
        return trimesh.Trimesh(vertices=pts, faces=np.zeros((0, 3), dtype=np.int64))


class ShapeLLMAdapter(ModelAdapter):
    name = "shapellm"
    supported_tasks = frozenset({"understanding", "vqvae_recon", "generation"})
    capabilities = {
        "batched_understanding": False,
        "batched_vqvae_recon": False,
        "batched_generation": False,
        "generation_produces_mesh": True,
    }

    def __init__(self) -> None:
        self._bundle = None
        self._cfg: Dict[str, Any] = {}
        self._device = torch.device("cpu")

    def load(self, cfg: Dict[str, Any], device: torch.device) -> None:
        from eval.model_loader import load_models

        self._cfg = cfg
        self._device = device
        self._bundle = load_models(cfg)
        if device.type == "cuda" and self._bundle.vqvae is not None:
            self._bundle.vqvae = self._bundle.vqvae.to(device)
        if self._bundle.llm is not None:
            # keep HF device_map behavior from config; only move if single device
            pass

    def unload(self) -> None:
        self._bundle = None

    def encode_shape_to_tokens(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[TokenSeq]:
        from eval.utils.mesh_processing import load_vertices, positions_to_voxel_tensor
        from eval.utils.token_utils import token_to_words

        out: List[TokenSeq] = []
        for m in batch:
            vox = load_vertices(m.mesh_path)
            t = positions_to_voxel_tensor(vox).to(dtype=torch.float32, device=self._device)
            with torch.no_grad():
                enc = self._bundle.vqvae.Encode(t)
            tl = enc[0].cpu().numpy().tolist()
            s = token_to_words(tl)
            out.append(TokenSeq(mesh_token_string=s, token_ids=tl, num_tokens=len(tl)))
        return out

    def caption_from_shape(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        from eval.inference.understanding_engine import UnderstandingEngine

        eng = UnderstandingEngine(self._bundle, cfg)
        samples = [
            {
                "sample_id": m.sample_id,
                "mesh_path": m.mesh_path,
                "prompt": m.prompt or cfg.get("data", {}).get(
                    "default_prompt", "Describe this 3D object."
                ),
                "ground_truth": m.ground_truth or "",
                "ground_truths": m.ground_truths or [m.ground_truth or ""],
            }
            for m in batch
        ]
        rows = eng.run(samples)
        return rows

    def reconstruct_mesh(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        from eval.utils.mesh_processing import load_vertices, positions_to_voxel_tensor

        results: List[Dict[str, Any]] = []
        for m in batch:
            vox = load_vertices(m.mesh_path)
            orig = positions_to_voxel_tensor(vox).to(dtype=torch.float32, device=self._device)
            with torch.no_grad():
                enc = self._bundle.vqvae.Encode(orig)
                recon = self._bundle.vqvae.Decode(enc)
            recon_bin = (recon[0].detach().cpu() > 0).long()
            pred_mesh = _voxel_tensor_to_trimesh(recon_bin)
            import trimesh

            gt_mesh = trimesh.load(m.mesh_path, force="mesh")
            if not isinstance(gt_mesh, trimesh.Trimesh):
                gt_mesh = list(gt_mesh.geometry.values())[0]
            results.append(
                {
                    "sample_id": m.sample_id,
                    "mesh_path": m.mesh_path,
                    "pred_mesh": pred_mesh,
                    "gt_mesh": gt_mesh,
                    "num_tokens": int(enc.shape[1])
                    if hasattr(enc, "shape") and len(enc.shape) >= 2
                    else int(enc.numel()),
                }
            )
        return results

    def generate_from_text(
        self, prompts: List[str], sample_ids: List[str], cfg: Dict[str, Any]
    ) -> List[GenResult]:
        from eval.inference.generation_engine import GenerationEngine

        eng = GenerationEngine(self._bundle, cfg)
        samples = [
            {"sample_id": sid, "prompt": p, "reference_mesh_path": None}
            for sid, p in zip(sample_ids, prompts)
        ]
        raw = eng.run(samples)
        out: List[GenResult] = []
        col = self._cfg.get("model", {}).get("colorization") or {}
        for r in raw:
            vg = r.get("voxel_grid")
            pred_mesh = _voxel_tensor_to_trimesh(vg[0]) if vg is not None else None
            from eval.utils.token_utils import parse_mesh_tokens

            toks = parse_mesh_tokens(r.get("raw_response", ""))
            extra: Dict[str, Any] = {}
            if r.get("trellis_outputs") and str(col.get("enabled", "")).lower() in {"1", "true", "yes", "y", "on"}:
                try:
                    from trellis.utils import postprocessing_utils

                    tre = r["trellis_outputs"]
                    glb = postprocessing_utils.to_glb(
                        tre["gaussian"][0],
                        tre["mesh"][0],
                        simplify=float(col.get("simplify", 0.95)),
                        texture_size=int(col.get("texture_size", 1024)),
                        verbose=False,
                    )
                    extra["glb_trimesh"] = glb
                except Exception as exc:  # noqa: BLE001
                    extra["colorize_error"] = repr(exc)
            out.append(
                GenResult(
                    raw_response=r.get("raw_response", ""),
                    mesh_token_ids=toks,
                    pred_mesh=pred_mesh,
                    voxel_grid=vg,
                    num_occupied_voxels=int(r.get("num_occupied_voxels", 0)),
                    extra=extra,
                )
            )
        return out
