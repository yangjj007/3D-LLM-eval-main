"""
Inference engine for VQVAE reconstruction quality evaluation.

Processes each sample: mesh → voxelize → VQVAE encode → decode → binarize.
Outputs both original and reconstructed voxel grids for metric computation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from tqdm import tqdm

from ..model_loader import ModelBundle
from ..utils.mesh_processing import load_vertices, positions_to_voxel_tensor
from .base_engine import InferenceEngine


class VQVAEEngine(InferenceEngine):

    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for sample in tqdm(samples, desc="VQVAE Reconstruction"):
            voxel_positions = load_vertices(sample["mesh_path"])
            original_tensor = positions_to_voxel_tensor(voxel_positions)
            # original_tensor shape: (1, 1, 64, 64, 64)

            input_tensor = original_tensor.to(
                dtype=torch.float32, device=self.models.device
            )

            with torch.no_grad():
                encoding_indices = self.models.vqvae.Encode(input_tensor)
                recon = self.models.vqvae.Decode(encoding_indices)

            # Binarize reconstruction
            recon_binary = (recon[0].detach().cpu() > 0).long()
            original_binary = original_tensor[0].long()

            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "mesh_path": sample["mesh_path"],
                    "original_voxel": original_binary,   # (1, 64, 64, 64)
                    "reconstructed_voxel": recon_binary,  # (1, 64, 64, 64)
                    "num_tokens": encoding_indices.shape[1],
                    "original_occupied": int(original_binary.sum().item()),
                    "reconstructed_occupied": int(recon_binary.sum().item()),
                }
            )
        return results
