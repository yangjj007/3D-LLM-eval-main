"""
Inference engine for 3D generation evaluation (text-to-3D).

Processes each sample: prompt → LLM generate → parse mesh tokens →
VQVAE decode → voxel grid. Optionally runs Trellis for mesh refinement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..model_loader import ModelBundle
from ..utils.token_utils import (
    token_to_words,
    parse_mesh_tokens,
    pad_tokens,
    clean_response,
)
from .base_engine import InferenceEngine


class GenerationEngine(InferenceEngine):

    def __init__(self, model_bundle: ModelBundle, config: Dict[str, Any]) -> None:
        super().__init__(model_bundle, config)
        inf_cfg = config.get("inference", {})
        self.max_new_tokens: int = inf_cfg.get("max_new_tokens", 2048)
        self.temperature: float = inf_cfg.get("temperature", 0.7)
        self.top_k: int = inf_cfg.get("top_k", 8192)
        self.top_p: float = inf_cfg.get("top_p", 0.7)
        self.seed: int = inf_cfg.get("seed", 42)
        self.run_trellis: bool = inf_cfg.get("run_trellis", False)
        self.prompt_prefix: str = inf_cfg.get(
            "prompt_prefix",
            "Please generate a 3D asset based on the prompt I provided: ",
        )

    def _generate_tokens(self, prompt: str) -> tuple[str, List[int]]:
        """Run LLM to generate mesh tokens from a text prompt."""
        full_prompt = f"{self.prompt_prefix}{prompt}"
        messages = [
            {"role": "user", "content": [{"type": "text", "text": full_prompt}]}
        ]
        text_input = self.models.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.models.processor(
            text=[text_input], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.models.llm.device)

        eos_token_id = [self.models.tokenizer.eos_token_id, 159858]

        torch.manual_seed(self.seed)
        with torch.no_grad():
            generated_ids = self.models.llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=eos_token_id,
            )

        generated_ids = [
            out[len(inp) :]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        raw_response = self.models.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        mesh_tokens = parse_mesh_tokens(raw_response)
        return raw_response, mesh_tokens

    def _decode_voxels(self, mesh_tokens: List[int]) -> Optional[torch.Tensor]:
        """Decode mesh tokens via VQVAE into a binary voxel grid."""
        if len(mesh_tokens) == 0:
            return None

        padded = pad_tokens(mesh_tokens, 1024)
        encoding_indices = (
            torch.tensor(padded).unsqueeze(0).to(self.models.device)
        )

        with torch.no_grad():
            recon = self.models.vqvae.Decode(encoding_indices)

        # shape: (1, 1, 64, 64, 64) → binarize
        z_s = recon[0].detach().cpu()
        z_s = (z_s > 0).long()
        return z_s

    def _run_trellis(
        self, voxel_grid: torch.Tensor, prompt: str
    ) -> Optional[Dict[str, Any]]:
        """Optionally refine voxels through Trellis pipeline."""
        if self.models.pipeline_text is None:
            return None

        # model_loader restores the previous backend (typically torchsparse) after
        # loading the pipeline. We must switch to spconv for inference because the
        # Trellis text pipeline was built against spconv weights. Without this, the
        # global SPARSE_BACKEND remains torchsparse and SparseTensor wraps torchsparse
        # tensors, which then crash inside spconv's SubMConv3d (_conv_forward checks
        # input.is_quantized which torchsparse SparseTensor does not expose).
        import trellis.modules.sparse as _tsp
        from eval.colorization.sparse_quant_compat import apply_sparse_is_quantized_compat

        apply_sparse_is_quantized_compat()
        prev_backend = _tsp.BACKEND
        _tsp.set_sparse_backend("spconv")

        indices = torch.nonzero(voxel_grid[0] == 1)
        position = (indices.float() + 0.5) / 64 - 0.5
        coords = ((position + 0.5) * 64).int().contiguous()
        ss = torch.zeros(1, 64, 64, 64, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        ss = ss.unsqueeze(0)
        coords_tensor = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
        coords_tensor = coords_tensor.to(self.models.device)

        try:
            with torch.no_grad():
                cond = self.models.pipeline_text.get_cond([prompt[:70]])
                slat = self.models.pipeline_text.sample_slat(cond, coords_tensor)
                outputs = self.models.pipeline_text.decode_slat(
                    slat, ["mesh", "gaussian"]
                )
        finally:
            _tsp.set_sparse_backend(prev_backend)

        return outputs

    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for sample in tqdm(samples, desc="Generation Inference"):
            raw_response, mesh_tokens = self._generate_tokens(sample["prompt"])
            voxel_grid = self._decode_voxels(mesh_tokens)

            result: Dict[str, Any] = {
                "sample_id": sample["sample_id"],
                "prompt": sample["prompt"],
                "num_tokens_generated": len(mesh_tokens),
                "raw_response": raw_response,
                "reference_mesh_path": sample.get("reference_mesh_path"),
            }

            if voxel_grid is not None:
                result["voxel_grid"] = voxel_grid
                result["num_occupied_voxels"] = int(voxel_grid.sum().item())

                if self.run_trellis:
                    trellis_out = self._run_trellis(voxel_grid, sample["prompt"])
                    result["trellis_outputs"] = trellis_out
            else:
                result["voxel_grid"] = None
                result["num_occupied_voxels"] = 0

            results.append(result)
        return results
