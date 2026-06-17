"""
Inference engine for 3D understanding tasks (captioning, QA).

Processes each sample: mesh → VQVAE encode → token string → LLM generate → text response.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from tqdm import tqdm

from ..model_loader import ModelBundle
from ..utils.mesh_processing import load_vertices, positions_to_voxel_tensor
from ..utils.token_utils import token_to_words, clean_response, strip_mesh_tokens
from .base_engine import InferenceEngine


class UnderstandingEngine(InferenceEngine):

    def __init__(self, model_bundle: ModelBundle, config: Dict[str, Any]) -> None:
        super().__init__(model_bundle, config)
        inf_cfg = config.get("inference", {})
        self.max_new_tokens: int = inf_cfg.get("max_new_tokens", 512)
        self.temperature: float = inf_cfg.get("temperature", 0.0)
        self.top_k: int = inf_cfg.get("top_k", 1)
        self.top_p: float = inf_cfg.get("top_p", 1.0)
        self.do_sample: bool = self.temperature > 0

    def _encode_mesh(self, mesh_path: str) -> str:
        """Load mesh → voxelize → VQVAE encode → token string."""
        voxel_positions = load_vertices(mesh_path)
        voxel_tensor = positions_to_voxel_tensor(voxel_positions)
        voxel_tensor = voxel_tensor.to(
            dtype=torch.float32, device=self.models.device
        )

        with torch.no_grad():
            token_indices = self.models.vqvae.Encode(voxel_tensor)
            token_list = token_indices[0].cpu().numpy().tolist()

        return token_to_words(token_list)

    def _generate(self, full_prompt: str) -> str:
        """Run LLM inference on a single prompt."""
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

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": eos_token_id,
        }
        if self.do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = self.models.llm.generate(**inputs, **gen_kwargs)

        generated_ids = [
            out[len(inp) :]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response = self.models.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return response

    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for sample in tqdm(samples, desc="Understanding Inference"):
            mesh_tokens_str = self._encode_mesh(sample["mesh_path"])
            full_prompt = f"{mesh_tokens_str}\n{sample['prompt']}"
            raw_response = self._generate(full_prompt)
            prediction = strip_mesh_tokens(clean_response(raw_response))

            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "prediction": prediction,
                    "ground_truth": sample.get("ground_truth", ""),
                    "ground_truths": sample.get("ground_truths", []),
                    "prompt": sample["prompt"],
                    "raw_response": raw_response,
                }
            )
        return results
