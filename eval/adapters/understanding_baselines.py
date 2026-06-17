"""3D object understanding adapters backed by official baseline repositories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval.adapters.base import MeshInput
from eval.adapters.external_baseline_adapter import ExternalBaselineAdapter, ExternalBaselineError


class PointLLM13BAdapter(ExternalBaselineAdapter):
    name = "pointllm_13b"
    baseline_name = "pointllm_13b"
    supported_tasks = frozenset({"understanding"})
    capabilities = {"batched_understanding": False}

    def __init__(self) -> None:
        super().__init__()
        self.torch: Any = None
        self.np: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.conv_templates: Any = None
        self.SeparatorStyle: Any = None
        self.KeywordsStoppingCriteria: Any = None

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("pointllm/eval/eval_objaverse.py")
        if self._mock_external_enabled(cfg):
            return
        with self._temporary_sys_path(self.repo_dir):
            import numpy as np
            import torch
            from pointllm.conversation import SeparatorStyle, conv_templates
            from pointllm.model import PointLLMLlamaForCausalLM
            from pointllm.model.utils import KeywordsStoppingCriteria
            from pointllm.utils import disable_torch_init
            from transformers import AutoTokenizer

            disable_torch_init()
            model_name = self._model_cfg(cfg).get("model_name", "RunsenXu/PointLLM_13B_v1.2")
            dtype = _torch_dtype(torch, self._model_cfg(cfg).get("torch_dtype", "bfloat16"))
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = PointLLMLlamaForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=False,
                use_cache=True,
                torch_dtype=dtype,
            )
            self.model.initialize_tokenizer_point_backbone_config_wo_embedding(self.tokenizer)
            self.model.to(device)
            self.model.eval()
            self.torch = torch
            self.np = np
            self.conv_templates = conv_templates
            self.SeparatorStyle = SeparatorStyle
            self.KeywordsStoppingCriteria = KeywordsStoppingCriteria

    def caption_from_shape(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._mock_external_enabled(cfg):
            return self._mock_caption_rows(batch)
        if self.model is None or self.tokenizer is None:
            raise ExternalBaselineError("PointLLM model is not loaded")
        rows: list[Dict[str, Any]] = []
        for item in batch:
            point_cloud = self._load_point_cloud(item, cfg)
            prediction = self._generate_one(item.prompt or self._default_prompt(cfg), point_cloud, cfg)
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "prompt": item.prompt or self._default_prompt(cfg),
                    "prediction": prediction,
                    "raw_response": prediction,
                    "ground_truth": item.ground_truth or "",
                    "ground_truths": item.ground_truths or ([] if not item.ground_truth else [item.ground_truth]),
                    "debug": {"official_repo": str(self.repo_dir), "official_entrypoint": self.spec.entrypoint},
                }
            )
        return rows

    def _default_prompt(self, cfg: Dict[str, Any]) -> str:
        return str((cfg.get("data", {}) or {}).get("default_prompt") or "Caption this 3D model in detail.")

    def _load_point_cloud(self, item: MeshInput, cfg: Dict[str, Any]) -> Any:
        model_cfg = self._model_cfg(cfg)
        pointnum = int(model_cfg.get("pointnum", 8192))
        point_path = self._point_path_for_item(item, cfg)
        if point_path is not None and point_path.suffix.lower() == ".npy":
            arr = self.np.load(str(point_path)).astype("float32")
        elif item.mesh_path:
            arr = self._sample_mesh_points(Path(item.mesh_path), pointnum)
        else:
            raise ExternalBaselineError(f"PointLLM sample {item.sample_id} has no point cloud or mesh path")
        if arr.shape[1] == 3:
            colors = self.np.zeros((arr.shape[0], 3), dtype="float32")
            arr = self.np.concatenate([arr[:, :3], colors], axis=1)
        if arr.shape[0] != pointnum:
            idx = self.np.linspace(0, arr.shape[0] - 1, pointnum).astype("int64")
            arr = arr[idx]
        arr = _normalize_xyz(self.np, arr)
        tensor = self.torch.from_numpy(arr.astype("float32")).unsqueeze(0).to(self.device)
        return tensor.to(getattr(self.model, "dtype", self.torch.bfloat16))

    def _point_path_for_item(self, item: MeshInput, cfg: Dict[str, Any]) -> Path | None:
        sample_map = self._model_cfg(cfg).get("sample_point_map") or {}
        mapped = sample_map.get(item.sample_id)
        if mapped:
            return Path(str(mapped))
        if item.sdf_path and str(item.sdf_path).lower().endswith(".npy"):
            return Path(item.sdf_path)
        if item.mesh_path and str(item.mesh_path).lower().endswith(".npy"):
            return Path(item.mesh_path)
        data_path = self._model_cfg(cfg).get("point_cloud_dir")
        if data_path:
            pointnum = int(self._model_cfg(cfg).get("pointnum", 8192))
            candidate = Path(str(data_path)) / f"{item.sample_id}_{pointnum}.npy"
            if candidate.exists():
                return candidate
        return None

    def _sample_mesh_points(self, mesh_path: Path, pointnum: int) -> Any:
        import trimesh

        loaded = trimesh.load(str(mesh_path), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            if getattr(loaded, "geometry", None):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            else:
                raise ExternalBaselineError(f"Unable to load mesh for PointLLM: {mesh_path}")
        points, face_idx = trimesh.sample.sample_surface(loaded, pointnum)
        colors = self.np.zeros((pointnum, 3), dtype="float32")
        if getattr(loaded.visual, "kind", None) == "face" and hasattr(loaded.visual, "face_colors"):
            face_colors = self.np.asarray(loaded.visual.face_colors[face_idx, :3], dtype="float32") / 255.0
            colors = face_colors
        return self.np.concatenate([points.astype("float32"), colors], axis=1)

    def _generate_one(self, question: str, point_clouds: Any, cfg: Dict[str, Any]) -> str:
        torch = self.torch
        conv = self.conv_templates["vicuna_v1_1"].copy()
        point_cfg = self.model.get_model().point_backbone_config
        point_token_len = point_cfg["point_token_len"]
        patch = point_cfg["default_point_patch_token"]
        start = point_cfg["default_point_start_token"]
        end = point_cfg["default_point_end_token"]
        if point_cfg.get("mm_use_point_start_end", False):
            question = start + patch * point_token_len + end + "\n" + question
        else:
            question = patch * point_token_len + "\n" + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping = self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        infer_cfg = self._infer_cfg(cfg)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                point_clouds=point_clouds,
                do_sample=bool(infer_cfg.get("do_sample", True)),
                temperature=float(infer_cfg.get("temperature", 1.0)),
                top_k=int(infer_cfg.get("top_k", 50)),
                max_length=int(infer_cfg.get("max_length", 2048)),
                top_p=float(infer_cfg.get("top_p", 0.95)),
                stopping_criteria=[stopping],
            )
        input_len = input_ids.shape[1]
        output = self.tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)].strip()
        return output


class ThreeDLLMAdapter(ExternalBaselineAdapter):
    name = "three_d_llm"
    baseline_name = "three_d_llm"
    supported_tasks = frozenset({"understanding"})
    capabilities = {"batched_understanding": False}

    def __init__(self) -> None:
        super().__init__()
        self.torch: Any = None
        self.np: Any = None
        self.model: Any = None
        self.text_processor: Any = None

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        base_dir = self._repo_file("3DLLM_BLIP2-base/inference.py").parent
        if self._mock_external_enabled(cfg):
            return
        with self._temporary_sys_path(base_dir, self.repo_dir):
            import numpy as np
            import torch
            from lavis.common.registry import registry
            from omegaconf import OmegaConf

            model_cfg = OmegaConf.create({"arch": "blip2_t5", "model_type": "pretrain_flant5xl", "use_grad_checkpoint": False})
            self.model = registry.get_model_class(model_cfg.arch).from_pretrained(model_type=model_cfg.model_type)
            ckpt_path = self._checkpoint_path(cfg, base_dir)
            if not ckpt_path.exists():
                raise ExternalBaselineError(f"3D-LLM checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(str(ckpt_path), map_location="cpu")
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.model.eval()
            self.model.to(device)
            processor_cfg = OmegaConf.create({"name": "blip_question", "prompt": ""})
            self.text_processor = registry.get_processor_class(processor_cfg.name).from_config(processor_cfg)
            self.torch = torch
            self.np = np

    def caption_from_shape(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._mock_external_enabled(cfg):
            return self._mock_caption_rows(batch)
        rows: list[Dict[str, Any]] = []
        for item in batch:
            feature_path, points_path = self._feature_paths_for(item, cfg)
            prompt = item.prompt or str((cfg.get("data", {}) or {}).get("default_prompt") or "Describe the 3D scene.")
            prediction = self._generate_one(prompt, feature_path, points_path, cfg)
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "prompt": prompt,
                    "prediction": prediction,
                    "raw_response": prediction,
                    "ground_truth": item.ground_truth or "",
                    "ground_truths": item.ground_truths or ([] if not item.ground_truth else [item.ground_truth]),
                    "debug": {
                        "feature_path": str(feature_path),
                        "points_path": str(points_path),
                        "official_repo": str(self.repo_dir),
                        "official_entrypoint": self.spec.entrypoint,
                    },
                }
            )
        return rows

    def _checkpoint_path(self, cfg: Dict[str, Any], base_dir: Path) -> Path:
        path = self._model_cfg(cfg).get("checkpoint_path", "pretrain_blip2_sam_flant5xl_v1.pth")
        p = Path(str(path))
        return p if p.is_absolute() else base_dir / p

    def _feature_paths_for(self, item: MeshInput, cfg: Dict[str, Any]) -> Tuple[Path, Path]:
        model_cfg = self._model_cfg(cfg)
        sample_map = model_cfg.get("sample_feature_map") or {}
        mapped = sample_map.get(item.sample_id)
        if mapped:
            return Path(mapped["feature_path"]), Path(mapped["points_path"])
        feature_dir = model_cfg.get("feature_dir")
        points_dir = model_cfg.get("points_dir")
        if not feature_dir or not points_dir:
            raise ExternalBaselineError(
                "3D-LLM requires model.feature_dir and model.points_dir, or model.sample_feature_map."
            )
        feature_suffix = str(model_cfg.get("feature_suffix", "_outside.pt"))
        points_suffix = str(model_cfg.get("points_suffix", "_outside.npy"))
        feature_path = Path(str(feature_dir)) / f"{item.sample_id}{feature_suffix}"
        points_path = Path(str(points_dir)) / f"{item.sample_id}{points_suffix}"
        if not feature_path.exists() or not points_path.exists():
            raise ExternalBaselineError(f"3D-LLM feature/points missing for {item.sample_id}: {feature_path}, {points_path}")
        return feature_path, points_path

    def _generate_one(self, prompt: str, feature_path: Path, points_path: Path, cfg: Dict[str, Any]) -> str:
        torch = self.torch
        text_input = self.text_processor(prompt)
        pc_feature = torch.load(str(feature_path), map_location="cpu")
        if isinstance(pc_feature, self.np.ndarray):
            pc_feature = torch.from_numpy(pc_feature)
        pc_feature = pc_feature.to(self.device).unsqueeze(0)
        pc_points = torch.from_numpy(self.np.load(str(points_path))).long().to(self.device).unsqueeze(0)
        model_inputs = {"text_input": text_input, "pc_feat": pc_feature, "pc": pc_points}
        infer_cfg = self._infer_cfg(cfg)
        with torch.inference_mode():
            outputs = self.model.predict_answers(
                samples=model_inputs,
                max_len=int(infer_cfg.get("max_len", 50)),
                length_penalty=float(infer_cfg.get("length_penalty", 1.2)),
                repetition_penalty=float(infer_cfg.get("repetition_penalty", 1.5)),
            )
        return str(outputs[0]).strip()


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    key = str(dtype_name).lower()
    if key == "float16":
        return torch.float16
    if key == "float32":
        return torch.float32
    if key == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _normalize_xyz(np: Any, arr: Any) -> Any:
    xyz = arr[:, :3]
    rest = arr[:, 3:]
    xyz = xyz - np.mean(xyz, axis=0)
    radius = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    if radius > 0:
        xyz = xyz / radius
    return np.concatenate([xyz, rest], axis=1)
