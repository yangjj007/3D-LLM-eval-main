"""3D object understanding adapters backed by official baseline repositories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval.adapters.base import MeshInput
from eval.adapters.external_baseline_adapter import ExternalBaselineAdapter, ExternalBaselineError
from eval.utils.path_bootstrap import repo_root


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


class MeshRenderedImageBridgeAdapter(ExternalBaselineAdapter):
    """Base for 2D VLM baselines evaluated on rendered 3D object views."""

    def _default_prompt(self, cfg: Dict[str, Any]) -> str:
        return str((cfg.get("data", {}) or {}).get("default_prompt") or "Describe this 3D object.")

    def _image_paths_for_item(self, item: MeshInput, cfg: Dict[str, Any]) -> List[Path]:
        model_cfg = self._model_cfg(cfg)
        sample_map = model_cfg.get("sample_image_map") or {}
        mapped = sample_map.get(item.sample_id)
        if mapped:
            values = mapped if isinstance(mapped, list) else [mapped]
            return [self._path_from_cfg_value(v) for v in values]
        image_dir = model_cfg.get("input_image_dir")
        if image_dir:
            base = self._path_from_cfg_value(image_dir)
            matches: list[Path] = []
            for suffix in (".png", ".jpg", ".jpeg", ".webp"):
                for stem in (self._safe_sample_id(item.sample_id), item.sample_id):
                    candidate = base / f"{stem}{suffix}"
                    if candidate.exists():
                        matches.append(candidate)
            if matches:
                return matches
        if item.mesh_path and Path(item.mesh_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return [Path(item.mesh_path)]
        if not item.mesh_path:
            raise ExternalBaselineError(f"{self.name} sample {item.sample_id} has no mesh or image path")
        return self._render_mesh_views(item, cfg)

    def _render_mesh_views(self, item: MeshInput, cfg: Dict[str, Any]) -> List[Path]:
        import trimesh
        from PIL import Image
        from eval.utils.mesh_multiview_render import render_colored_trimesh_multiview, render_trimesh_multiview_pyvista

        infer_cfg = self._infer_cfg(cfg)
        nviews = int(infer_cfg.get("nviews", 1))
        resolution = int(infer_cfg.get("resolution", 336))
        work_dir = self._make_work_dir(cfg, item.sample_id) / "rendered_views"
        work_dir.mkdir(parents=True, exist_ok=True)
        loaded = trimesh.load(str(item.mesh_path), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            if getattr(loaded, "geometry", None):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            else:
                raise ExternalBaselineError(f"Unable to render mesh for {self.name}: {item.mesh_path}")
        try:
            images = render_colored_trimesh_multiview(loaded, nviews=nviews, resolution=resolution, background="white")
        except Exception:
            images = render_trimesh_multiview_pyvista(loaded, nviews=nviews, resolution=resolution, background="white")
        paths: list[Path] = []
        for idx, arr in enumerate(images[:nviews]):
            path = work_dir / f"view_{idx:02d}.png"
            Image.fromarray(arr).save(path)
            paths.append(path)
        return paths


class InstructBLIP13BAdapter(MeshRenderedImageBridgeAdapter):
    name = "instructblip_13b"
    baseline_name = "instructblip_13b"
    supported_tasks = frozenset({"understanding"})
    capabilities = {"batched_understanding": False}

    def __init__(self) -> None:
        super().__init__()
        self.torch: Any = None
        self.Image: Any = None
        self.model: Any = None
        self.vis_processors: Any = None

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("projects/instructblip/README.md")
        if self._mock_external_enabled(cfg):
            return
        with self._temporary_sys_path(self.repo_dir):
            import torch
            from PIL import Image
            from lavis.models import load_model_and_preprocess

            model_cfg = self._model_cfg(cfg)
            self.model, self.vis_processors, _ = load_model_and_preprocess(
                name=model_cfg.get("model_name", "blip2_vicuna_instruct"),
                model_type=model_cfg.get("model_type", "vicuna13b"),
                is_eval=True,
                device=device,
            )
            self.torch = torch
            self.Image = Image

    def caption_from_shape(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._mock_external_enabled(cfg):
            return self._mock_caption_rows(batch)
        rows: list[Dict[str, Any]] = []
        for item in batch:
            prompt = item.prompt or self._default_prompt(cfg)
            image_paths = self._image_paths_for_item(item, cfg)
            raw_image = self._compose_image_grid(image_paths)
            image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
            gen_kwargs = dict(self._infer_cfg(cfg).get("official_args") or {})
            with self.torch.inference_mode():
                output = self.model.generate({"image": image, "prompt": prompt}, **gen_kwargs)
            prediction = str(output[0] if isinstance(output, list) else output).strip()
            rows.append(_caption_row(item, prompt, prediction, self, {"image_paths": [str(p) for p in image_paths]}))
        return rows

    def _compose_image_grid(self, image_paths: List[Path]) -> Any:
        images = [self.Image.open(path).convert("RGB") for path in image_paths]
        if len(images) == 1:
            return images[0]
        width, height = images[0].size
        canvas = self.Image.new("RGB", (width * len(images), height), "white")
        for idx, image in enumerate(images):
            canvas.paste(image.resize((width, height)), (idx * width, 0))
        return canvas


class LLaVA13BAdapter(MeshRenderedImageBridgeAdapter):
    name = "llava_13b"
    baseline_name = "llava_13b"
    supported_tasks = frozenset({"understanding"})
    capabilities = {"batched_understanding": False}

    def __init__(self) -> None:
        super().__init__()
        self.torch: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.image_processor: Any = None
        self.llava_symbols: Dict[str, Any] = {}

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("llava/eval/run_llava.py")
        if self._mock_external_enabled(cfg):
            return
        with self._temporary_sys_path(self.repo_dir):
            import torch
            from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init

            disable_torch_init()
            model_cfg = self._model_cfg(cfg)
            model_path = model_cfg.get("model_path", "liuhaotian/llava-v1.5-13b")
            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path,
                model_cfg.get("model_base"),
                model_name,
                bool(model_cfg.get("load_8bit", False)),
                bool(model_cfg.get("load_4bit", False)),
                device=str(device),
            )
            self.torch = torch
            self.llava_symbols = {
                "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
                "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
                "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
                "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
                "SeparatorStyle": SeparatorStyle,
                "conv_templates": conv_templates,
                "process_images": process_images,
                "tokenizer_image_token": tokenizer_image_token,
                "model_name": model_name,
            }

    def caption_from_shape(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._mock_external_enabled(cfg):
            return self._mock_caption_rows(batch)
        rows: list[Dict[str, Any]] = []
        for item in batch:
            prompt = item.prompt or self._default_prompt(cfg)
            image_paths = self._image_paths_for_item(item, cfg)
            prediction = self._generate_one(prompt, image_paths, cfg)
            rows.append(_caption_row(item, prompt, prediction, self, {"image_paths": [str(p) for p in image_paths]}))
        return rows

    def _generate_one(self, question: str, image_paths: List[Path], cfg: Dict[str, Any]) -> str:
        from PIL import Image

        s = self.llava_symbols
        images = [Image.open(path).convert("RGB") for path in image_paths]
        image_sizes = [img.size for img in images]
        image_token = s["DEFAULT_IM_START_TOKEN"] + s["DEFAULT_IMAGE_TOKEN"] + s["DEFAULT_IM_END_TOKEN"]
        if getattr(self.model.config, "mm_use_im_start_end", False):
            question = image_token + "\n" + question
        else:
            question = s["DEFAULT_IMAGE_TOKEN"] + "\n" + question
        conv_mode = self._infer_cfg(cfg).get("conv_mode") or _llava_conv_mode(str(s["model_name"]))
        conv = s["conv_templates"][conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images_tensor = s["process_images"](images, self.image_processor, self.model.config).to(
            self.model.device, dtype=self.torch.float16
        )
        input_ids = s["tokenizer_image_token"](
            prompt, self.tokenizer, s["IMAGE_TOKEN_INDEX"], return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        infer_cfg = self._infer_cfg(cfg)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=bool(float(infer_cfg.get("temperature", 0.2)) > 0),
                temperature=float(infer_cfg.get("temperature", 0.2)),
                top_p=infer_cfg.get("top_p"),
                num_beams=int(infer_cfg.get("num_beams", 1)),
                max_new_tokens=int(infer_cfg.get("max_new_tokens", 512)),
                use_cache=True,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


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


def _caption_row(
    item: MeshInput,
    prompt: str,
    prediction: str,
    adapter: ExternalBaselineAdapter,
    debug_extra: Dict[str, Any],
) -> Dict[str, Any]:
    debug = {
        "official_repo": str(adapter.repo_dir),
        "official_entrypoint": adapter.spec.entrypoint,
        "bridge": adapter.spec.entry_kind,
    }
    debug.update(debug_extra)
    return {
        "sample_id": item.sample_id,
        "prompt": prompt,
        "prediction": prediction,
        "raw_response": prediction,
        "ground_truth": item.ground_truth or "",
        "ground_truths": item.ground_truths or ([] if not item.ground_truth else [item.ground_truth]),
        "debug": debug,
    }


def _llava_conv_mode(model_name: str) -> str:
    lower = model_name.lower()
    if "llama-2" in lower:
        return "llava_llama_2"
    if "mistral" in lower:
        return "mistral_instruct"
    if "v1.6-34b" in lower:
        return "chatml_direct"
    if "v1" in lower:
        return "llava_v1"
    if "mpt" in lower:
        return "mpt"
    return "llava_v0"
