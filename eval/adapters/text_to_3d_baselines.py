"""Text-to-3D adapters backed by official baseline repositories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from eval.adapters.base import GenResult
from eval.adapters.external_baseline_adapter import (
    ExternalBaselineAdapter,
    ExternalBaselineError,
    OfficialCommandTextTo3DAdapter,
)


class SAR3DAdapter(OfficialCommandTextTo3DAdapter):
    name = "sar3d"
    baseline_name = "sar3d"
    supported_tasks = frozenset({"generation"})
    capabilities = {"batched_generation": False, "generation_produces_mesh": True}

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("test.py")

    def _extra_context(self, prompt: str, sample_id: str, work_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
        prompt_json = work_dir / "test_text.json"
        self._write_json(prompt_json, {"test_promts": [prompt]})
        model_cfg = self._model_cfg(cfg)
        return {
            "prompt_json": str(prompt_json),
            "vqvae_pretrained_path": str(model_cfg.get("vqvae_pretrained_path", "./checkpoint/vqvae-ckpt.pt")),
            "ar_ckpt_path": str(model_cfg.get("ar_ckpt_path", "./checkpoint/text-condition-ckpt.pth")),
            "depth": str(model_cfg.get("depth", 16)),
            "fp16": str(model_cfg.get("fp16", 2)),
            "flexicubes": str(model_cfg.get("flexicubes", False)),
        }

    def _default_command(self, cfg: Dict[str, Any], context: Dict[str, Any]) -> Sequence[str]:
        # Mirrors official test_text.sh while keeping output/prompt paths sample-local.
        return [
            "torchrun",
            "--nproc_per_node=1",
            "--nnodes=1",
            "--rdzv-endpoint=localhost:2980",
            "--rdzv_backend=c10d",
            "test.py",
            "--depth={depth}",
            "--fp16={fp16}",
            "--vqvae_pretrained_path={vqvae_pretrained_path}",
            "--ar_ckpt_path={ar_ckpt_path}",
            "--save_path={work_dir}",
            "--flexicubes={flexicubes}",
            "--text_conditioned=True",
            "--text_json_path={prompt_json}",
        ]


class GaussianCubeAdapter(OfficialCommandTextTo3DAdapter):
    name = "gaussiancube"
    baseline_name = "gaussiancube"
    supported_tasks = frozenset({"generation"})
    capabilities = {"batched_generation": False, "generation_produces_mesh": True}

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("inference.py")

    def _extra_context(self, prompt: str, sample_id: str, work_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
        model_cfg = self._model_cfg(cfg)
        infer_cfg = self._infer_cfg(cfg)
        return {
            "config": str(model_cfg.get("official_config", "configs/objaverse_text_cond.yml")),
            "model_name": str(model_cfg.get("model_name", "objaverse_v1.1")),
            "guidance_scale": str(infer_cfg.get("guidance_scale", 3.5)),
            "num_samples": str(infer_cfg.get("num_samples", 1)),
        }

    def _default_command(self, cfg: Dict[str, Any], context: Dict[str, Any]) -> Sequence[str]:
        command = [
            "{python}",
            "inference.py",
            "--model_name",
            "{model_name}",
            "--exp_name",
            "{work_dir}",
            "--config",
            "{config}",
            "--text",
            "{prompt}",
            "--guidance_scale",
            "{guidance_scale}",
            "--num_samples",
            "{num_samples}",
        ]
        if self._infer_cfg(cfg).get("render_video", False):
            command.append("--render_video")
        return command


class ThreeDTopiaXLAdapter(OfficialCommandTextTo3DAdapter):
    name = "3dtopia_xl"
    baseline_name = "3dtopia_xl"
    supported_tasks = frozenset({"generation"})
    capabilities = {"batched_generation": False, "generation_produces_mesh": True}

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("configs/inference_dit_text.yml")
        self._repo_file("models/conditioner/text.py")

    def _extra_context(self, prompt: str, sample_id: str, work_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
        helper = Path(__file__).resolve().parents[1] / "baselines" / "run_3dtopia_text.py"
        model_cfg = self._model_cfg(cfg)
        infer_cfg = self._infer_cfg(cfg)
        return {
            "helper": str(helper),
            "official_config": str(model_cfg.get("official_config", "configs/inference_dit_text.yml")),
            "checkpoint_path": str(model_cfg.get("checkpoint_path", "./pretrained/scaleup_text_ckpt_backup_fp16.pt")),
            "vae_checkpoint_path": str(model_cfg.get("vae_checkpoint_path", "./pretrained/model_vae_fp16.pt")),
            "text_encoder_path": str(model_cfg.get("text_encoder_path", "./pretrained/open_clip_pytorch_model.bin")),
            "seed": str(infer_cfg.get("seed", 42)),
            "ddim": str(infer_cfg.get("ddim", 25)),
            "cfg_scale": str(infer_cfg.get("cfg", 6)),
            "mc_resolution": str(infer_cfg.get("mc_resolution", 256)),
        }

    def _default_command(self, cfg: Dict[str, Any], context: Dict[str, Any]) -> Sequence[str]:
        return [
            "{python}",
            "{helper}",
            "--repo-dir",
            "{repo_dir}",
            "--config",
            "{official_config}",
            "--prompt",
            "{prompt}",
            "--output-dir",
            "{work_dir}",
            "--checkpoint-path",
            "{checkpoint_path}",
            "--vae-checkpoint-path",
            "{vae_checkpoint_path}",
            "--text-encoder-path",
            "{text_encoder_path}",
            "--seed",
            "{seed}",
            "--ddim",
            "{ddim}",
            "--cfg",
            "{cfg_scale}",
            "--mc-resolution",
            "{mc_resolution}",
        ]


class LGMAdapter(OfficialCommandTextTo3DAdapter):
    name = "lgm"
    baseline_name = "lgm"
    supported_tasks = frozenset({"generation"})
    capabilities = {"batched_generation": False, "generation_produces_mesh": True}

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("app.py")

    def _extra_context(self, prompt: str, sample_id: str, work_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
        helper = Path(__file__).resolve().parents[1] / "baselines" / "run_lgm_text.py"
        model_cfg = self._model_cfg(cfg)
        infer_cfg = self._infer_cfg(cfg)
        return {
            "helper": str(helper),
            "checkpoint_path": str(model_cfg.get("checkpoint_path", "pretrained/model_fp16_fixrot.safetensors")),
            "model_size": str(model_cfg.get("model_size", "big")),
            "seed": str(infer_cfg.get("seed", 42)),
            "steps": str(infer_cfg.get("num_steps", 30)),
            "elevation": str(infer_cfg.get("elevation", 0)),
            "negative_prompt": str(infer_cfg.get("negative_prompt", "")),
        }

    def _default_command(self, cfg: Dict[str, Any], context: Dict[str, Any]) -> Sequence[str]:
        return [
            "{python}",
            "{helper}",
            "--repo-dir",
            "{repo_dir}",
            "--model-size",
            "{model_size}",
            "--resume",
            "{checkpoint_path}",
            "--workspace",
            "{work_dir}",
            "--prompt",
            "{prompt}",
            "--negative-prompt",
            "{negative_prompt}",
            "--seed",
            "{seed}",
            "--num-steps",
            "{steps}",
            "--elevation",
            "{elevation}",
        ]


class TrellisAdapter(ExternalBaselineAdapter):
    name = "trellis"
    baseline_name = "trellis"
    supported_tasks = frozenset({"generation"})
    capabilities = {"batched_generation": False, "generation_produces_mesh": True}

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Any = None
        self.postprocessing_utils: Any = None

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("example_text.py")
        if self._mock_external_enabled(cfg):
            return
        with self._temporary_sys_path(self.repo_dir):
            from trellis.pipelines import TrellisTextTo3DPipeline
            from trellis.utils import postprocessing_utils

            model_id = self._model_cfg(cfg).get("model_id", "microsoft/TRELLIS-text-xlarge")
            self.pipeline = TrellisTextTo3DPipeline.from_pretrained(model_id)
            if str(self.device).startswith("cuda") and hasattr(self.pipeline, "cuda"):
                self.pipeline.cuda()
            self.postprocessing_utils = postprocessing_utils

    def generate_from_text(self, prompts: List[str], sample_ids: List[str], cfg: Dict[str, Any]) -> List[GenResult]:
        if self._mock_external_enabled(cfg):
            return self._mock_generation_results(prompts, sample_ids)
        if self.pipeline is None:
            raise ExternalBaselineError("Trellis pipeline is not loaded")
        infer_cfg = self._infer_cfg(cfg)
        color_cfg = (self._model_cfg(cfg).get("colorization") or {})
        results: list[GenResult] = []
        for prompt, sample_id in zip(prompts, sample_ids):
            outputs = self.pipeline.run(
                prompt,
                seed=int(infer_cfg.get("seed", 1)),
                formats=infer_cfg.get("formats", ["gaussian", "mesh"]),
                sparse_structure_sampler_params=infer_cfg.get("sparse_structure_sampler_params"),
                slat_sampler_params=infer_cfg.get("slat_sampler_params"),
            )
            mesh_obj = outputs.get("mesh", [None])[0] if isinstance(outputs, dict) else None
            glb = None
            if (
                self.postprocessing_utils is not None
                and isinstance(outputs, dict)
                and outputs.get("gaussian")
                and outputs.get("mesh")
            ):
                glb = self.postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    simplify=float(color_cfg.get("simplify", 0.95)),
                    texture_size=int(color_cfg.get("texture_size", 1024)),
                )
            results.append(
                GenResult(
                    raw_response="trellis pipeline output",
                    pred_mesh=mesh_obj if _is_trimesh(mesh_obj) else None,
                    extra={
                        "caption": prompt,
                        "prompt": prompt,
                        "glb_trimesh": glb if _is_trimesh(glb) else None,
                        "official_repo": str(self.repo_dir),
                        "official_entrypoint": self.spec.entrypoint,
                    },
                )
            )
        return results


class ShapEAdapter(ExternalBaselineAdapter):
    name = "shape_e"
    baseline_name = "shape_e"
    supported_tasks = frozenset({"generation"})
    capabilities = {"batched_generation": False, "generation_produces_mesh": True}

    def __init__(self) -> None:
        super().__init__()
        self.xm: Any = None
        self.model: Any = None
        self.diffusion: Any = None
        self.sample_latents: Any = None
        self.decode_latent_mesh: Any = None

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        super().load(cfg, device)
        self._repo_file("shap_e/examples/sample_text_to_3d.ipynb")
        if self._mock_external_enabled(cfg):
            return
        with self._temporary_sys_path(self.repo_dir):
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.diffusion.sample import sample_latents
            from shap_e.models.download import load_config, load_model
            from shap_e.util.notebooks import decode_latent_mesh

            device_str = "cuda" if str(self.device).startswith("cuda") else "cpu"
            self.xm = load_model("transmitter", device=device_str)
            self.model = load_model(self._model_cfg(cfg).get("model_name", "text300M"), device=device_str)
            self.diffusion = diffusion_from_config(load_config("diffusion"))
            self.sample_latents = sample_latents
            self.decode_latent_mesh = decode_latent_mesh

    def generate_from_text(self, prompts: List[str], sample_ids: List[str], cfg: Dict[str, Any]) -> List[GenResult]:
        if self._mock_external_enabled(cfg):
            return self._mock_generation_results(prompts, sample_ids)
        if not all([self.xm, self.model, self.diffusion, self.sample_latents, self.decode_latent_mesh]):
            raise ExternalBaselineError("Shap-E models are not loaded")
        import torch

        results: list[GenResult] = []
        infer_cfg = self._infer_cfg(cfg)
        for prompt, sample_id in zip(prompts, sample_ids):
            latents = self.sample_latents(
                batch_size=1,
                model=self.model,
                diffusion=self.diffusion,
                guidance_scale=float(infer_cfg.get("guidance_scale", 15.0)),
                model_kwargs={"texts": [prompt]},
                progress=bool(infer_cfg.get("progress", False)),
                clip_denoised=True,
                use_fp16=str(self.device).startswith("cuda"),
                use_karras=True,
                karras_steps=int(infer_cfg.get("karras_steps", infer_cfg.get("num_inference_steps", 64))),
                sigma_min=float(infer_cfg.get("sigma_min", 1e-3)),
                sigma_max=float(infer_cfg.get("sigma_max", 160)),
                s_churn=float(infer_cfg.get("s_churn", 0)),
            )
            with torch.no_grad():
                tri_mesh = self.decode_latent_mesh(self.xm, latents[0]).tri_mesh()
            work_dir = self._make_work_dir(cfg, sample_id)
            obj_path = work_dir / f"{self._safe_sample_id(sample_id)}.obj"
            tri_mesh.write_obj(str(obj_path))
            mesh = self._load_mesh_from_path(obj_path)
            results.append(
                GenResult(
                    raw_response=str(obj_path),
                    pred_mesh=mesh,
                    extra={
                        "caption": prompt,
                        "prompt": prompt,
                        "official_repo": str(self.repo_dir),
                        "official_entrypoint": self.spec.entrypoint,
                        "output_path": str(obj_path),
                    },
                )
            )
        return results


def _is_trimesh(value: Any) -> bool:
    try:
        import trimesh
    except Exception:
        return False
    return isinstance(value, trimesh.Trimesh)
