"""Registry of official baseline repositories used by evaluation adapters.

The registry intentionally separates enabled integrations from audited skips.
Enabled entries have a direct official inference path for one of the two tasks
supported by the eval runner. Skipped entries are documented so users can see
why they are not exposed as runnable adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from eval.utils.path_bootstrap import repo_root


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    adapter: Optional[str]
    task: str
    repo_url: str
    repo_dir_name: str
    status: str
    entry_kind: str
    entrypoint: str
    config_path: Optional[str] = None
    output_patterns: tuple[str, ...] = ()
    skip_reason: Optional[str] = None
    notes: str = ""

    @property
    def default_repo_dir(self) -> Path:
        return repo_root() / "third_party" / "baselines" / self.repo_dir_name

    @property
    def is_enabled(self) -> bool:
        return self.status in {"enabled", "bridged"} and self.adapter is not None


BASELINE_SPECS: Dict[str, BaselineSpec] = {
    "sar3d": BaselineSpec(
        name="sar3d",
        adapter="sar3d",
        task="generation",
        repo_url="https://github.com/cyw-3d/SAR3D.git",
        repo_dir_name="SAR3D",
        status="enabled",
        entry_kind="subprocess",
        entrypoint="test.py",
        config_path="eval/configs/tasks/baselines/sar3d_generation.yaml",
        output_patterns=("**/*.glb", "**/*.obj", "**/*.ply"),
        notes="Official text path is test_text.sh -> test.py --text_conditioned True.",
    ),
    "trellis": BaselineSpec(
        name="trellis",
        adapter="trellis",
        task="generation",
        repo_url="https://github.com/microsoft/TRELLIS.git",
        repo_dir_name="TRELLIS",
        status="enabled",
        entry_kind="python_api",
        entrypoint="trellis.pipelines.TrellisTextTo3DPipeline",
        config_path="eval/configs/tasks/baselines/trellis_generation.yaml",
        output_patterns=("**/*.glb", "**/*.obj", "**/*.ply"),
        notes="Official example_text.py exposes TrellisTextTo3DPipeline.",
    ),
    "gaussiancube": BaselineSpec(
        name="gaussiancube",
        adapter="gaussiancube",
        task="generation",
        repo_url="https://github.com/GaussianCube/GaussianCube.git",
        repo_dir_name="GaussianCube",
        status="enabled",
        entry_kind="subprocess",
        entrypoint="inference.py",
        config_path="eval/configs/tasks/baselines/gaussiancube_generation.yaml",
        output_patterns=("**/*.glb", "**/*.obj", "**/*.ply", "**/*.pt"),
        notes="Official inference.py supports --text for Objaverse text-conditioned generation.",
    ),
    "shape_e": BaselineSpec(
        name="shape_e",
        adapter="shape_e",
        task="generation",
        repo_url="https://github.com/openai/shap-e.git",
        repo_dir_name="shap-e",
        status="enabled",
        entry_kind="python_api",
        entrypoint="shap_e/examples/sample_text_to_3d.ipynb",
        config_path="eval/configs/tasks/baselines/shape_e_generation.yaml",
        output_patterns=("**/*.obj", "**/*.ply"),
        notes="Official notebook uses shap_e APIs for text-conditioned sampling.",
    ),
    "three_d_llm": BaselineSpec(
        name="three_d_llm",
        adapter="three_d_llm",
        task="understanding",
        repo_url="https://github.com/UMass-Embodied-AGI/3D-LLM.git",
        repo_dir_name="3D-LLM",
        status="enabled",
        entry_kind="python_api",
        entrypoint="3DLLM_BLIP2-base/inference.py",
        config_path="eval/configs/tasks/baselines/three_d_llm_understanding.yaml",
        output_patterns=(),
        notes="Official inference.py consumes precomputed 3D features and point grids.",
    ),
    "pointllm_13b": BaselineSpec(
        name="pointllm_13b",
        adapter="pointllm_13b",
        task="understanding",
        repo_url="https://github.com/InternRobotics/PointLLM.git",
        repo_dir_name="PointLLM",
        status="enabled",
        entry_kind="python_api",
        entrypoint="pointllm/eval/eval_objaverse.py",
        config_path="eval/configs/tasks/baselines/pointllm_13b_understanding.yaml",
        output_patterns=(),
        notes="Official eval and chat scripts accept colored point clouds.",
    ),
    "3dtopia_xl": BaselineSpec(
        name="3dtopia_xl",
        adapter="3dtopia_xl",
        task="generation",
        repo_url="https://github.com/3DTopia/3DTopia-XL.git",
        repo_dir_name="3DTopia-XL",
        status="bridged",
        entry_kind="image_conditioned_bridge",
        entrypoint="inference.py",
        config_path="eval/configs/tasks/baselines/3dtopia_xl_generation.yaml",
        output_patterns=("**/*.glb", "**/*.obj", "**/*.ply"),
        notes=(
            "Bridge mode: official text config exists but inference.py still reads images; "
            "adapter runs the official image-conditioned inference with configured proxy images."
        ),
    ),
    "lgm": BaselineSpec(
        name="lgm",
        adapter="lgm",
        task="generation",
        repo_url="https://github.com/3DTopia/LGM.git",
        repo_dir_name="LGM",
        status="bridged",
        entry_kind="gradio_function_bridge",
        entrypoint="app.py",
        config_path="eval/configs/tasks/baselines/lgm_generation.yaml",
        output_patterns=("**/*.ply", "**/*.glb", "**/*.obj"),
        notes=(
            "Bridge mode: executes the official app.py setup up to process() and calls "
            "the official text-to-3D Gradio function without launching the UI."
        ),
    ),
    "instantmesh": BaselineSpec(
        name="instantmesh",
        adapter="instantmesh",
        task="generation",
        repo_url="https://github.com/TencentARC/InstantMesh.git",
        repo_dir_name="InstantMesh",
        status="bridged",
        entry_kind="image_conditioned_bridge",
        entrypoint="run.py",
        config_path="eval/configs/tasks/baselines/instantmesh_generation.yaml",
        output_patterns=("**/*.obj", "**/*.glb", "**/*.ply"),
        notes="Bridge mode: official model is image-to-3D, so adapter runs proxy images configured per sample.",
    ),
    "instructblip_13b": BaselineSpec(
        name="instructblip_13b",
        adapter="instructblip_13b",
        task="understanding",
        repo_url="https://github.com/salesforce/LAVIS.git",
        repo_dir_name="LAVIS",
        status="bridged",
        entry_kind="mesh_render_bridge",
        entrypoint="projects/instructblip",
        config_path="eval/configs/tasks/baselines/instructblip_13b_understanding.yaml",
        notes="Bridge mode: renders 3D meshes to 2D views before calling official LAVIS InstructBLIP.",
    ),
    "llava_13b": BaselineSpec(
        name="llava_13b",
        adapter="llava_13b",
        task="understanding",
        repo_url="https://github.com/haotian-liu/LLaVA.git",
        repo_dir_name="LLaVA",
        status="bridged",
        entry_kind="mesh_render_bridge",
        entrypoint="llava/eval/run_llava.py",
        config_path="eval/configs/tasks/baselines/llava_13b_understanding.yaml",
        notes="Bridge mode: renders 3D meshes to 2D views before calling official LLaVA.",
    ),
}


def get_spec(name: str) -> BaselineSpec:
    try:
        return BASELINE_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(BASELINE_SPECS))
        raise KeyError(f"Unknown baseline {name!r}; known: {known}") from exc


def enabled_specs() -> Iterable[BaselineSpec]:
    return (spec for spec in BASELINE_SPECS.values() if spec.is_enabled)


def skipped_specs() -> Iterable[BaselineSpec]:
    return (spec for spec in BASELINE_SPECS.values() if spec.status == "skipped")
