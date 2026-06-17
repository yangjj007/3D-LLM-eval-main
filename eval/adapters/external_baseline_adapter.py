"""Shared helpers for adapters that wrap official external baseline code."""

from __future__ import annotations

import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from eval.adapters.base import GenResult, MeshInput, ModelAdapter
from eval.baselines.registry import BaselineSpec, get_spec
from eval.utils.path_bootstrap import repo_root


class ExternalBaselineError(RuntimeError):
    """Raised when an official baseline command/API cannot produce an output."""


class ExternalBaselineAdapter(ModelAdapter):
    """Base class for adapters that call code cloned from official repositories."""

    baseline_name: str = ""

    def __init__(self) -> None:
        self.cfg: Dict[str, Any] = {}
        self.device: Any = None
        self.spec: BaselineSpec = get_spec(self.baseline_name)
        self.repo_dir: Path = self.spec.default_repo_dir

    def load(self, cfg: Dict[str, Any], device: Any) -> None:
        self.cfg = cfg
        self.device = device
        self.repo_dir = self._resolve_repo_dir(cfg)
        self._require_repo_dir()

    def caption_from_shape(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def reconstruct_mesh(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError(f"{self.name} does not support vqvae_recon")

    def generate_from_text(self, prompts: List[str], sample_ids: List[str], cfg: Dict[str, Any]) -> List[GenResult]:
        raise NotImplementedError

    def _resolve_repo_dir(self, cfg: Dict[str, Any]) -> Path:
        model_cfg = cfg.get("model", {}) or {}
        override = (
            model_cfg.get("baseline_repo_dir")
            or model_cfg.get(f"{self.baseline_name}_repo_dir")
            or (cfg.get("baseline_repos", {}) or {}).get(self.baseline_name)
        )
        path = Path(str(override)) if override else self.spec.default_repo_dir
        if not path.is_absolute():
            path = repo_root() / path
        return path.resolve()

    def _require_repo_dir(self) -> None:
        if not self.repo_dir.is_dir():
            raise ExternalBaselineError(
                f"{self.name} official repo not found at {self.repo_dir}. "
                "Run `python -m eval.baselines.clone_official_repos "
                f"{self.baseline_name}` first, or set model.baseline_repo_dir."
            )

    def _repo_file(self, relative: str) -> Path:
        path = self.repo_dir / relative
        if not path.exists():
            raise ExternalBaselineError(f"{self.name} expected official file missing: {path}")
        return path

    def _model_cfg(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return (cfg or self.cfg).get("model", {}) or {}

    def _infer_cfg(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return (cfg or self.cfg).get("inference", {}) or {}

    def _timeout_sec(self, cfg: Dict[str, Any]) -> Optional[int]:
        value = self._infer_cfg(cfg).get("timeout_sec")
        return None if value in (None, "") else int(value)

    def _python_executable(self, cfg: Dict[str, Any]) -> str:
        return str(self._model_cfg(cfg).get("python_executable") or sys.executable)

    def _run_subprocess(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        proc = subprocess.run(
            list(command),
            cwd=str(cwd or self.repo_dir),
            env=merged_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise ExternalBaselineError(
                f"{self.name} official command failed with exit code {proc.returncode}\n"
                f"command: {shlex.join(map(str, command))}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        return proc

    def _command_from_template(self, cfg: Dict[str, Any], context: Dict[str, Any], default: Sequence[str]) -> List[str]:
        template = self._infer_cfg(cfg).get("command_template")
        if not template:
            parts: Iterable[str] = default
        elif isinstance(template, str):
            parts = shlex.split(template)
        else:
            parts = [str(x) for x in template]
        return [str(part).format(**context) for part in parts]

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _find_first_output(self, output_dir: Path, patterns: Optional[Sequence[str]] = None) -> Optional[Path]:
        pats = tuple(patterns or self.spec.output_patterns)
        found: list[Path] = []
        for pattern in pats:
            found.extend(p for p in output_dir.glob(pattern) if p.is_file())
        if not found:
            return None
        return max(found, key=lambda p: p.stat().st_mtime)

    def _load_mesh_from_path(self, path: Optional[Path]) -> Any:
        if path is None or not path.exists() or path.suffix.lower() == ".pt":
            return None
        import trimesh

        loaded = trimesh.load(str(path), force="mesh")
        if isinstance(loaded, trimesh.Trimesh):
            return loaded
        if isinstance(loaded, trimesh.Scene) and loaded.geometry:
            try:
                return trimesh.util.concatenate(tuple(loaded.geometry.values()))
            except Exception:
                return next(iter(loaded.geometry.values()))
        return None

    def _safe_sample_id(self, sample_id: str) -> str:
        return re.sub(r"[^\w.\-]+", "_", str(sample_id))[:160] or "sample"

    def _path_from_cfg_value(self, value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = repo_root() / path
        return path

    def _input_image_for_sample(self, sample_id: str, cfg: Dict[str, Any]) -> Path:
        model_cfg = self._model_cfg(cfg)
        sample_map = model_cfg.get("sample_image_map") or {}
        if sample_id in sample_map:
            path = self._path_from_cfg_value(sample_map[sample_id])
            if path.exists():
                return path
        image_dir = model_cfg.get("input_image_dir")
        if image_dir:
            base = self._path_from_cfg_value(image_dir)
            for suffix in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = base / f"{self._safe_sample_id(sample_id)}{suffix}"
                if candidate.exists():
                    return candidate
                candidate = base / f"{sample_id}{suffix}"
                if candidate.exists():
                    return candidate
        default_image = model_cfg.get("default_input_image")
        if default_image:
            path = self._path_from_cfg_value(default_image)
            if path.exists():
                return path
        raise ExternalBaselineError(
            f"{self.name} needs a proxy input image for sample {sample_id!r}. "
            "Set model.sample_image_map, model.input_image_dir, or model.default_input_image."
        )

    def _copy_input_image(self, source: Path, target_dir: Path, sample_id: str) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower() or ".png"
        target = target_dir / f"{self._safe_sample_id(sample_id)}{suffix}"
        shutil.copyfile(source, target)
        return target

    @contextmanager
    def _temporary_sys_path(self, *paths: Path) -> Iterator[None]:
        values = [str(p) for p in paths]
        old = list(sys.path)
        for value in reversed(values):
            if value in sys.path:
                sys.path.remove(value)
            sys.path.insert(0, value)
        try:
            yield
        finally:
            sys.path[:] = old

    def _make_work_dir(self, cfg: Dict[str, Any], sample_id: str) -> Path:
        root = self._infer_cfg(cfg).get("work_dir")
        if root:
            base = Path(str(root))
            if not base.is_absolute():
                base = repo_root() / base
            path = base / self.name / self._safe_sample_id(sample_id)
            path.mkdir(parents=True, exist_ok=True)
            return path
        return Path(tempfile.mkdtemp(prefix=f"{self.name}_{self._safe_sample_id(sample_id)}_"))

    def _mock_external_enabled(self, cfg: Dict[str, Any]) -> bool:
        mock = self._infer_cfg(cfg).get("mock_external") or {}
        return bool(mock.get("enabled"))

    def _mock_mesh(self) -> Any:
        import trimesh

        return trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    def _mock_generation_results(self, prompts: List[str], sample_ids: List[str]) -> List[GenResult]:
        rows: list[GenResult] = []
        for prompt, sample_id in zip(prompts, sample_ids):
            rows.append(
                GenResult(
                    raw_response=f"mock external baseline output for {sample_id}",
                    pred_mesh=self._mock_mesh(),
                    extra={
                        "caption": prompt,
                        "prompt": prompt,
                        "mock_external": True,
                        "official_repo": str(self.repo_dir),
                        "official_entrypoint": self.spec.entrypoint,
                    },
                )
            )
        return rows

    def _mock_caption_rows(self, batch: List[MeshInput]) -> List[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        for item in batch:
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "prompt": item.prompt or "",
                    "prediction": f"mock external caption for {item.sample_id}",
                    "raw_response": f"mock external caption for {item.sample_id}",
                    "ground_truth": item.ground_truth or "",
                    "ground_truths": item.ground_truths or ([] if not item.ground_truth else [item.ground_truth]),
                    "debug": {
                        "mock_external": True,
                        "official_repo": str(self.repo_dir),
                        "official_entrypoint": self.spec.entrypoint,
                    },
                }
            )
        return rows


class OfficialCommandTextTo3DAdapter(ExternalBaselineAdapter):
    """Subprocess wrapper for official text-to-3D scripts."""

    def _default_command(self, cfg: Dict[str, Any], context: Dict[str, Any]) -> Sequence[str]:
        raise NotImplementedError

    def _extra_context(self, prompt: str, sample_id: str, work_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def generate_from_text(self, prompts: List[str], sample_ids: List[str], cfg: Dict[str, Any]) -> List[GenResult]:
        if self._mock_external_enabled(cfg):
            return self._mock_generation_results(prompts, sample_ids)
        results: list[GenResult] = []
        patterns = tuple(self._infer_cfg(cfg).get("output_patterns") or self.spec.output_patterns)
        for prompt, sample_id in zip(prompts, sample_ids):
            work_dir = self._make_work_dir(cfg, sample_id)
            context = {
                "prompt": prompt,
                "sample_id": sample_id,
                "work_dir": str(work_dir),
                "repo_dir": str(self.repo_dir),
                "python": self._python_executable(cfg),
            }
            context.update(self._extra_context(prompt, sample_id, work_dir, cfg))
            command = self._command_from_template(cfg, context, self._default_command(cfg, context))
            self._run_subprocess(command, cwd=self.repo_dir, timeout=self._timeout_sec(cfg))
            output_path = self._find_first_output(work_dir, patterns)
            mesh = self._load_mesh_from_path(output_path)
            extra = {
                "caption": prompt,
                "prompt": prompt,
                "official_repo": str(self.repo_dir),
                "official_entrypoint": self.spec.entrypoint,
                "output_path": str(output_path) if output_path else None,
                "command": command,
            }
            results.append(GenResult(raw_response=str(output_path or ""), pred_mesh=mesh, extra=extra))
        return results
