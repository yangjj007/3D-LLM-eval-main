"""
Render-based metrics: Inception-V3 FD / KD (Gaussian-kernel MMD²), CLIP score, PSNR, and SSIM on mesh renders.

PSNR / SSIM compare paired reference vs generated RGB views (same viewpoint index); reported values are
means over all sample–view pairs (same convention as :meth:`RenderMetrics.compute`).

Default **pooled** mode: concatenate all samples' multi-view renders, then compute
Frechet distance and unbiased MMD² with an RBF (Gaussian) kernel on Inception features.
Optional **per_sample_mean** (``metrics_config.render.inception_aggregate``): compute FD/KD
per 3D sample (that sample's renders only), then average across samples — closer to
paper tables that report a distribution summarized by the mean.
``kd_inception`` reports ``100 * MMD²`` (same ×10² scaling idea as common paper tables).

``clip_score`` is reported on the common paper scale ``100 * mean_cosine_similarity`` in
``[-100, 100]`` (invalid / skipped samples still return ``-1.0``).
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

RENDER_METRIC_NAMES = frozenset(
    {"fd_inception", "kd_inception", "clip_score", "psnr", "ssim"}
)


def _cuda_synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _truthy_env(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _cuda_device_index(device: torch.device) -> Optional[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _torch_cuda_memory_stats(device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {"torch_cuda_available": bool(torch.cuda.is_available())}
    if device.type != "cuda" or not torch.cuda.is_available():
        return out
    idx = _cuda_device_index(device)
    if idx is None:
        return out
    try:
        with torch.cuda.device(idx):
            out["device_index"] = idx
            out["memory_allocated_mib"] = round(
                float(torch.cuda.memory_allocated()) / (1024.0**2), 3
            )
            out["memory_reserved_mib"] = round(
                float(torch.cuda.memory_reserved()) / (1024.0**2), 3
            )
            out["max_memory_allocated_mib"] = round(
                float(torch.cuda.max_memory_allocated()) / (1024.0**2), 3
            )
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _nvidia_smi_query_gpu(gpu_index: int) -> Optional[Dict[str, Any]]:
    """单次查询 ``nvidia-smi``；失败时返回 ``None``（未安装驱动/无 GPU 等）。"""
    cmd = [
        "nvidia-smi",
        f"--id={int(gpu_index)}",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if cp.returncode != 0 or not (cp.stdout or "").strip():
            return None
        line = cp.stdout.strip().splitlines()[-1]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None
        return {
            "utilization_gpu_pct": int(float(parts[0])) if parts[0] else None,
            "utilization_memory_pct": int(float(parts[1])) if len(parts) > 1 and parts[1] else None,
            "memory_used_mib": int(float(parts[2])) if len(parts) > 2 and parts[2] else None,
            "memory_total_mib": int(float(parts[3])) if len(parts) > 3 and parts[3] else None,
            "temperature_c": float(parts[4]) if len(parts) > 4 and parts[4] else None,
            "power_draw_w": float(parts[5]) if len(parts) > 5 and parts[5] else None,
            "raw_csv_line": line,
        }
    except Exception:
        return None


class _GpuUtilPollThread:
    """后台轮询 ``nvidia-smi``，用于观察指标计算期间的 GPU 利用率曲线（有开销）。"""

    def __init__(self, gpu_index: int, interval_sec: float) -> None:
        self._gpu_index = int(gpu_index)
        self._interval_sec = max(0.05, float(interval_sec))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, Any]] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="render_metrics_gpu_poll", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self._gpu_index}",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.wait(self._interval_sec):
            try:
                cp = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=False,
                )
                if cp.returncode != 0 or not (cp.stdout or "").strip():
                    continue
                line = cp.stdout.strip().splitlines()[-1]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                self.samples.append(
                    {
                        "t_wall": time.perf_counter(),
                        "utilization_gpu_pct": int(float(parts[0])),
                        "utilization_memory_pct": int(float(parts[1])),
                        "memory_used_mib": int(float(parts[2])),
                        "memory_total_mib": int(float(parts[3])),
                    }
                )
            except Exception:
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=12.0)

    def summary(self) -> Optional[Dict[str, Any]]:
        if not self.samples:
            return None
        utils = [int(s["utilization_gpu_pct"]) for s in self.samples if "utilization_gpu_pct" in s]
        mems = [int(s["memory_used_mib"]) for s in self.samples if "memory_used_mib" in s]
        if not utils:
            return {"sample_count": len(self.samples)}
        return {
            "sample_count": len(self.samples),
            "utilization_gpu_pct_min": min(utils),
            "utilization_gpu_pct_max": max(utils),
            "utilization_gpu_pct_mean": round(sum(utils) / len(utils), 2),
            "memory_used_mib_max": max(mems) if mems else None,
        }


def _rank_timings_sec(timings: Dict[str, float], *, top: int = 12) -> List[Tuple[str, float]]:
    items = [(k, float(v)) for k, v in timings.items() if isinstance(v, (int, float)) and float(v) > 0]
    items.sort(key=lambda kv: -kv[1])
    return items[: int(top)]


def _first_module_tensor_device(module: Any) -> Optional[torch.device]:
    """Torchmetrics may store the actual model in params or buffers."""
    for attr in ("parameters", "buffers"):
        try:
            for t in getattr(module, attr)():
                if torch.is_tensor(t):
                    return t.device
        except Exception:
            continue
    return None


def _metric_device_summary(module: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "param_devices": [],
        "buffer_devices": [],
        "first_tensor_device": None,
    }
    for attr, key in (("parameters", "param_devices"), ("buffers", "buffer_devices")):
        seen = set()
        try:
            for t in getattr(module, attr)():
                if torch.is_tensor(t):
                    seen.add(str(t.device))
        except Exception:
            pass
        summary[key] = sorted(seen)
    first = _first_module_tensor_device(module)
    summary["first_tensor_device"] = str(first) if first is not None else None
    return summary


def _emit_metric_on_gpu_debug(kind: str, module: Any, requested: torch.device) -> None:
    """若请求在 CUDA 上且模块张量已在 GPU，则打印一行 debug。"""
    if requested.type != "cuda":
        return
    got = _first_module_tensor_device(module)
    if got is not None and got.type == "cuda":
        print(
            f"[render_metrics][debug] {kind}: gpu_ok requested={requested!s} first_tensor={got!s}",
            flush=True,
        )
    else:
        print(
            f"[render_metrics][warn] {kind}: requested CUDA but module tensors are not on GPU "
            f"(requested={requested!s}, first_tensor={got!s})",
            flush=True,
        )


def _parse_int_list_value(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in {"none", "false", "off", "cpu"}:
            return []
        if s.lower() == "auto":
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(raw)]


def _render_worker_gpu_ids(render_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> List[int]:
    env = (os.environ.get("EVAL_RENDER_METRICS_RENDER_GPU_IDS") or "").strip()
    raw = env or render_cfg.get("render_gpu_ids", render_cfg.get("mesh_render_gpu_ids"))
    if raw is None:
        raw = (cfg.get("parallel") or {}).get("gpu_ids")
    ids = _parse_int_list_value(raw)
    if not ids or not torch.cuda.is_available():
        return []
    ndev = int(torch.cuda.device_count())
    return [i for i in ids if 0 <= int(i) < ndev]


def _render_worker_count(render_cfg: Dict[str, Any], gpu_ids: Sequence[int]) -> int:
    raw = os.environ.get("EVAL_RENDER_METRICS_RENDER_WORKERS")
    if raw is None:
        raw = render_cfg.get("render_num_workers", render_cfg.get("mesh_render_num_workers"))
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0
    return len(gpu_ids) if len(gpu_ids) > 1 else 0


def _render_metric_sample_worker(args: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        sample_index,
        record,
        out_dir,
        nviews,
        resolution,
        gpu_id,
        mesh_backend,
    ) = args
    if mesh_backend:
        os.environ["EVAL_MESH_RENDER_BACKEND"] = str(mesh_backend)
    if gpu_id is not None:
        os.environ["EVAL_MESH_RENDER_DEVICE"] = f"cuda:{int(gpu_id)}"
        os.environ.setdefault("EVAL_MESH_RENDER_BACKEND", "cuda")
        try:
            torch.cuda.set_device(int(gpu_id))
        except Exception:
            pass

    from eval.utils.mesh_multiview_render import (
        load_colored_trimesh_any,
        render_colored_trimesh_multiview,
    )

    ref_path = str(record.get("reference_mesh_path"))
    gen_path = os.path.join(out_dir, str(record.get("glb_rel_path")))
    row: Dict[str, Any] = {
        "sample_id": record.get("sample_id"),
        "sample_index": int(sample_index),
        "render_worker_gpu_id": int(gpu_id) if gpu_id is not None else None,
    }
    t0 = time.perf_counter()
    try:
        t_ml0 = time.perf_counter()
        gt = load_colored_trimesh_any(ref_path)
        row["mesh_load_ref_sec"] = time.perf_counter() - t_ml0
        t_ml1 = time.perf_counter()
        pr = load_colored_trimesh_any(gen_path)
        row["mesh_load_gen_sec"] = time.perf_counter() - t_ml1

        t_r0 = time.perf_counter()
        ref_imgs = render_colored_trimesh_multiview(gt, nviews=nviews, resolution=resolution)
        row["render_ref_sec"] = time.perf_counter() - t_r0
        t_r1 = time.perf_counter()
        gen_imgs = render_colored_trimesh_multiview(pr, nviews=nviews, resolution=resolution)
        row["render_gen_sec"] = time.perf_counter() - t_r1
        row["render_worker_wall_sec"] = time.perf_counter() - t0
        return {
            "ok": True,
            "sample_index": int(sample_index),
            "row": row,
            "ref_imgs": ref_imgs,
            "gen_imgs": gen_imgs,
        }
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True))
        row["render_worker_wall_sec"] = time.perf_counter() - t0
        return {
            "ok": False,
            "sample_index": int(sample_index),
            "row": row,
            "error": repr(exc),
            "traceback": tb,
        }


def generation_render_metric_names(metric_names: Sequence[str]) -> List[str]:
    return [m for m in metric_names if m in RENDER_METRIC_NAMES]


def _ref_gen_np_to_tensors(
    ref_arr: np.ndarray, gen_arr: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HWC uint8 RGB → (C,H,W) float32 in [0,1]; order matches :meth:`RenderMetrics.compute`."""
    ref_arr = np.ascontiguousarray(np.asarray(ref_arr, dtype=np.uint8))
    gen_arr = np.ascontiguousarray(np.asarray(gen_arr, dtype=np.uint8))
    ref_t = torch.from_numpy(ref_arr).float().permute(2, 0, 1) / 255.0
    gen_t = torch.from_numpy(gen_arr).float().permute(2, 0, 1) / 255.0
    return ref_t, gen_t


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """(C,H,W) in [0,1]."""
    try:
        from trellis.utils.loss_utils import psnr

        return float(psnr(img1, img2).item())
    except Exception:
        mse = torch.nn.functional.mse_loss(img1, img2)
        return float((20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12))).item())


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    try:
        from trellis.utils.loss_utils import ssim

        return float(ssim(img1.unsqueeze(0), img2.unsqueeze(0)).item())
    except Exception:
        warnings.warn("trellis SSIM unavailable, returning -1")
        return -1.0


def compute_lpips(img1: torch.Tensor, img2: torch.Tensor) -> float:
    try:
        from trellis.utils.loss_utils import lpips

        return float(lpips(img1.unsqueeze(0), img2.unsqueeze(0)).item())
    except Exception:
        warnings.warn("trellis LPIPS unavailable, returning -1")
        return -1.0


def compute_fid(
    real_images: List[np.ndarray],
    generated_images: List[np.ndarray],
    *,
    device: Optional[torch.device] = None,
    inception_batch_size: int = 128,
) -> float:
    """Legacy FID on uint8 (H,W,3); Inception runs on ``device`` when CUDA is available."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
        _batched_metric_updates(
            fid, real_images, real=True, device=dev, batch_size=inception_batch_size, as_float01=True
        )
        _batched_metric_updates(
            fid,
            generated_images,
            real=False,
            device=dev,
            batch_size=inception_batch_size,
            as_float01=True,
        )
        return float(fid.compute().item())
    except Exception as exc:
        warnings.warn(f"FID compute failed: {exc!r}")
        return -1.0


def _hwc_rgb_uint8_contiguous(im: np.ndarray) -> np.ndarray:
    """PyVista/Matplotlib often返回带负 stride 的视图；torch.from_numpy 不接受。"""
    arr = np.asarray(im, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB uint8, got shape={tuple(arr.shape)}")
    return np.ascontiguousarray(arr)


def _images_to_chw_float01(images: List[np.ndarray]) -> torch.Tensor:
    """List of (H,W,3) uint8 -> (N,3,H,W) float32 in [0,1]."""
    tensors = []
    for im in images:
        arr = _hwc_rgb_uint8_contiguous(im)
        t = torch.from_numpy(arr).float() / 255.0
        tensors.append(t.permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def _images_to_chw_uint8(images: List[np.ndarray]) -> torch.Tensor:
    """List of (H,W,3) uint8 -> (N,3,H,W) uint8."""
    tensors = []
    for im in images:
        arr = _hwc_rgb_uint8_contiguous(im)
        t = torch.from_numpy(arr)
        tensors.append(t.permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def _images_to_uint8_batch(images: List[np.ndarray]) -> torch.Tensor:
    """List of (H,W,3) uint8 -> (N,3,H,W) float [0,1] (FID / CLIP-style)."""
    return _images_to_chw_float01(images)


def _optional_positive_float(val: Any) -> Optional[float]:
    """Parse optional Gaussian-RBF ``gamma``; ``None`` / empty means use ``1/feature_dim``."""
    if val is None or val == "":
        return None
    try:
        x = float(val)
        return x if x > 0.0 else None
    except (TypeError, ValueError):
        return None


def _resolve_render_metrics_device(render_cfg: Dict[str, Any]) -> torch.device:
    """
    Device for Inception FD/KD and CLIP during aggregate render metrics.

    Priority: ``EVAL_RENDER_METRICS_DEVICE`` env > ``metrics_config.render.device``.
    Values: ``auto`` (default) | ``cuda`` | ``cuda:N`` | ``cpu``.
    """
    env = (os.environ.get("EVAL_RENDER_METRICS_DEVICE") or "").strip()
    raw = env if env else str(render_cfg.get("device", "auto") or "auto").strip()
    key = raw.lower()
    if key in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return torch.device(raw)


def _pairwise_squared_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise squared Euclidean distances between ``a`` (na, d) and ``b`` (nb, d) -> (na, nb)."""
    a = a.to(dtype=torch.float64)
    b = b.to(dtype=torch.float64)
    sa = (a * a).sum(dim=1, keepdim=True)
    sb = (b * b).sum(dim=1, keepdim=True).T
    return torch.clamp(sa + sb - 2.0 * a.matmul(b.T), min=0.0)


def _batched_metric_updates(
    metric: Any,
    images: List[np.ndarray],
    *,
    real: bool,
    device: torch.device,
    batch_size: int,
    as_float01: bool,
) -> None:
    """Repeated ``metric.update`` on GPU-sized chunks to limit VRAM."""
    n = len(images)
    if n == 0:
        return
    bs = max(1, int(batch_size))
    for start in range(0, n, bs):
        end = min(n, start + bs)
        sl = images[start:end]
        batch = _images_to_chw_float01(sl) if as_float01 else _images_to_chw_uint8(sl)
        batch = batch.to(device, non_blocking=(device.type == "cuda"))
        metric.update(batch, real=real)


def _as_feature_matrix(features: Any) -> torch.Tensor:
    """Normalize Inception feature extractor outputs to ``(N, D)``."""
    if isinstance(features, (tuple, list)):
        if not features:
            raise ValueError("Empty Inception feature output")
        features = features[0]
    if isinstance(features, dict):
        if "2048" in features:
            features = features["2048"]
        else:
            features = next(iter(features.values()))
    if not torch.is_tensor(features):
        raise TypeError(f"Unexpected Inception feature output type: {type(features)!r}")
    if features.dim() > 2:
        features = torch.flatten(features, start_dim=1)
    if features.dim() != 2:
        raise ValueError(f"Expected Inception features shaped (N, D), got {tuple(features.shape)}")
    return features


def _inception_feature_extractor(device: torch.device) -> Any:
    """Build torchmetrics' Inception network but keep stats/math under our control."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=True)
    model = metric.inception.to(device)
    model.eval()
    return model


def _extract_inception_features_gpu(
    images: List[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    profile: Optional[Dict[str, float]] = None,
    profile_prefix: str = "inception",
    feature_extractor: Optional[Any] = None,
) -> torch.Tensor:
    """Extract Inception features and keep features on ``device``.

    torchmetrics' wrapped torch-fidelity Inception expects uint8 CHW images even when the
    owning FID metric was constructed with ``normalize=True``.
    """
    if not images:
        return torch.empty((0, 2048), device=device, dtype=torch.float64)
    t0 = time.perf_counter()
    model = feature_extractor if feature_extractor is not None else _inception_feature_extractor(device)
    if profile is not None and feature_extractor is None:
        profile[f"{profile_prefix}_feature_model_construct"] = time.perf_counter() - t0
    feats: List[torch.Tensor] = []
    bs = max(1, int(batch_size))
    t_extract0 = time.perf_counter()
    print(
        f"[render_metrics][debug] {profile_prefix}: feature_extract_start "
        f"device={device!s} images={len(images)} batch_size={bs} input_dtype=uint8",
        flush=True,
    )
    with torch.inference_mode():
        for start in range(0, len(images), bs):
            batch = _images_to_chw_uint8(images[start : start + bs])
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            try:
                out = model(batch)
            except Exception as exc:
                print(
                    f"[render_metrics][error] {profile_prefix}: feature_batch_failed "
                    f"range={start}:{min(len(images), start + bs)} "
                    f"shape={tuple(batch.shape)} dtype={batch.dtype} device={batch.device} "
                    f"error={exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            feats.append(_as_feature_matrix(out).to(device=device, dtype=torch.float64))
    _cuda_synchronize_if_needed(device)
    out_features = torch.cat(feats, dim=0)
    dt_extract = time.perf_counter() - t_extract0
    if profile is not None:
        profile[f"{profile_prefix}_feature_extract"] = dt_extract
    print(
        f"[render_metrics][debug] {profile_prefix}: feature_extract_done "
        f"sec={dt_extract:.3f} features_shape={tuple(out_features.shape)} "
        f"features_dtype={out_features.dtype} features_device={out_features.device}",
        flush=True,
    )
    return out_features


def _feature_mean_cov(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    features = features.to(dtype=torch.float64)
    n = int(features.shape[0])
    if n < 2:
        raise ValueError(f"Need at least 2 feature vectors, got {n}")
    mean = features.mean(dim=0)
    centered = features - mean
    cov = centered.T.matmul(centered) / float(n - 1)
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    mat = 0.5 * (mat + mat.T)
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=0.0)
    return (evecs * torch.sqrt(evals).unsqueeze(0)).matmul(evecs.T)


def _frechet_distance_from_features_gpu(real_feat: torch.Tensor, fake_feat: torch.Tensor) -> torch.Tensor:
    mu_r, sigma_r = _feature_mean_cov(real_feat)
    mu_f, sigma_f = _feature_mean_cov(fake_feat)
    diff = mu_r - mu_f
    sqrt_sigma_r = _sqrtm_psd(sigma_r)
    covmean = _sqrtm_psd(sqrt_sigma_r.matmul(sigma_f).matmul(sqrt_sigma_r))
    val = diff.dot(diff) + torch.trace(sigma_r) + torch.trace(sigma_f) - 2.0 * torch.trace(covmean)
    return torch.clamp(val.real, min=0.0)


def _mmd2_gaussian_rbf_from_features_gpu(
    real_feat: torch.Tensor,
    fake_feat: torch.Tensor,
    subset_size: int,
    *,
    gamma: float,
    num_subsets: int = 100,
) -> torch.Tensor:
    """Unbiased MMD² estimate with Gaussian RBF kernel ``k(x,y)=exp(-gamma||x-y||²)``."""
    n = min(int(real_feat.shape[0]), int(fake_feat.shape[0]))
    ss = max(2, min(int(subset_size), n))
    if n < 2:
        raise ValueError(f"Need at least 2 feature vectors for MMD², got {n}")
    real_feat = real_feat[:n].to(dtype=torch.float64)
    fake_feat = fake_feat[:n].to(dtype=torch.float64)
    g = float(gamma)
    rounds = 1 if ss >= n else max(1, int(num_subsets))
    vals: List[torch.Tensor] = []
    for _ in range(rounds):
        if ss < n:
            ix = torch.randperm(n, device=real_feat.device)[:ss]
            iy = torch.randperm(n, device=fake_feat.device)[:ss]
            x = real_feat.index_select(0, ix)
            y = fake_feat.index_select(0, iy)
        else:
            x = real_feat
            y = fake_feat
        dist_xx = _pairwise_squared_l2(x, x)
        dist_yy = _pairwise_squared_l2(y, y)
        dist_xy = _pairwise_squared_l2(x, y)
        k_xx = torch.exp(-g * dist_xx)
        k_yy = torch.exp(-g * dist_yy)
        k_xy = torch.exp(-g * dist_xy)
        denom = float(ss * (ss - 1))
        vals.append(
            (k_xx.sum() - torch.diagonal(k_xx).sum()) / denom
            + (k_yy.sum() - torch.diagonal(k_yy).sum()) / denom
            - 2.0 * k_xy.mean()
        )
    return torch.stack(vals).mean()


def _effective_kid_subset_size(subset_size: int, n: int) -> int:
    """Keep the previous torchmetrics subset heuristic while avoiding invalid tiny-N values."""
    if n <= 0:
        return 0
    return min(int(n), max(2, max(3, min(int(subset_size), int(n) // 2))))


def _compute_fd_inception(
    real_u8: List[np.ndarray],
    fake_u8: List[np.ndarray],
    *,
    device: torch.device,
    inception_batch_size: int = 128,
    profile: Optional[Dict[str, float]] = None,
    feature_extractor: Optional[Any] = None,
) -> float:
    if not real_u8 or not fake_u8:
        return -1.0
    try:
        t0 = time.perf_counter()
        extractor = (
            feature_extractor if feature_extractor is not None else _inception_feature_extractor(device)
        )
        _emit_metric_on_gpu_debug("Inception feature extractor (fd_inception)", extractor, device)
        if profile is not None and feature_extractor is None:
            profile["fd_feature_model_construct"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        real_feat = _extract_inception_features_gpu(
            real_u8,
            device=device,
            batch_size=inception_batch_size,
            profile=profile,
            profile_prefix="fd_real",
            feature_extractor=extractor,
        )
        if profile is not None:
            profile["fd_extract_real_total"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fake_feat = _extract_inception_features_gpu(
            fake_u8,
            device=device,
            batch_size=inception_batch_size,
            profile=profile,
            profile_prefix="fd_fake",
            feature_extractor=extractor,
        )
        if profile is not None:
            profile["fd_extract_fake_total"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _cuda_synchronize_if_needed(device)
        print(
            f"[render_metrics][debug] fd_inception: computing Frechet stats on {device!s} "
            f"(real_features={tuple(real_feat.shape)} fake_features={tuple(fake_feat.shape)})",
            flush=True,
        )
        val_t = _frechet_distance_from_features_gpu(real_feat, fake_feat)
        _cuda_synchronize_if_needed(device)
        dt_compute = time.perf_counter() - t0
        if profile is not None:
            profile["fd_compute"] = dt_compute
        print(
            f"[render_metrics][debug] fd_inception: GPU stats compute finished in {dt_compute:.3f}s",
            flush=True,
        )
        return float(val_t.item())
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True))
        print(f"[render_metrics][error] fd_inception failed: {exc!r}\n{tb}", file=sys.stderr, flush=True)
        warnings.warn(f"fd_inception failed: {exc!r}")
        return -1.0


def _compute_kid_inception(
    real_u8: List[np.ndarray],
    fake_u8: List[np.ndarray],
    subset_size: int,
    *,
    device: torch.device,
    inception_batch_size: int = 128,
    rbf_gamma: Optional[float] = None,
    profile: Optional[Dict[str, float]] = None,
    feature_extractor: Optional[Any] = None,
) -> float:
    if not real_u8 or not fake_u8:
        return -1.0
    try:
        n = min(len(real_u8), len(fake_u8))
        ss = _effective_kid_subset_size(subset_size, n)
        t0 = time.perf_counter()
        extractor = (
            feature_extractor if feature_extractor is not None else _inception_feature_extractor(device)
        )
        _emit_metric_on_gpu_debug("Inception feature extractor (kd_inception)", extractor, device)
        if profile is not None and feature_extractor is None:
            profile["kid_feature_model_construct"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        real_feat = _extract_inception_features_gpu(
            real_u8,
            device=device,
            batch_size=inception_batch_size,
            profile=profile,
            profile_prefix="kid_real",
            feature_extractor=extractor,
        )
        if profile is not None:
            profile["kid_extract_real_total"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fake_feat = _extract_inception_features_gpu(
            fake_u8,
            device=device,
            batch_size=inception_batch_size,
            profile=profile,
            profile_prefix="kid_fake",
            feature_extractor=extractor,
        )
        if profile is not None:
            profile["kid_extract_fake_total"] = time.perf_counter() - t0
        dim = max(1, int(real_feat.shape[1]))
        gamma_eff = float(rbf_gamma) if rbf_gamma is not None else (1.0 / float(dim))
        if profile is not None:
            profile["kid_rbf_gamma"] = gamma_eff
        t0 = time.perf_counter()
        _cuda_synchronize_if_needed(device)
        print(
            f"[render_metrics][debug] kd_inception: Gaussian RBF MMD² on {device!s} "
            f"(real_features={tuple(real_feat.shape)} fake_features={tuple(fake_feat.shape)} "
            f"subset_size={ss} gamma={gamma_eff})",
            flush=True,
        )
        out = _mmd2_gaussian_rbf_from_features_gpu(
            real_feat, fake_feat, ss, gamma=gamma_eff
        )
        _cuda_synchronize_if_needed(device)
        dt_compute = time.perf_counter() - t0
        mmd2_raw = float(out.item())
        if profile is not None:
            profile["kid_compute"] = dt_compute
            profile["kid_mmd2_unscaled"] = mmd2_raw
            profile["kid_output_scale"] = 100.0
        print(
            f"[render_metrics][debug] kd_inception: GPU MMD² compute finished in {dt_compute:.3f}s "
            f"(mmd2_raw={mmd2_raw:.6g} reported=100*mmd2_raw)",
            flush=True,
        )
        return 100.0 * mmd2_raw
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True))
        print(f"[render_metrics][error] kd_inception (Gaussian MMD²) failed: {exc!r}\n{tb}", file=sys.stderr, flush=True)
        warnings.warn(f"kd_inception (Gaussian MMD²) failed: {exc!r}")
        return -1.0


def _load_clip_model_processor(model_name: str, device: torch.device) -> Tuple[Any, Any]:
    """Load CLIP once for many samples (avoid repeated HF I/O during eval)."""
    from huggingface_hub import hf_hub_download
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor

    def _load_clip_weights_safetensors_only(repo_id: str, load_device: str) -> CLIPModel:
        """Load CLIP weights without touching ``torch.load`` on ``pytorch_model.bin``."""
        try:
            from safetensors.torch import load_file as safetensors_load_file
        except Exception as exc:
            raise RuntimeError(
                "CLIP fallback needs the `safetensors` package "
                "(``pip install safetensors``) to avoid torch.load on legacy .bin weights."
            ) from exc
        try:
            weight_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                repo_type="model",
            )
        except Exception as exc:
            raise RuntimeError(
                f"CLIP safetensors weight download failed for {repo_id!r}: {exc!r}. "
                "Install `safetensors` and ensure the repo provides `model.safetensors`, "
                "or set metrics_config.render.clip_model to a repo that does."
            ) from exc

        cfg = CLIPConfig.from_pretrained(repo_id)
        model = CLIPModel(cfg)
        state = safetensors_load_file(weight_path, device=load_device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            warnings.warn(f"CLIP load_state_dict missing keys (non-fatal): {missing[:8]!r} ...")
        if unexpected:
            warnings.warn(f"CLIP load_state_dict unexpected keys (non-fatal): {unexpected[:8]!r} ...")
        return model

    load_dev = str(device) if device.type == "cuda" else "cpu"
    try:
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    except Exception as exc:
        msg = str(exc)
        if (
            "CVE-2025-32434" in msg
            or "upgrade torch to at least v2.6" in msg
            or "safetensors" in msg.lower()
        ):
            model = _load_clip_weights_safetensors_only(model_name, load_dev)
            model = model.to(device)
        else:
            raise
    proc = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, proc


def _as_clip_feature_matrix(out: Any) -> torch.Tensor:
    """Normalize CLIP ``get_*_features`` return type across transformers versions."""
    if torch.is_tensor(out):
        t = out
    else:
        po = getattr(out, "pooler_output", None)
        if po is not None and torch.is_tensor(po):
            t = po
        else:
            lhs = getattr(out, "last_hidden_state", None)
            if lhs is None or not torch.is_tensor(lhs):
                raise TypeError(f"Unexpected CLIP features type: {type(out)!r}")
            t = lhs[:, 0, :]
    if t.dim() != 2:
        raise ValueError(f"Expected (N, D) CLIP features, got shape={tuple(t.shape)}")
    return F.normalize(t, p=2, dim=-1, eps=1e-8)


def _clip_mean_score_preloaded(
    model: Any,
    proc: Any,
    images_u8: List[np.ndarray],
    text: str,
    *,
    device: torch.device,
    clip_batch_size: int = 32,
) -> float:
    """Mean cosine similarity in ``[-1, 1]`` (aggregate ``clip_score`` multiplies by 100)."""
    if not images_u8 or not str(text).strip():
        return -1.0
    scores: List[float] = []
    txt = proc(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    txt = {k: v.to(device) for k, v in txt.items()}
    with torch.no_grad():
        te_raw = model.get_text_features(
            input_ids=txt["input_ids"],
            attention_mask=txt.get("attention_mask"),
        )
        te = _as_clip_feature_matrix(te_raw)
    bs = max(1, int(clip_batch_size))
    for i in range(0, len(images_u8), bs):
        chunk = [_hwc_rgb_uint8_contiguous(x) for x in images_u8[i : i + bs]]
        pix = proc(images=chunk, return_tensors="pt", padding=True)
        pix = {k: v.to(device) for k, v in pix.items()}
        with torch.no_grad():
            ie_raw = model.get_image_features(pixel_values=pix["pixel_values"])
        ie = _as_clip_feature_matrix(ie_raw)
        sim = (ie * te.expand_as(ie)).sum(dim=-1)
        scores.extend([float(s.item()) for s in sim])
    return float(np.mean(scores)) if scores else -1.0


def _clip_mean_score(
    images_u8: List[np.ndarray],
    text: str,
    *,
    device: torch.device,
    model_name: str = "openai/clip-vit-base-patch32",
    clip_batch_size: int = 32,
) -> float:
    """Load CLIP then score (legacy path; prefer preloaded model in aggregate loop)."""
    if not images_u8 or not str(text).strip():
        return -1.0
    try:
        model, proc = _load_clip_model_processor(model_name, device)
        return _clip_mean_score_preloaded(
            model, proc, images_u8, text, device=device, clip_batch_size=clip_batch_size
        )
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True))
        print(f"[render_metrics][error] clip_score failed: {exc!r}\n{tb}", file=sys.stderr, flush=True)
        warnings.warn(f"clip_score failed: {exc!r}")
        return -1.0


def render_multiview_for_comparison(
    sample: Any, resolution: int = 512, nviews: int = 30
) -> List[np.ndarray]:
    """Reuses trellis.utils.render_utils.render_multiview when Trellis is available."""
    from trellis.utils.render_utils import render_multiview

    colors, _, _ = render_multiview(sample, resolution=resolution, nviews=nviews)
    return colors


def _caption_for_clip_from_record(record: Dict[str, Any]) -> str:
    """CLIP text must be the dataset caption only, never the generation prompt wrapper."""
    cap = record.get("caption")
    if isinstance(cap, str) and cap.strip():
        return cap.strip()
    extra = record.get("extra")
    if isinstance(extra, dict):
        cap = extra.get("caption")
        if isinstance(cap, str) and cap.strip():
            return cap.strip()
    return ""


def _truthy_config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_path_component(value: Any, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or fallback


def _save_metric_render_images(
    *,
    out_dir: str,
    record: Dict[str, Any],
    sample_index: int,
    ref_imgs: List[np.ndarray],
    gen_imgs: List[np.ndarray],
    metric_names: Sequence[str],
    resolution: int,
) -> None:
    from PIL import Image
    from eval.utils.mesh_multiview_render import fixed_metric_view_names

    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    sid = record.get("sample_id") or extra.get("sample_id") or f"sample_{sample_index:06d}"
    safe_sid = _safe_path_component(sid, f"sample_{sample_index:06d}")
    sample_dir = os.path.join(out_dir, "metric_renders", safe_sid)
    shutil.rmtree(sample_dir, ignore_errors=True)

    view_names = fixed_metric_view_names(min(len(ref_imgs), len(gen_imgs)))
    saved: Dict[str, List[str]] = {"reference": [], "generated": []}
    for kind, images in (("reference", ref_imgs), ("generated", gen_imgs)):
        kind_dir = os.path.join(sample_dir, kind)
        os.makedirs(kind_dir, exist_ok=True)
        for i, img in enumerate(images):
            view_name = view_names[i] if i < len(view_names) else f"view_{i:03d}"
            path = os.path.join(kind_dir, f"{view_name}.png")
            Image.fromarray(_hwc_rgb_uint8_contiguous(img)).save(path)
            saved[kind].append(os.path.relpath(path, out_dir).replace(os.sep, "/"))

    record["metric_render_images"] = {
        "metrics": [m for m in metric_names if m in RENDER_METRIC_NAMES],
        "resolution": int(resolution),
        "views": view_names,
        "reference": saved["reference"],
        "generated": saved["generated"],
    }


def compute_aggregate_generation_render_metrics(
    records: List[Dict[str, Any]],
    out_dir: str,
    cfg: Dict[str, Any],
    metric_names: Sequence[str],
) -> Dict[str, float]:
    """
    FD/KD on Inception features over multi-view renders; CLIP as mean(view, text);
    PSNR/SSIM as mean over all paired reference/generated views.

    Inception FD/KD aggregation is controlled by ``metrics_config.render.inception_aggregate``:
    ``pooled`` (default) concatenates all samples' views; ``per_sample_mean`` computes FD/KD
    per 3D sample then averages (paper-style table reporting).

    Each ``record`` should have ``reference_mesh_path`` and generated ``glb_rel_path``.
    Image metrics intentionally require the Trellis-textured GLB; the geometry-only OBJ
    fallback is not used because it would feed non-colored renders into CLIP/FID/KD.
    """
    from eval.utils.mesh_multiview_render import (
        load_colored_trimesh_any,
        render_colored_trimesh_multiview,
    )

    names = [m for m in metric_names if m in RENDER_METRIC_NAMES]
    if not names:
        return {}

    t_total0 = time.perf_counter()
    mc = cfg.get("metrics_config") or {}
    base_render = dict(mc.get("render") or {})
    render_cfg = dict(base_render)
    # 常见误配：把 render 指标调试键写在 model: 下；此处兼容合并并提示迁到 metrics_config.render
    _RENDER_METRIC_DEBUG_KEYS = (
        "verbose_metrics_timeline",
        "debug_gpu_poll",
        "debug_gpu_poll_interval_sec",
        "debug_gpu_phase_snapshots",
    )
    model_cfg = cfg.get("model") or {}
    _migrated_debug_keys: List[str] = []
    for _k in _RENDER_METRIC_DEBUG_KEYS:
        if _k not in base_render and _k in model_cfg:
            render_cfg[_k] = model_cfg[_k]
            _migrated_debug_keys.append(_k)
    if _migrated_debug_keys:
        print(
            f"[render_metrics][warn] 以下键在 cfg.model 中定义，已临时并入 metrics 逻辑（请改到 "
            f"metrics_config.render）: {_migrated_debug_keys!r}",
            flush=True,
        )
    nviews = int(render_cfg.get("nviews", 4))
    resolution = int(render_cfg.get("resolution", 512))
    kid_subset = int(render_cfg.get("kid_subset_size", 50))
    kid_rbf_gamma = _optional_positive_float(render_cfg.get("kid_rbf_gamma"))
    clip_model = str(render_cfg.get("clip_model", "openai/clip-vit-base-patch32"))
    save_images = _truthy_config_bool(render_cfg.get("save_images"), False)
    device = _resolve_render_metrics_device(render_cfg)
    inception_batch_size = int(render_cfg.get("inception_batch_size", 128))
    clip_batch_size = int(render_cfg.get("clip_batch_size", 32))
    verbose_timeline = _truthy_config_bool(render_cfg.get("verbose_metrics_timeline"), False) or _truthy_env(
        "EVAL_RENDER_METRICS_TIMELINE"
    )
    poll_gpu = _truthy_config_bool(render_cfg.get("debug_gpu_poll"), False) or _truthy_env(
        "EVAL_RENDER_METRICS_GPU_POLL"
    )
    poll_interval = float(render_cfg.get("debug_gpu_poll_interval_sec", 0.25) or 0.25)
    phase_snapshots = _truthy_config_bool(
        render_cfg.get("debug_gpu_phase_snapshots"), True
    )
    inception_agg = str(render_cfg.get("inception_aggregate", "pooled")).lower().strip()
    per_sample_mean_inception = inception_agg in ("per_sample_mean", "per_sample", "paper")

    gpu_idx = _cuda_device_index(device)
    gpu_poller: Optional[_GpuUtilPollThread] = None
    if poll_gpu and gpu_idx is not None:
        gpu_poller = _GpuUtilPollThread(gpu_idx, poll_interval)
        gpu_poller.start()
        print(
            f"[render_metrics][debug] GPU poll thread started (gpu={gpu_idx}, interval={poll_interval}s). "
            f"Disable with metrics_config.render.debug_gpu_poll=false or unset EVAL_RENDER_METRICS_GPU_POLL.",
            flush=True,
        )

    gpu_snapshots: Dict[str, Any] = {}
    if device.type == "cuda" and torch.cuda.is_available() and gpu_idx is not None and phase_snapshots:
        snap = _nvidia_smi_query_gpu(gpu_idx)
        if snap is not None:
            gpu_snapshots["aggregate_start"] = snap

    if device.type == "cuda" and torch.cuda.is_available() and gpu_idx is not None:
        try:
            torch.cuda.reset_peak_memory_stats(gpu_idx)
        except Exception:
            pass

    print(
        f"[render_metrics][debug] aggregate render metrics start: "
        f"device={device!s} cuda_available={torch.cuda.is_available()} "
        f"metrics={names} nviews={nviews} resolution={resolution} "
        f"records={len(records)} inception_aggregate={inception_agg!r} "
        f"inception_batch_size={inception_batch_size} "
        f"clip_batch_size={clip_batch_size} kid_subset_size={kid_subset} "
        f"kid_rbf_gamma={'auto(1/dim)' if kid_rbf_gamma is None else kid_rbf_gamma} "
        f"clip_model={clip_model!r} "
        f"verbose_timeline={verbose_timeline} debug_gpu_poll={poll_gpu} gpu_index={gpu_idx}",
        flush=True,
    )
    print(
        f"[render_metrics][debug] cuda_mem(aggregate_start)={_torch_cuda_memory_stats(device)}",
        flush=True,
    )
    if device.type == "cuda" and gpu_idx is not None and not poll_gpu:
        print(
            "[render_metrics][debug] nvidia-smi GPU utilization polling is OFF. "
            "Enable: metrics_config.render.debug_gpu_poll=true or env EVAL_RENDER_METRICS_GPU_POLL=1 "
            "(adds subprocess overhead).",
            flush=True,
        )

    clip_model_obj: Any = None
    clip_proc_obj: Any = None
    clip_preload_sec = 0.0
    if "clip_score" in names:
        try:
            t_clip_ld0 = time.perf_counter()
            clip_model_obj, clip_proc_obj = _load_clip_model_processor(clip_model, device)
            clip_preload_sec = time.perf_counter() - t_clip_ld0
            _emit_metric_on_gpu_debug(f"CLIP ({clip_model})", clip_model_obj, device)
        except Exception as exc:
            warnings.warn(f"CLIP 预加载失败，将跳过 clip_score: {exc!r}")

    shared_inception_fe: Optional[Any] = None
    if per_sample_mean_inception and ("fd_inception" in names or "kd_inception" in names):
        t_fe0 = time.perf_counter()
        shared_inception_fe = _inception_feature_extractor(device)
        print(
            f"[render_metrics][debug] inception_aggregate={inception_agg!r}: "
            f"shared Inception-V3 extractor ready in {time.perf_counter() - t_fe0:.3f}s",
            flush=True,
        )

    if phase_snapshots and gpu_idx is not None:
        snap = _nvidia_smi_query_gpu(gpu_idx)
        if snap is not None:
            gpu_snapshots["after_clip_preload"] = snap
    print(
        f"[render_metrics][debug] cuda_mem(after_clip_preload)={_torch_cuda_memory_stats(device)}",
        flush=True,
    )

    all_real: List[np.ndarray] = []
    all_fake: List[np.ndarray] = []
    per_sample_fd: List[float] = []
    per_sample_kd: List[float] = []
    clip_vals: List[float] = []
    per_pair_psnr: List[float] = []
    per_pair_ssim: List[float] = []
    t_mesh_load_ref = 0.0
    t_mesh_load_gen = 0.0
    t_render_ref_total = 0.0
    t_render_gen_total = 0.0
    t_save_images_total = 0.0
    t_clip_score_total = 0.0
    t_fd_sec = 0.0
    t_kd_sec = 0.0
    n_samples_rendered = 0
    per_sample_rows: List[Dict[str, Any]] = []
    parallel_render_wall_sec = 0.0
    render_worker_failures = 0
    rendered_by_index: Dict[int, Dict[str, Any]] = {}
    render_gpu_ids = _render_worker_gpu_ids(render_cfg, cfg)
    render_worker_count = _render_worker_count(render_cfg, render_gpu_ids)
    mesh_backend = str(render_cfg.get("mesh_render_backend") or "").strip() or None

    if render_worker_count > 1:
        render_tasks: List[Tuple[Any, ...]] = []
        for sample_index, r in enumerate(records):
            refp = r.get("reference_mesh_path")
            rel_g = r.get("glb_rel_path")
            if not refp or not rel_g:
                continue
            ref_path = str(refp)
            gen_path = os.path.join(out_dir, str(rel_g))
            if not os.path.isfile(ref_path) or not os.path.isfile(gen_path):
                continue
            gpu_id = render_gpu_ids[len(render_tasks) % len(render_gpu_ids)] if render_gpu_ids else None
            render_tasks.append((sample_index, r, out_dir, nviews, resolution, gpu_id, mesh_backend))

        if render_tasks:
            print(
                f"[render_metrics][debug] parallel RGB mesh render start: "
                f"tasks={len(render_tasks)} workers={render_worker_count} gpu_ids={render_gpu_ids} "
                f"mesh_backend={mesh_backend or os.environ.get('EVAL_MESH_RENDER_BACKEND', 'auto')!r}",
                flush=True,
            )
            t_pr0 = time.perf_counter()
            try:
                ctx = mp.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=render_worker_count,
                    mp_context=ctx,
                ) as ex:
                    futures = [ex.submit(_render_metric_sample_worker, task) for task in render_tasks]
                    for fut in concurrent.futures.as_completed(futures):
                        res = fut.result()
                        idx = int(res.get("sample_index", -1))
                        if res.get("ok"):
                            rendered_by_index[idx] = res
                        else:
                            render_worker_failures += 1
                            row = res.get("row") if isinstance(res.get("row"), dict) else {}
                            warnings.warn(
                                f"parallel render failed sample_id={row.get('sample_id')!r} "
                                f"gpu={row.get('render_worker_gpu_id')!r}: {res.get('error')}"
                            )
            except Exception as exc:
                rendered_by_index.clear()
                render_worker_failures = 0
                warnings.warn(
                    f"parallel RGB mesh render failed ({exc!r}); falling back to serial render."
                )
            parallel_render_wall_sec = time.perf_counter() - t_pr0
            if rendered_by_index:
                print(
                    f"[render_metrics][debug] parallel RGB mesh render done: "
                    f"ok={len(rendered_by_index)}/{len(render_tasks)} failures={render_worker_failures} "
                    f"wall={parallel_render_wall_sec:.3f}s",
                    flush=True,
                )

    for sample_index, r in enumerate(records):
        refp = r.get("reference_mesh_path")
        rel_g = r.get("glb_rel_path")
        sid = r.get("sample_id")
        if not refp or not rel_g:
            if not refp and not rel_g:
                detail = "reference_mesh_path 与 glb_rel_path 均为空"
            elif not refp:
                detail = "缺少 reference_mesh_path（参考 glb 未写入记录，检查 metadata+glb_dir 加载）"
            else:
                detail = (
                    "缺少 glb_rel_path：图像类指标需要「推理后 Trellis 上色并保存」的生成 glb；"
                    "仅有磁盘上的参考 glb 不够。请确认 model.colorization.enabled、Trellis 管线已加载且上色成功"
                    "（见 per_sample.jsonl 中 colorize_error / 控制台 Trellis 报错）。"
                )
            warnings.warn(
                f"skip render metrics sample_id={sid!r}: {detail}。"
                "（若不需要 FD/KD/CLIP，可从 metrics 列表中移除这些项。）"
            )
            continue
        ref_path = str(refp)
        gen_path = os.path.join(out_dir, str(rel_g))
        if not os.path.isfile(ref_path) or not os.path.isfile(gen_path):
            warnings.warn(
                f"skip render metrics sample: missing mesh file ref={ref_path!r} gen={gen_path!r}"
            )
            continue
        pre_rendered = rendered_by_index.get(sample_index)
        if pre_rendered is not None:
            row = dict(pre_rendered.get("row") or {"sample_id": sid, "sample_index": sample_index})
            ref_imgs = list(pre_rendered.get("ref_imgs") or [])
            gen_imgs = list(pre_rendered.get("gen_imgs") or [])
            t_mesh_load_ref += float(row.get("mesh_load_ref_sec") or 0.0)
            t_mesh_load_gen += float(row.get("mesh_load_gen_sec") or 0.0)
            t_render_ref_total += float(row.get("render_ref_sec") or 0.0)
            t_render_gen_total += float(row.get("render_gen_sec") or 0.0)
        elif render_worker_count > 1 and rendered_by_index:
            warnings.warn(f"skip sample render after parallel worker failure: sample_id={sid!r}")
            continue
        else:
            row: Dict[str, Any] = {"sample_id": sid, "sample_index": sample_index}
            try:
                t_ml0 = time.perf_counter()
                gt = load_colored_trimesh_any(ref_path)
                row["mesh_load_ref_sec"] = time.perf_counter() - t_ml0
                t_ml1 = time.perf_counter()
                pr = load_colored_trimesh_any(gen_path)
                row["mesh_load_gen_sec"] = time.perf_counter() - t_ml1
                t_mesh_load_ref += float(row["mesh_load_ref_sec"])
                t_mesh_load_gen += float(row["mesh_load_gen_sec"])
            except Exception as exc:
                warnings.warn(f"skip sample mesh load: {exc!r}")
                continue

            try:
                t_r0 = time.perf_counter()
                ref_imgs = render_colored_trimesh_multiview(
                    gt, nviews=nviews, resolution=resolution
                )
                row["render_ref_sec"] = time.perf_counter() - t_r0
                t_r1 = time.perf_counter()
                gen_imgs = render_colored_trimesh_multiview(
                    pr, nviews=nviews, resolution=resolution
                )
                row["render_gen_sec"] = time.perf_counter() - t_r1
                t_render_ref_total += float(row["render_ref_sec"])
                t_render_gen_total += float(row["render_gen_sec"])
            except Exception as exc:
                warnings.warn(f"skip sample render: {exc!r}")
                continue

        n = min(len(ref_imgs), len(gen_imgs))
        if n == 0:
            continue
        ref_imgs = ref_imgs[:n]
        gen_imgs = gen_imgs[:n]
        n_samples_rendered += 1

        if "psnr" in names or "ssim" in names:
            for vi in range(n):
                ref_t, gen_t = _ref_gen_np_to_tensors(ref_imgs[vi], gen_imgs[vi])
                if "psnr" in names:
                    per_pair_psnr.append(compute_psnr(gen_t, ref_t))
                if "ssim" in names:
                    v_ssim = compute_ssim(gen_t, ref_t)
                    if v_ssim >= 0.0:
                        per_pair_ssim.append(v_ssim)

        if save_images:
            try:
                ts0 = time.perf_counter()
                _save_metric_render_images(
                    out_dir=out_dir,
                    record=r,
                    sample_index=sample_index,
                    ref_imgs=ref_imgs,
                    gen_imgs=gen_imgs,
                    metric_names=names,
                    resolution=resolution,
                )
                row["save_metric_images_sec"] = time.perf_counter() - ts0
                t_save_images_total += float(row["save_metric_images_sec"])
            except Exception as exc:
                row["save_metric_images_sec"] = None
                warnings.warn(
                    f"save render metric images failed for sample={r.get('sample_id')!r}: {exc!r}"
                )
        else:
            row["save_metric_images_sec"] = 0.0

        if "fd_inception" in names or "kd_inception" in names:
            if per_sample_mean_inception:
                if "fd_inception" in names:
                    _cuda_synchronize_if_needed(device)
                    t_fd0 = time.perf_counter()
                    v_fd = _compute_fd_inception(
                        ref_imgs,
                        gen_imgs,
                        device=device,
                        inception_batch_size=inception_batch_size,
                        profile=None,
                        feature_extractor=shared_inception_fe,
                    )
                    _cuda_synchronize_if_needed(device)
                    t_fd_sec += time.perf_counter() - t_fd0
                    per_sample_fd.append(v_fd)
                    row["fd_inception"] = v_fd
                if "kd_inception" in names:
                    _cuda_synchronize_if_needed(device)
                    t_kd0 = time.perf_counter()
                    v_kd = _compute_kid_inception(
                        ref_imgs,
                        gen_imgs,
                        kid_subset,
                        device=device,
                        inception_batch_size=inception_batch_size,
                        rbf_gamma=kid_rbf_gamma,
                        profile=None,
                        feature_extractor=shared_inception_fe,
                    )
                    _cuda_synchronize_if_needed(device)
                    t_kd_sec += time.perf_counter() - t_kd0
                    per_sample_kd.append(v_kd)
                    row["kd_inception"] = v_kd
            else:
                all_real.extend(ref_imgs)
                all_fake.extend(gen_imgs)

        if "clip_score" in names and clip_model_obj is not None and clip_proc_obj is not None:
            cap = _caption_for_clip_from_record(r)
            if cap:
                tc0 = time.perf_counter()
                s = _clip_mean_score_preloaded(
                    clip_model_obj,
                    clip_proc_obj,
                    gen_imgs,
                    cap,
                    device=device,
                    clip_batch_size=clip_batch_size,
                )
                row["clip_score_sec"] = time.perf_counter() - tc0
                t_clip_score_total += float(row["clip_score_sec"])
                clip_vals.append(s)
            else:
                row["clip_score_sec"] = None
                warnings.warn(
                    f"skip clip_score sample={r.get('sample_id')!r}: missing caption field"
                )
        else:
            row["clip_score_sec"] = None

        row["mesh_render_sum_sec"] = float(row.get("render_ref_sec", 0.0)) + float(
            row.get("render_gen_sec", 0.0)
        )
        per_sample_rows.append(row)
        if verbose_timeline:
            print(f"[render_metrics][debug] sample_timeline {row}", flush=True)

    if phase_snapshots and gpu_idx is not None:
        snap = _nvidia_smi_query_gpu(gpu_idx)
        if snap is not None:
            gpu_snapshots["after_sample_render_loop"] = snap
    print(
        f"[render_metrics][debug] cuda_mem(after_sample_render_loop)={_torch_cuda_memory_stats(device)}",
        flush=True,
    )

    out: Dict[str, float] = {}
    fd_profile: Dict[str, float] = {}
    kid_profile: Dict[str, float] = {}

    def _mean_valid_inception(vals: List[float]) -> float:
        ok = [x for x in vals if float(x) >= 0.0]
        return float(np.mean(ok)) if ok else -1.0

    if per_sample_mean_inception:
        if "fd_inception" in names:
            out["fd_inception"] = _mean_valid_inception(per_sample_fd)
        if "kd_inception" in names:
            out["kd_inception"] = _mean_valid_inception(per_sample_kd)
    else:
        if "fd_inception" in names:
            if phase_snapshots and gpu_idx is not None:
                snap = _nvidia_smi_query_gpu(gpu_idx)
                if snap is not None:
                    gpu_snapshots["before_fd_inception"] = snap
            _cuda_synchronize_if_needed(device)
            t0 = time.perf_counter()
            out["fd_inception"] = _compute_fd_inception(
                all_real,
                all_fake,
                device=device,
                inception_batch_size=inception_batch_size,
                profile=fd_profile,
            )
            _cuda_synchronize_if_needed(device)
            t_fd_sec = time.perf_counter() - t0
            if phase_snapshots and gpu_idx is not None:
                snap = _nvidia_smi_query_gpu(gpu_idx)
                if snap is not None:
                    gpu_snapshots["after_fd_inception"] = snap
            print(
                f"[render_metrics][debug] cuda_mem(after_fd_inception)={_torch_cuda_memory_stats(device)}",
                flush=True,
            )
        if "kd_inception" in names:
            if phase_snapshots and gpu_idx is not None:
                snap = _nvidia_smi_query_gpu(gpu_idx)
                if snap is not None:
                    gpu_snapshots["before_kd_inception"] = snap
            _cuda_synchronize_if_needed(device)
            t0 = time.perf_counter()
            out["kd_inception"] = _compute_kid_inception(
                all_real,
                all_fake,
                kid_subset,
                device=device,
                inception_batch_size=inception_batch_size,
                rbf_gamma=kid_rbf_gamma,
                profile=kid_profile,
            )
            _cuda_synchronize_if_needed(device)
            t_kd_sec = time.perf_counter() - t0
            if phase_snapshots and gpu_idx is not None:
                snap = _nvidia_smi_query_gpu(gpu_idx)
                if snap is not None:
                    gpu_snapshots["after_kd_inception"] = snap
            print(
                f"[render_metrics][debug] cuda_mem(after_kd_inception)={_torch_cuda_memory_stats(device)}",
                flush=True,
            )
    if "clip_score" in names:
        if clip_vals:
            clip_cos = float(np.mean(clip_vals))
            out["clip_score"] = 100.0 * clip_cos
        else:
            out["clip_score"] = -1.0

    if "psnr" in names:
        out["psnr"] = float(np.mean(per_pair_psnr)) if per_pair_psnr else -1.0
    if "ssim" in names:
        out["ssim"] = float(np.mean(per_pair_ssim)) if per_pair_ssim else -1.0

    if gpu_poller is not None:
        gpu_poller.stop()

    wall_total = time.perf_counter() - t_total0
    mesh_render_sum = t_render_ref_total + t_render_gen_total
    accounted = (
        clip_preload_sec
        + t_mesh_load_ref
        + t_mesh_load_gen
        + mesh_render_sum
        + t_save_images_total
        + t_clip_score_total
        + t_fd_sec
        + t_kd_sec
    )
    merge_rank: Dict[str, float] = {
        "clip_preload": clip_preload_sec,
        "mesh_load_reference_glb": t_mesh_load_ref,
        "mesh_load_generated_glb": t_mesh_load_gen,
        "pyvista_render_reference_views": t_render_ref_total,
        "pyvista_render_generated_views": t_render_gen_total,
        "save_metric_render_pngs": t_save_images_total,
        "clip_score_forward_all_samples": t_clip_score_total,
        "fd_inception_total": t_fd_sec,
        "kd_inception_total": t_kd_sec,
        "parallel_rgb_mesh_render_wall": parallel_render_wall_sec,
    }
    merge_rank.update({f"fd_inception::{k}": v for k, v in fd_profile.items()})
    merge_rank.update({f"kd_inception::{k}": v for k, v in kid_profile.items()})
    ranked = _rank_timings_sec(merge_rank, top=16)

    debug_payload: Dict[str, Any] = {
        "device": str(device),
        "gpu_index": gpu_idx,
        "metric_debug_keys_migrated_from_model": list(_migrated_debug_keys),
        "cuda_available": bool(torch.cuda.is_available()),
        "metrics_requested": list(names),
        "nviews": nviews,
        "resolution": resolution,
        "kid_subset_size": kid_subset,
        "kid_rbf_gamma_config": kid_rbf_gamma,
        "kid_inception_reported_as": "100 * MMD² (Gaussian RBF)",
        "kid_inception_output_scale": 100,
        "clip_model": clip_model,
        "inception_batch_size": inception_batch_size,
        "clip_batch_size": clip_batch_size,
        "mesh_render_backend_config": mesh_backend,
        "render_gpu_ids": list(render_gpu_ids),
        "render_worker_count": int(render_worker_count),
        "parallel_render_wall_sec": round(parallel_render_wall_sec, 6),
        "parallel_render_failures": int(render_worker_failures),
        "num_records": len(records),
        "samples_rendered": n_samples_rendered,
        "inception_aggregate": inception_agg,
        "pooled_view_count": len(all_real),
        "per_sample_inception_fd": list(per_sample_fd) if per_sample_mean_inception else None,
        "per_sample_inception_kd": list(per_sample_kd) if per_sample_mean_inception else None,
        "clip_scores_count": len(clip_vals),
        "clip_score_output_scale": 100,
        "clip_score_cosine_mean": float(np.mean(clip_vals)) if clip_vals else None,
        "timings_sec": {
            "clip_preload": round(clip_preload_sec, 6),
            "mesh_load_reference_glb_total": round(t_mesh_load_ref, 6),
            "mesh_load_generated_glb_total": round(t_mesh_load_gen, 6),
            "mesh_render_reference_views_total": round(t_render_ref_total, 6),
            "mesh_render_generated_views_total": round(t_render_gen_total, 6),
            "mesh_render_ref_plus_gen_total": round(mesh_render_sum, 6),
            "save_metric_render_pngs_total": round(t_save_images_total, 6),
            "clip_score_all_samples": round(t_clip_score_total, 6),
            "fd_inception": round(t_fd_sec, 6),
            "kd_inception": round(t_kd_sec, 6),
            "parallel_rgb_mesh_render_wall": round(parallel_render_wall_sec, 6),
            "wall_total": round(wall_total, 6),
            "accounted_major_phases": round(accounted, 6),
            "unaccounted_overhead": round(max(0.0, wall_total - accounted), 6),
        },
        "fd_inception_profile_sec": {k: round(v, 6) for k, v in fd_profile.items()},
        "kd_inception_profile_sec": {k: round(v, 6) for k, v in kid_profile.items()},
        "bottleneck_rank_sec": [{"phase": k, "sec": round(v, 6)} for k, v in ranked],
        "per_sample_timings": per_sample_rows,
        "gpu_snapshots_nvidia_smi": gpu_snapshots,
        "metric_values": {k: float(v) for k, v in out.items()},
        "cuda_memory": {
            "end": _torch_cuda_memory_stats(device),
        },
        "debug_flags": {
            "verbose_metrics_timeline": verbose_timeline,
            "debug_gpu_poll": poll_gpu,
            "debug_gpu_poll_interval_sec": poll_interval,
            "debug_gpu_phase_snapshots": phase_snapshots,
        },
    }
    if gpu_poller is not None:
        debug_payload["gpu_poll_summary"] = gpu_poller.summary()
        debug_payload["gpu_poll_interval_sec"] = poll_interval
    if phase_snapshots and gpu_idx is not None:
        snap = _nvidia_smi_query_gpu(gpu_idx)
        if snap is not None:
            gpu_snapshots["aggregate_end"] = snap
    if clip_model_obj is not None:
        debug_payload["clip_device_summary"] = _metric_device_summary(clip_model_obj)
    print(
        f"[render_metrics][debug] bottleneck_rank_sec (top phases): {ranked}",
        flush=True,
    )
    if debug_payload.get("gpu_poll_summary"):
        print(
            f"[render_metrics][debug] gpu_poll_summary={debug_payload['gpu_poll_summary']!r}",
            flush=True,
        )
    try:
        dbg_path = os.path.join(out_dir, "render_metrics_debug.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(dbg_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=False, indent=2)
        print(f"[render_metrics][debug] wrote timing log {dbg_path!r}", flush=True)
    except Exception as exc:
        print(f"[render_metrics][warn] could not write render_metrics_debug.json: {exc!r}", flush=True)

    print(
        f"[render_metrics][debug] aggregate done wall={wall_total:.3f}s "
        f"clip_preload={clip_preload_sec:.3f}s "
        f"mesh_load_ref={t_mesh_load_ref:.3f}s mesh_load_gen={t_mesh_load_gen:.3f}s "
        f"render_ref={t_render_ref_total:.3f}s render_gen={t_render_gen_total:.3f}s "
        f"(mesh_render_sum={mesh_render_sum:.3f}s) save_pngs={t_save_images_total:.3f}s "
        f"clip_score={t_clip_score_total:.3f}s fd_inception={t_fd_sec:.3f}s "
        f"kd_inception={t_kd_sec:.3f}s overhead={max(0.0, wall_total - accounted):.3f}s "
        f"samples_rendered={n_samples_rendered} inception_agg={inception_agg!r} "
        f"pooled_views={len(all_real)} per_sample_fd_n={len(per_sample_fd)} "
        f"per_sample_kd_n={len(per_sample_kd)} clip_n={len(clip_vals)}",
        flush=True,
    )
    return out


class RenderMetrics:
    """Unified interface (legacy + trellis object path)."""

    @staticmethod
    def compute(
        results: List[Dict],
        metric_names: List[str],
        resolution: int = 512,
        nviews: int = 30,
    ) -> Dict[str, float]:
        per_image_metrics: Dict[str, List[float]] = {"psnr": [], "ssim": [], "lpips": []}
        all_real_images: List[np.ndarray] = []
        all_gen_images: List[np.ndarray] = []

        for r in results:
            trellis_out = r.get("trellis_outputs")
            ref_path = r.get("reference_mesh_path")

            if trellis_out is None or ref_path is None:
                continue

            gen_rep = trellis_out.get("gaussian", [None])[0]
            if gen_rep is None:
                gen_rep = trellis_out.get("mesh", [None])[0]
            if gen_rep is None:
                continue

            gen_images = render_multiview_for_comparison(gen_rep, resolution, nviews)

            from trellis.utils.render_utils import render_multiview

            try:
                ref_images, _, _ = render_multiview(ref_path, resolution, nviews)
            except Exception:
                continue

            n = min(len(gen_images), len(ref_images))
            for i in range(n):
                gen_arr = np.ascontiguousarray(np.asarray(gen_images[i], dtype=np.uint8))
                ref_arr = np.ascontiguousarray(np.asarray(ref_images[i], dtype=np.uint8))
                gen_t = torch.from_numpy(gen_arr).float().permute(2, 0, 1) / 255.0
                ref_t = torch.from_numpy(ref_arr).float().permute(2, 0, 1) / 255.0

                if "psnr" in metric_names:
                    per_image_metrics["psnr"].append(compute_psnr(gen_t, ref_t))
                if "ssim" in metric_names:
                    per_image_metrics["ssim"].append(compute_ssim(gen_t, ref_t))
                if "lpips" in metric_names:
                    per_image_metrics["lpips"].append(compute_lpips(gen_t, ref_t))

            if "fid" in metric_names:
                all_real_images.extend(ref_images[:n])
                all_gen_images.extend(gen_images[:n])

        aggregated: Dict[str, float] = {}
        for name in ["psnr", "ssim", "lpips"]:
            if name in metric_names and per_image_metrics[name]:
                vals = per_image_metrics[name]
                aggregated[name] = sum(vals) / len(vals)

        if "fid" in metric_names and all_real_images:
            aggregated["fid"] = compute_fid(all_real_images, all_gen_images)

        return aggregated
