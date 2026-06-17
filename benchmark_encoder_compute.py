#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark encoder compute for SparseSDF VQVAE vs ShapeLLM VQVAE3D vs PointLLM point backbone.

This script intentionally keeps all benchmark logic in one file:
  1. generate a fixed random mesh file;
  2. build resolution-dependent dense voxel and sparse SDF-like inputs;
  3. run the three encoders (optional: skip PointLLM with ``--disable-pointllm``);
  4. estimate MACs/FLOPs with forward hooks;
  5. save a CSV and a multi-series plot.

python benchmark_encoder_compute.py \
  --shapellm-device cuda:0 --sparse-device cuda:0 \
  --pointllm-device cuda:0 \
  --dense-autocast \
  --continue-on-oom \
  --resolutions 32 48 64 128 256 512 1024

Notes:
  - ShapeLLM's official VQVAE3D.Encode hardcodes 64^3 -> 8*8*16 tokens.
    For variable-resolution profiling we call only VQVAE3D.Encoder.
  - Sparse convolution MACs are estimated as active output voxels * kernel volume
    * Cin * Cout. Real sparse kernels may do fewer operations depending on the
    neighborhood map, so treat these as comparable estimates.
  - VRAM: dense R^3 activations dominate at large R. Putting encoders on two GPUs
    splits **weights** across cards; it does not shard one dense forward across GPUs.
    Use ``--dense-autocast``, offload-on-shared-GPU (default), and/or
    ``--continue-on-oom`` for very large R.
  - Sparse on ``cuda:1+``: this script sets ``SPARSE_ATTN_BACKEND=xformers`` before
    importing Trellis sparse (unless you already set ``SPARSE_ATTN_BACKEND``), because
    ``flash_attn`` often hits illegal memory access on non-default GPUs.
  - PointLLM input size matches **ShapeLLM discretization**: number of distinct occupied
    voxels from projecting the same surface samples onto the ``R³`` grid (same rule as
    ``dense_voxels_from_surface``), not the sparse SDF active count.
  - PointLLM runs on ``--pointllm-device`` (default: same as ``--sparse-device``). Use
    ``--disable-pointllm`` to skip. ``--download-pointllm-config`` only fetches HF
    ``config.json`` metadata.
  - Dense ShapeLLM encoder cost is dominated by full ``R³`` tensors; estimated FLOPs
    typically scale roughly like ``R³`` (plus overhead). With ``--continue-on-oom``,
    OOM points can be filled by regressing FLOPs on ``R³`` from successful resolutions.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = REPO_ROOT / "eval_results" / "encoder_compute"
DEFAULT_SPARSE_CONFIG = REPO_ROOT / "eval" / "configs" / "vae" / "sdf_vqvae_stage2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare encoder compute under different input resolutions."
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 48, 64, 96],
        help="Input resolutions to benchmark. Use multiples of 8 for ShapeLLM encoder.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mesh-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Fallback when --shapellm-device / --sparse-device are omitted.",
    )
    parser.add_argument(
        "--shapellm-device",
        type=str,
        default=None,
        help="Device for ShapeLLM dense encoder (e.g. cuda:0). Default: cuda:0 if CUDA else cpu.",
    )
    parser.add_argument(
        "--sparse-device",
        type=str,
        default=None,
        help="Device for SparseSDF encoder. Default: cuda:1 when >=2 GPUs else same as ShapeLLM.",
    )
    parser.add_argument(
        "--no-offload-idle-encoder",
        action="store_true",
        help="When ShapeLLM and Sparse share one GPU, do NOT move the idle encoder to CPU "
        "before each forward (higher peak VRAM).",
    )
    parser.add_argument(
        "--dense-autocast",
        action="store_true",
        help="Run ShapeLLM dense forward under torch.autocast(fp16) on CUDA to reduce activation memory.",
    )
    parser.add_argument(
        "--continue-on-oom",
        action="store_true",
        help="On CUDA OOM, record status=oom and continue. After the sweep, fit each model's "
        "measured FLOPs as FLOPs≈a·R³+b on successful points and fill OOM rows (status=extrapolated).",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--sparse-threshold-voxels", type=float, default=0.5)
    parser.add_argument(
        "--sparse-config",
        type=Path,
        default=DEFAULT_SPARSE_CONFIG,
        help="SparseSDFVQVAE json config. Used to instantiate only its encoder path.",
    )
    parser.add_argument(
        "--sparse-ckpt",
        type=Path,
        default=None,
        help="Optional SparseSDFVQVAE checkpoint. If omitted, random weights are used.",
    )
    parser.add_argument(
        "--shapellm-ckpt",
        type=Path,
        default=None,
        help="Optional ShapeLLM 3DVQVAE checkpoint file. If omitted, random weights are used.",
    )
    parser.add_argument(
        "--download-shapellm-ckpt",
        action="store_true",
        help="Download yejunliang23/3DVQVAE from HuggingFace when --shapellm-ckpt is not set.",
    )
    parser.add_argument(
        "--max-sparse-points",
        type=int,
        default=250000,
        help="Cap sparse input points per resolution to avoid accidental OOM.",
    )
    parser.add_argument(
        "--disable-pointllm",
        action="store_true",
        help="Skip PointLLM point backbone benchmark (ShapeLLM + Sparse only).",
    )
    parser.add_argument(
        "--pointllm-device",
        type=str,
        default=None,
        help="Device for PointLLM point encoder. Default: same as --sparse-device.",
    )
    parser.add_argument(
        "--pointllm-no-color",
        action="store_true",
        help="Use xyz-only input (3 channels) instead of xyz+rgb (6 channels).",
    )
    parser.add_argument(
        "--pointllm-max-groups",
        type=int,
        default=512,
        help="FPS group count cap (PointLLM default 512). Capped by active sparse point count per resolution.",
    )
    parser.add_argument(
        "--pointllm-group-size",
        type=int,
        default=32,
        help="KNN neighborhood size per group (PointLLM default 32). Capped by active sparse point count.",
    )
    parser.add_argument(
        "--pointllm-hf-repo",
        type=str,
        default="RunsenXu/PointLLM_7B_v1.2",
        help="HuggingFace repo id for optional --download-pointllm-config (metadata only).",
    )
    parser.add_argument(
        "--download-pointllm-config",
        action="store_true",
        help="Fetch config.json from --pointllm-hf-repo (prints backbone fields; no weight download).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_repo_to_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _pick_other_cuda(first: torch.device) -> torch.device:
    """If multiple CUDA devices exist, return one different from ``first``; else ``first``."""
    if first.type != "cuda" or torch.cuda.device_count() < 2:
        return first
    i0 = int(first.index) if first.index is not None else 0
    for j in range(torch.cuda.device_count()):
        if j != i0:
            return torch.device(f"cuda:{j}")
    return first


def resolve_encoder_devices(
    *,
    primary: str,
    shapellm_s: Optional[str],
    sparse_s: Optional[str],
) -> Tuple[torch.device, torch.device]:
    """Pick devices for the two encoders: spread across visible GPUs when possible."""
    if shapellm_s and sparse_s:
        return torch.device(shapellm_s), torch.device(sparse_s)
    if shapellm_s and not sparse_s:
        a = torch.device(shapellm_s)
        return a, _pick_other_cuda(a)
    if sparse_s and not shapellm_s:
        b = torch.device(sparse_s)
        return _pick_other_cuda(b), b

    primary_dev = torch.device(primary)
    if primary_dev.type != "cuda":
        return primary_dev, primary_dev
    if torch.cuda.device_count() >= 2:
        return torch.device("cuda:0"), torch.device("cuda:1")
    return primary_dev, primary_dev


def module_device(module: nn.Module) -> torch.device:
    p = next(module.parameters(), None)
    if p is not None:
        return p.device
    b = next(module.buffers(), None)
    if b is not None:
        return b.device
    return torch.device("cpu")


def empty_cuda_caches() -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


@contextlib.contextmanager
def cuda_device_ctx(dev: torch.device) -> Iterable[None]:
    """Set current CUDA device for the block (helps some kernels/extensions)."""
    if dev.type != "cuda":
        yield
        return
    idx = int(dev.index) if dev.index is not None else 0
    prev = torch.cuda.current_device()
    if prev != idx:
        torch.cuda.set_device(idx)
    try:
        yield
    finally:
        if prev != idx:
            torch.cuda.set_device(prev)


def configure_sparse_attention_env(sparse_forward_device: torch.device) -> None:
    """
    Must run **before** the first ``import trellis.modules.sparse``.

    ``flash_attn`` paths frequently hit illegal memory access when the sparse
    model runs on a non-default GPU (e.g. ``cuda:1``). Switch to ``xformers`` in
    that case unless the user already set ``SPARSE_ATTN_BACKEND`` / ``ATTN_BACKEND``.
    """
    if os.environ.get("SPARSE_ATTN_BACKEND") or os.environ.get("ATTN_BACKEND"):
        return
    if sparse_forward_device.type == "cuda" and (sparse_forward_device.index or 0) != 0:
        os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
        print(
            "[Benchmark] SPARSE_ATTN_BACKEND=xformers "
            f"(sparse on {sparse_forward_device}; flash_attn is unreliable on non-cuda:0)."
        )


def format_num(x: float) -> str:
    units = ["", "K", "M", "G", "T", "P"]
    v = float(x)
    for unit in units:
        if abs(v) < 1000.0 or unit == units[-1]:
            return f"{v:.3f}{unit}"
        v /= 1000.0
    return f"{x:.3f}"


def generate_fixed_random_mesh(mesh_path: Path, seed: int) -> Path:
    import trimesh

    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.45)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    dirs = verts / np.maximum(np.linalg.norm(verts, axis=1, keepdims=True), 1e-12)
    # Deterministic low-amplitude radial perturbation: fixed but not a perfect sphere.
    phase = rng.normal(size=(3,))
    radial = (
        1.0
        + 0.10 * np.sin(7.0 * dirs[:, 0] + phase[0])
        + 0.07 * np.cos(5.0 * dirs[:, 1] - phase[1])
        + 0.05 * np.sin(6.0 * dirs[:, 2] + phase[2])
    )
    mesh.vertices = verts * radial[:, None]
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(0.9 / max(float(mesh.extents.max()), 1e-12))
    mesh.export(mesh_path)
    return mesh_path


def load_mesh_surface_points(mesh_path: Path, n: int, seed: int) -> np.ndarray:
    import trimesh

    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        points, _ = trimesh.sample.sample_surface(mesh, n)
    finally:
        np.random.set_state(rng_state)
    points = np.asarray(points, dtype=np.float32)
    center = (points.max(axis=0) + points.min(axis=0)) * 0.5
    scale = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1e-6)
    points = (points - center) / scale
    return np.clip(points, -0.499, 0.499)


def dense_voxels_from_surface(surface_points: np.ndarray, resolution: int) -> torch.Tensor:
    coords = np.floor((surface_points + 0.5) * resolution).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    vox = torch.zeros((1, 1, resolution, resolution, resolution), dtype=torch.float32)
    vox[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return vox


def count_shape_voxels_from_surface(surface_points: np.ndarray, resolution: int) -> int:
    """
    Count distinct occupied voxels for the same discretization as ``dense_voxels_from_surface``
    (surface points projected to an ``R³`` grid; duplicate bin indices collapse to one).
    """
    coords = np.floor((surface_points + 0.5) * resolution).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    return int(np.unique(coords, axis=0).shape[0])


def _neighbor_offsets(radius_voxels: float) -> np.ndarray:
    r = int(math.ceil(radius_voxels))
    offsets: List[Tuple[int, int, int]] = []
    for x in range(-r, r + 1):
        for y in range(-r, r + 1):
            for z in range(-r, r + 1):
                if math.sqrt(x * x + y * y + z * z) <= radius_voxels + 1e-6:
                    offsets.append((x, y, z))
    return np.asarray(offsets, dtype=np.int64)


def sparse_sdf_from_surface(
    surface_points: np.ndarray,
    resolution: int,
    threshold_voxels: float,
    max_points: int,
) -> Dict[str, torch.Tensor]:
    coords = np.floor((surface_points + 0.5) * resolution).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    coords = np.unique(coords, axis=0)

    offsets = _neighbor_offsets(threshold_voxels)
    band = coords[:, None, :] + offsets[None, :, :]
    band = band.reshape(-1, 3)
    valid = np.all((band >= 0) & (band < resolution), axis=1)
    band = np.unique(band[valid], axis=0)
    if band.shape[0] > max_points:
        idx = np.linspace(0, band.shape[0] - 1, max_points).astype(np.int64)
        band = band[idx]

    # Approximate unsigned surface distance in voxel units. The encoder benchmark
    # only needs realistic sparse feature density, not exact watertight SDF signs.
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords.astype(np.float32))
        dist, _ = tree.query(band.astype(np.float32), k=1)
        sdf = (dist.astype(np.float32) / max(threshold_voxels, 1e-6))[:, None]
    except Exception:
        src = torch.from_numpy(coords.astype(np.float32))
        dst = torch.from_numpy(band.astype(np.float32))
        chunks: List[torch.Tensor] = []
        for part in dst.split(8192):
            chunks.append(torch.cdist(part, src).min(dim=1).values)
        sdf = (torch.cat(chunks).numpy().astype(np.float32) / max(threshold_voxels, 1e-6))[:, None]

    return {
        "sparse_sdf": torch.from_numpy(sdf).float(),
        "sparse_index": torch.from_numpy(band.astype(np.int64)).long(),
        "batch_idx": torch.zeros((band.shape[0],), dtype=torch.long),
    }


def disable_checkpointing(module: nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "use_checkpoint"):
            try:
                setattr(m, "use_checkpoint", False)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# PointLLM point backbone (minimal in-file port of PointBERT / PointTransformer)
# Config matches PointTransformer_8192point_2layer.yaml from InternRobotics/PointLLM.
# ---------------------------------------------------------------------------


@dataclass
class PointLLMBackboneConfig:
    trans_dim: int = 384
    depth: int = 12
    drop_path_rate: float = 0.1
    cls_dim: int = 40
    num_heads: int = 6
    group_size: int = 32
    num_group: int = 512
    encoder_dims: int = 256
    point_dims: int = 6
    use_max_pool: bool = False


def fetch_pointllm_hf_config_json(repo_id: str) -> Optional[Dict[str, Any]]:
    """Stdlib-only fetch of ``config.json`` from a HuggingFace model repo."""
    if not repo_id or not str(repo_id).strip():
        return None
    url = f"https://huggingface.co/{repo_id.strip()}/resolve/main/config.json"
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "benchmark_encoder_compute"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[PointLLM] could not fetch HF config from {url!r}: {exc}")
        return None


def _pointllm_index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points [B,N,C], idx [B,S] -> [B,S,C]"""
    device = points.device
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(b, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def _pointllm_fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sample; xyz [B,N,3] -> [B,npoint,3] coordinates."""
    device = xyz.device
    b, n, _c = xyz.shape
    npoint = int(min(max(npoint, 1), n))
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.ones(b, n, device=device) * 1e10
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return _pointllm_index_points(xyz, centroids)


def _pointllm_square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    b, n, _ = src.shape
    _b2, m, _2 = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src**2, dim=-1).view(b, n, 1)
    dist += torch.sum(dst**2, dim=-1).view(b, 1, m)
    return dist


def _pointllm_knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Return idx [B,S,nsample] nearest neighbors in xyz for each new_xyz."""
    sqrdists = _pointllm_square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def pointllm_group_tokens(
    pts: torch.Tensor, num_group: int, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pts: [B, N, C] with C==3 or 6 (xyz [+-0.5] + optional rgb [0,1]).
    Returns neighborhood [B,G,M,C] and centers [B,G,3].
    """
    b, n, c = pts.shape
    if c > 3:
        xyz = pts[:, :, :3].contiguous()
        rgb = pts[:, :, 3:].contiguous()
    else:
        xyz = pts.contiguous()
        rgb = None
    num_group = int(min(max(num_group, 1), n))
    group_size = int(min(max(group_size, 1), n))
    center = _pointllm_fps(xyz, num_group)
    idx = _pointllm_knn_point(group_size, xyz, center)
    batch_size, _g, _m = idx.shape
    idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * n
    idx_flat = (idx + idx_base).view(-1)
    neighborhood_xyz = xyz.view(batch_size * n, -1)[idx_flat, :]
    neighborhood_xyz = neighborhood_xyz.view(batch_size, num_group, group_size, 3).contiguous()
    neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
    if rgb is not None:
        neighborhood_rgb = rgb.view(batch_size * n, -1)[idx_flat, :]
        neighborhood_rgb = neighborhood_rgb.view(batch_size, num_group, group_size, -1).contiguous()
        neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
    else:
        neighborhood = neighborhood_xyz
    return neighborhood, center


class PointLLMPointEncoder(nn.Module):
    """Local PointNet-style encoder over grouped points (per PointLLM)."""

    def __init__(self, encoder_channel: int, point_input_dims: int = 3) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.point_input_dims = point_input_dims
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        bs, g, n, c = point_groups.shape
        x = point_groups.reshape(bs * g, n, c)
        feature = self.first_conv(x.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class PointLLMPointAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PointLLMMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PointLLMTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attn = PointLLMPointAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.mlp = PointLLMMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PointLLMTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                PointLLMTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


class PointLLMPointTransformer(nn.Module):
    """Point backbone aligned with PointLLM v1.2 (12-layer, 384-dim, 512 groups x 32)."""

    def __init__(self, config: PointLLMBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.use_max_pool = config.use_max_pool
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.encoder_dims = config.encoder_dims
        self.point_dims = config.point_dims

        self.encoder = PointLLMPointEncoder(
            encoder_channel=self.encoder_dims, point_input_dims=self.point_dims
        )
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.blocks = PointLLMTransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=config.drop_path_rate,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

    def forward(
        self,
        pts: torch.Tensor,
        num_group: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> torch.Tensor:
        ng = int(num_group if num_group is not None else self.config.num_group)
        gs = int(group_size if group_size is not None else self.config.group_size)
        neighborhood, center = pointllm_group_tokens(pts, ng, gs)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        if not self.use_max_pool:
            return x
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(dim=1)[0]], dim=-1).unsqueeze(1)
        return concat_f


def load_pointllm_encoder(
    device: torch.device,
    *,
    point_dims: int,
) -> nn.Module:
    cfg = PointLLMBackboneConfig(point_dims=point_dims)
    enc = PointLLMPointTransformer(cfg).to(device).eval()
    disable_checkpointing(enc)
    print(
        f"[PointLLM] point backbone (random init): trans_dim={cfg.trans_dim}, depth={cfg.depth}, "
        f"heads={cfg.num_heads}, num_group<={cfg.num_group}, group_size<={cfg.group_size}, "
        f"point_dims={point_dims}, use_max_pool={cfg.use_max_pool}"
    )
    return enc


def random_pointllm_points(
    batch: int,
    n: int,
    point_dims: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Random point cloud; xyz in [-0.5,0.5], rgb in [0,1] when point_dims==6."""
    if point_dims == 6:
        pts = torch.rand((batch, n, 6), generator=generator, dtype=torch.float32)
        pts[:, :, :3] = pts[:, :, :3] - 0.5
        return pts.to(device)
    if point_dims == 3:
        return (torch.rand((batch, n, 3), generator=generator, dtype=torch.float32) - 0.5).to(device)
    raise ValueError(f"Unsupported point_dims={point_dims} (use 3 or 6).")


def count_parameters(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def _as_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "feats"):
        return x.feats
    return None


def _num_sparse_points(x: Any) -> Optional[int]:
    if hasattr(x, "feats"):
        return int(x.feats.shape[0])
    return None


def _triple(value: Any) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, Sequence):
        vals = list(value)
        if len(vals) == 1:
            return (int(vals[0]),) * 3
        return (int(vals[0]), int(vals[1]), int(vals[2]))
    return (1, 1, 1)


@dataclass
class ComputeStats:
    macs: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)

    def add(self, key: str, value: float) -> None:
        value = float(value)
        self.macs += value
        self.details[key] = self.details.get(key, 0.0) + value


class MacProfiler:
    def __init__(self, module: nn.Module):
        self.module = module
        self.stats = ComputeStats()
        self.handles: List[Any] = []

    def __enter__(self) -> "MacProfiler":
        for submodule in self.module.modules():
            name = submodule.__class__.__name__
            if isinstance(submodule, nn.Conv3d):
                self.handles.append(submodule.register_forward_hook(self._conv3d_hook))
            elif isinstance(submodule, nn.Conv1d):
                self.handles.append(submodule.register_forward_hook(self._conv1d_hook))
            elif isinstance(submodule, nn.Linear):
                self.handles.append(submodule.register_forward_hook(self._linear_hook))
            elif name == "SparseConv3d":
                self.handles.append(submodule.register_forward_hook(self._sparse_conv_hook))
            elif name == "SparseMultiHeadAttention":
                self.handles.append(submodule.register_forward_hook(self._sparse_attn_hook))
            elif name == "PointLLMPointAttention":
                self.handles.append(submodule.register_forward_hook(self._pointllm_attn_hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _conv3d_hook(self, module: nn.Conv3d, inputs: Tuple[Any, ...], output: Any) -> None:
        x = _as_tensor(inputs[0])
        y = _as_tensor(output)
        if x is None or y is None or y.dim() != 5:
            return
        kernel_ops = int(np.prod(module.kernel_size)) * (module.in_channels // module.groups)
        out_elems = int(y.shape[0] * y.shape[2] * y.shape[3] * y.shape[4] * module.out_channels)
        self.stats.add("dense_conv3d", out_elems * kernel_ops)

    def _conv1d_hook(self, module: nn.Conv1d, inputs: Tuple[Any, ...], output: Any) -> None:
        x = inputs[0]
        y = output
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            return
        if x.dim() != 3 or y.dim() != 3:
            return
        k = module.kernel_size[0] if isinstance(module.kernel_size, (tuple, list)) else int(module.kernel_size)
        l_out = int(y.shape[2])
        g = max(int(module.groups), 1)
        ic = int(module.in_channels)
        oc = int(module.out_channels)
        ops = l_out * int(k) * (ic // g) * oc
        self.stats.add("dense_conv1d", float(ops))

    def _pointllm_attn_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        x = inputs[0]
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return
        b, n, c = x.shape
        self.stats.add("pointllm_attention_est", 2.0 * float(b * n * n * c))

    def _linear_hook(self, module: nn.Linear, inputs: Tuple[Any, ...], output: Any) -> None:
        x = _as_tensor(inputs[0])
        if x is None or x.numel() == 0:
            return
        tokens = int(x.numel() // max(int(module.in_features), 1))
        self.stats.add("linear", tokens * int(module.in_features) * int(module.out_features))

    def _sparse_conv_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        x = inputs[0] if inputs else None
        y_nnz = _num_sparse_points(output)
        x_feats = _as_tensor(x)
        if y_nnz is None or x_feats is None:
            return
        conv = getattr(module, "conv", None)
        in_ch = int(getattr(conv, "in_channels", x_feats.shape[-1]))
        out_ch = int(getattr(conv, "out_channels", getattr(_as_tensor(output), "shape", [0, x_feats.shape[-1]])[-1]))
        kernel_size = getattr(conv, "kernel_size", 1)
        if hasattr(kernel_size, "tolist"):
            kernel_size = kernel_size.tolist()
        kvol = int(np.prod(_triple(kernel_size)))
        self.stats.add("sparse_conv3d_est", y_nnz * in_ch * out_ch * kvol)

    def _sparse_attn_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        x = inputs[0] if inputs else None
        feats = _as_tensor(x)
        if feats is None or feats.numel() == 0:
            return
        n = int(feats.shape[0])
        c = int(getattr(module, "channels", feats.shape[-1]))
        mode = str(getattr(module, "attn_mode", "full"))
        if mode == "full":
            self.stats.add("sparse_attention_est", 2.0 * n * n * c)
            return
        window_size = getattr(module, "window_size", None) or 1
        if isinstance(window_size, Sequence):
            window_tokens = int(np.prod([max(int(v), 1) for v in window_size]))
        else:
            window_tokens = max(int(window_size), 1) ** 3
        self.stats.add("sparse_attention_est", 2.0 * n * min(n, window_tokens) * c)


@contextlib.contextmanager
def suppress_stdout(enabled: bool = True) -> Iterable[None]:
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_forward(
    fn: Any,
    warmup: int,
    repeats: int,
    sync_device: torch.device,
    suppress_logs: bool = False,
) -> Tuple[Any, float]:
    out = None
    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            with suppress_stdout(suppress_logs):
                out = fn()
        sync_if_needed(sync_device)
        times: List[float] = []
        for _ in range(max(repeats, 1)):
            start = time.perf_counter()
            with suppress_stdout(suppress_logs):
                out = fn()
            sync_if_needed(sync_device)
            times.append((time.perf_counter() - start) * 1000.0)
    return out, float(np.mean(times))


def profile_forward(
    module: nn.Module,
    fn: Any,
    warmup: int,
    repeats: int,
    sync_device: torch.device,
    suppress_logs: bool = False,
) -> Tuple[Any, ComputeStats, float]:
    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            with suppress_stdout(suppress_logs):
                fn()
        sync_if_needed(sync_device)

    with MacProfiler(module) as profiler:
        with torch.inference_mode(), suppress_stdout(suppress_logs):
            out = fn()
        sync_if_needed(sync_device)

    _, latency_ms = time_forward(fn, 0, repeats, sync_device, suppress_logs=suppress_logs)
    return out, profiler.stats, latency_ms


def load_shapellm_encoder(
    device: torch.device,
    ckpt: Optional[Path],
    download: bool,
) -> nn.Module:
    from trellis.models.sparse_structure_vqvae import VQVAE3D

    vqvae = VQVAE3D(num_embeddings=8192)
    if ckpt is None and download:
        from huggingface_hub import hf_hub_download

        ckpt = Path(hf_hub_download(repo_id="yejunliang23/3DVQVAE", filename="3DVQVAE.bin"))
    if ckpt is not None and ckpt.is_file():
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        vqvae.load_state_dict(state, strict=False)
        print(f"[ShapeLLM] loaded VQVAE checkpoint: {ckpt}")
    else:
        print("[ShapeLLM] no checkpoint provided; using random encoder weights.")
    enc = vqvae.Encoder.to(device).eval()
    if device.type == "cpu" and hasattr(enc, "convert_to_fp32"):
        enc.convert_to_fp32()
    disable_checkpointing(enc)
    return enc


def load_sparse_encoder(
    device: torch.device,
    config_path: Path,
    ckpt: Optional[Path],
) -> nn.Module:
    os.environ["SPARSE_BACKEND"] = "torchsparse"
    from trellis.modules import sparse as sp

    sp.set_sparse_backend("torchsparse")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vae_args = dict(cfg["models"]["vqvae"]["args"])

    state = None
    if ckpt is not None and ckpt.is_file():
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        emb = state.get("vq.embeddings.weight") if isinstance(state, dict) else None
        embed_dim = int(vae_args.get("embed_dim") or vae_args.get("latent_channels") or 0)
        if emb is not None and embed_dim > 0 and int(emb.shape[1]) % embed_dim == 0:
            vae_args["vq_group_size"] = int(emb.shape[1]) // embed_dim

    from trellis.models import SparseSDFVQVAE

    vae = SparseSDFVQVAE(**vae_args)
    if state is not None:
        vae.load_state_dict(state, strict=False)
        print(f"[SparseSDF] loaded VQVAE checkpoint: {ckpt}")
    else:
        print("[SparseSDF] no checkpoint provided; using random encoder weights.")
    enc = vae.encoder.to(device).eval()
    disable_checkpointing(enc)
    return enc


def make_sparse_tensor(batch: Dict[str, torch.Tensor], device: torch.device) -> Any:
    from trellis.modules import sparse as sp

    feats = batch["sparse_sdf"].to(device=device, dtype=torch.float32)
    xyz = batch["sparse_index"].to(device=device, dtype=torch.long)
    batch_idx = batch["batch_idx"].to(device=device, dtype=torch.long)
    coords = torch.cat([batch_idx[:, None], xyz], dim=1).long()
    coord_max = int(xyz.max().item()) + 1
    pack = 1
    while pack <= coord_max:
        pack *= 2
    keys = (
        coords[:, 0] * (pack**3)
        + coords[:, 1] * (pack**2)
        + coords[:, 2] * pack
        + coords[:, 3]
    )
    _, perm = torch.sort(keys)
    feats = feats[perm]
    coords = coords[perm].int()
    return sp.SparseTensor(feats, coords)


def prepare_encoder_pair(
    shapellm_encoder: nn.Module,
    sparse_encoder: nn.Module,
    shapellm_dev: torch.device,
    sparse_dev: torch.device,
    *,
    active: str,
    offload_idle: bool,
) -> None:
    """
    ``active`` is ``\"shapellm\"`` or ``\"sparse\"``.
    When both encoders share one CUDA device and ``offload_idle`` is True, move the idle
    one to CPU before the active forward to reduce peak VRAM on that device.
    """
    prepare_three_encoders(
        shapellm_encoder,
        sparse_encoder,
        None,
        shapellm_dev,
        sparse_dev,
        sparse_dev,
        active=active,
        offload_idle=offload_idle,
    )


def prepare_three_encoders(
    shapellm_encoder: nn.Module,
    sparse_encoder: nn.Module,
    pointllm_encoder: Optional[nn.Module],
    shapellm_dev: torch.device,
    sparse_dev: torch.device,
    pointllm_dev: torch.device,
    *,
    active: str,
    offload_idle: bool,
) -> None:
    """
    Activate one of ``shapellm`` / ``sparse`` / ``pointllm``. When ``offload_idle`` is True,
    any other encoder that shares the active device is moved to CPU first.
    """
    if pointllm_encoder is None:
        same = shapellm_dev == sparse_dev
        if active == "shapellm":
            if offload_idle and same:
                sparse_encoder.cpu()
                gc.collect()
                empty_cuda_caches()
            shapellm_encoder.to(shapellm_dev)
            return
        if active == "sparse":
            if offload_idle and same:
                shapellm_encoder.cpu()
                gc.collect()
                empty_cuda_caches()
            sparse_encoder.to(sparse_dev)
            return
        raise ValueError(active)

    if not offload_idle:
        shapellm_encoder.to(shapellm_dev)
        sparse_encoder.to(sparse_dev)
        pointllm_encoder.to(pointllm_dev)
        return

    if active == "shapellm":
        if sparse_dev == shapellm_dev:
            sparse_encoder.cpu()
        if pointllm_dev == shapellm_dev:
            pointllm_encoder.cpu()
        gc.collect()
        empty_cuda_caches()
        shapellm_encoder.to(shapellm_dev)
        return
    if active == "sparse":
        if shapellm_dev == sparse_dev:
            shapellm_encoder.cpu()
        if pointllm_dev == sparse_dev:
            pointllm_encoder.cpu()
        gc.collect()
        empty_cuda_caches()
        sparse_encoder.to(sparse_dev)
        return
    if active == "pointllm":
        if shapellm_dev == pointllm_dev:
            shapellm_encoder.cpu()
        if sparse_dev == pointllm_dev:
            sparse_encoder.cpu()
        gc.collect()
        empty_cuda_caches()
        pointllm_encoder.to(pointllm_dev)
        return
    raise ValueError(active)


def output_token_count(out: Any) -> int:
    if hasattr(out, "feats"):
        return int(out.feats.shape[0])
    if isinstance(out, torch.Tensor):
        return int(out.numel())
    return 0


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def fit_flops_linear_in_r_cubed(
    resolutions: np.ndarray, flops: np.ndarray
) -> Tuple[float, float]:
    """
    Fit ``flops ≈ a * R^3 + b`` (least squares in ``R^3``).

    With a single (R, flops) sample, use ``b=0`` and ``a=flops/R^3`` so extrapolation
    follows pure ``R^3`` scaling from that point.
    """
    R3 = np.maximum(resolutions.astype(np.float64), 0.0) ** 3
    y = flops.astype(np.float64)
    if R3.size < 2:
        denom = float(R3[0]) if R3.size else 1.0
        a = float(y[0] / max(denom, 1e-30))
        return a, 0.0
    X = np.column_stack([R3, np.ones_like(R3)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def extrapolate_oom_rows_cubic_volume(rows: List[Dict[str, Any]]) -> int:
    """
    For each model, replace ``status=oom`` rows using ``FLOPs ≈ a·R³+b`` fit on ``status=ok`` rows.

    Returns the number of rows updated.
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(str(r["model"]), []).append(r)

    filled = 0
    for _model, mrows in by_model.items():
        ok_rows = [
            r
            for r in mrows
            if str(r.get("status", "")).lower() == "ok" and _safe_float(r.get("flops")) > 0
        ]
        oom_rows = [r for r in mrows if str(r.get("status", "")).lower() == "oom"]
        if not oom_rows or not ok_rows:
            continue

        R_ok = np.array([int(r["resolution"]) for r in ok_rows], dtype=np.float64)
        F_ok = np.array([_safe_float(r.get("flops")) for r in ok_rows], dtype=np.float64)
        a, b = fit_flops_linear_in_r_cubed(R_ok, F_ok)

        for r in oom_rows:
            R = int(r["resolution"])
            pred = max(0.0, a * (float(R) ** 3) + b)
            fi = int(round(pred))
            r["flops"] = fi
            r["macs"] = max(0, fi // 2)
            r["latency_ms"] = ""
            r["status"] = "extrapolated"
            prev = str(r.get("error_message") or "")
            note = (
                f"cubic_volume_extrapolation: flops≈{a:.6g}*R^3+{b:.6g} "
                f"(fit on n={len(ok_rows)} measured points)"
            )
            r["error_message"] = (prev + " | " + note if prev.strip() else note)[:4000]
            r["mac_breakdown"] = json.dumps(
                {"method": "cubic_volume_lstsq", "a": a, "b": b, "n_fit": len(ok_rows)},
                ensure_ascii=False,
            )
            filled += 1

        print(
            f"[Extrapolation] {_model}: filled {len(oom_rows)} OOM row(s) with "
            f"FLOPs≈{a:.6g}·R³+{b:.6g} (n_fit={len(ok_rows)})"
        )
    return filled


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "resolution",
        "input_points_or_voxels",
        "encoder_output_elements",
        "params",
        "macs",
        "flops",
        "latency_ms",
        "status",
        "error_message",
        "mac_breakdown",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# def plot_csv(csv_path: Path, pdf_path: Path) -> None:
#     import matplotlib.pyplot as plt

#     plt.rcParams.update(
#         {
#             "font.size": 16,
#             "axes.labelsize": 18,
#             "xtick.labelsize": 15,
#             "ytick.labelsize": 15,
#             "legend.fontsize": 14,
#             "pdf.fonttype": 42,
#             "ps.fonttype": 42,
#         }
#     )

#     rows: List[Dict[str, Any]] = []
#     with open(csv_path, newline="", encoding="utf-8") as f:
#         rows.extend(csv.DictReader(f))
#     models = sorted({r["model"] for r in rows})
#     markers = ["o", "s", "^", "D", "P", "X"]
#     plt.figure(figsize=(8.2, 5.2))
#     for i, model in enumerate(models):
#         sub = [r for r in rows if r["model"] == model]
#         sub.sort(key=lambda r: int(r["resolution"]))
#         xs: List[int] = []
#         ys: List[float] = []
#         for r in sub:
#             st = str(r.get("status", "")).strip().lower()
#             if st not in {"", "ok", "extrapolated"}:
#                 continue
#             try:
#                 params_b = float(r["params"]) / 1e9
#                 if params_b <= 0:
#                     continue
#                 y_val = (float(r["flops"]) / 1e12) / params_b
#                 if y_val <= 0 or not math.isfinite(y_val):
#                     continue
#                 ys.append(y_val)
#                 xs.append(int(r["resolution"]))
#             except (ValueError, TypeError):
#                 continue
#         if xs:
#             plt.plot(
#                 xs,
#                 ys,
#                 marker=markers[i % len(markers)],
#                 markersize=7,
#                 linewidth=2.4,
#                 label=model,
#             )
#     plt.xlabel("Input Resolution (batch size = 1)")
#     plt.ylabel("TFLOPs / Params (B)")
#     ax = plt.gca()
#     ax.set_yscale("log")
#     ax.grid(True, which="major", linestyle="--", alpha=0.28)
#     ax.grid(True, which="minor", linestyle=":", alpha=0.15)
#     handles, labels = ax.get_legend_handles_labels()
#     if handles:
#         ax.legend(frameon=True)
#     plt.tight_layout()
#     pdf_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(pdf_path, bbox_inches="tight")
#     plt.close()

def plot_csv(csv_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    models = sorted({r["model"] for r in rows})
    
    # 定义配色方案（参考PDF）
    color_map = {
        "ShapeLLM": "#E74C3C",  # 红橙色
        "SPARC": "#3498DB",     # 蓝色
        "PointLLM": "#2ECC71",  # 绿色
    }
    
    markers = ["o", "s", "^", "D", "P", "X"]
    plt.figure(figsize=(8.2, 5.2))
    
    for i, model in enumerate(models):
        sub = [r for r in rows if r["model"] == model]
        sub.sort(key=lambda r: int(r["resolution"]))
        xs: List[int] = []
        ys: List[float] = []
        for r in sub:
            st = str(r.get("status", "")).strip().lower()
            if st not in {"", "ok", "extrapolated"}:
                continue
            try:
                params_b = float(r["params"]) / 1e9
                if params_b <= 0:
                    continue
                y_val = (float(r["flops"]) / 1e12) / params_b
                if y_val <= 0 or not math.isfinite(y_val):
                    continue
                ys.append(y_val)
                xs.append(int(r["resolution"]))
            except (ValueError, TypeError):
                continue
        if xs:
            # 使用自定义颜色
            color = color_map.get(model, f"C{i}")
            plt.plot(
                xs,
                ys,
                marker=markers[i % len(markers)],
                markersize=7,
                linewidth=2.4,
                label=model,
                color=color,
            )
    plt.xlabel("Input Resolution (batch size = 1)")
    plt.ylabel("TFLOPs / Params (B)")
    ax = plt.gca()
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=True)
    plt.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    add_repo_to_path()

    shapellm_dev, sparse_dev = resolve_encoder_devices(
        primary=args.device,
        shapellm_s=args.shapellm_device,
        sparse_s=args.sparse_device,
    )
    pointllm_dev = torch.device(args.pointllm_device) if args.pointllm_device else sparse_dev
    enable_pointllm = not args.disable_pointllm
    point_dims = 3 if args.pointllm_no_color else 6
    offload_idle = not args.no_offload_idle_encoder
    same_slot = shapellm_dev == sparse_dev

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.mesh_path or (args.out_dir / "fixed_random_mesh.obj")
    generate_fixed_random_mesh(mesh_path, args.seed)
    surface = load_mesh_surface_points(mesh_path, args.surface_samples, args.seed)

    print(f"[Benchmark] mesh: {mesh_path}")
    print(
        f"[Benchmark] ShapeLLM device: {shapellm_dev}, Sparse device: {sparse_dev}"
        + (f", PointLLM device: {pointllm_dev}" if enable_pointllm else "")
    )
    print(f"[Benchmark] torch.cuda.device_count()={torch.cuda.device_count()}")
    print(
        f"[Benchmark] offload idle encoder when sharing one device: "
        f"{bool(offload_idle and same_slot)}"
    )
    print(f"[Benchmark] PointLLM enabled: {enable_pointllm}")
    print(f"[Benchmark] dense autocast: {bool(args.dense_autocast)}")
    print(f"[Benchmark] continue on OOM: {bool(args.continue_on_oom)}")
    print(f"[Benchmark] resolutions: {args.resolutions}")

    if args.download_pointllm_config:
        meta = fetch_pointllm_hf_config_json(args.pointllm_hf_repo)
        if meta:
            print(
                f"[PointLLM] HF metadata ({args.pointllm_hf_repo}): "
                f"point_backbone={meta.get('point_backbone')!s}, "
                f"point_backbone_config_name={meta.get('point_backbone_config_name')!s}, "
                f"use_color={meta.get('use_color')!s}"
            )

    shapellm_encoder = load_shapellm_encoder(
        shapellm_dev, args.shapellm_ckpt, args.download_shapellm_ckpt
    )
    configure_sparse_attention_env(sparse_dev)
    sparse_init_dev = torch.device("cpu") if (same_slot and offload_idle) else sparse_dev
    sparse_encoder = load_sparse_encoder(sparse_init_dev, args.sparse_config, args.sparse_ckpt)
    if same_slot and offload_idle:
        sparse_encoder.cpu()

    pointllm_encoder: Optional[nn.Module] = None
    if enable_pointllm:
        pointllm_encoder = load_pointllm_encoder(torch.device("cpu"), point_dims=point_dims)

    rows: List[Dict[str, Any]] = []
    for res in args.resolutions:
        if res % 8 != 0:
            raise ValueError(f"ShapeLLM encoder expects resolutions divisible by 8, got {res}.")

        n_shape_occupied = count_shape_voxels_from_surface(surface, res)

        prepare_three_encoders(
            shapellm_encoder,
            sparse_encoder,
            pointllm_encoder,
            shapellm_dev,
            sparse_dev,
            pointllm_dev,
            active="shapellm",
            offload_idle=offload_idle,
        )
        sync_dev_sh = module_device(shapellm_encoder)
        dense_dtype = (
            torch.float16
            if args.dense_autocast and shapellm_dev.type == "cuda"
            else torch.float32
        )
        dense = dense_voxels_from_surface(surface, res).to(shapellm_dev, dtype=dense_dtype)

        def dense_fn() -> Any:
            with torch.inference_mode():
                if args.dense_autocast and shapellm_dev.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        return shapellm_encoder(dense)
                return shapellm_encoder(dense)

        try:
            dense_out, dense_stats, dense_latency = profile_forward(
                shapellm_encoder,
                dense_fn,
                args.warmup,
                args.repeats,
                sync_dev_sh,
            )
            rows.append(
                {
                    "model": "ShapeLLM",
                    "resolution": res,
                    "input_points_or_voxels": int(res**3),
                    "encoder_output_elements": output_token_count(dense_out),
                    "params": count_parameters(shapellm_encoder),
                    "macs": int(dense_stats.macs),
                    "flops": int(dense_stats.macs * 2),
                    "latency_ms": f"{dense_latency:.4f}",
                    "status": "ok",
                    "error_message": "",
                    "mac_breakdown": json.dumps(dense_stats.details, ensure_ascii=False),
                }
            )
        except RuntimeError as exc:
            msg = str(exc)
            is_oom = "out of memory" in msg.lower()
            if args.continue_on_oom and is_oom:
                rows.append(
                    {
                        "model": "ShapeLLM",
                        "resolution": res,
                        "input_points_or_voxels": int(res**3),
                        "encoder_output_elements": 0,
                        "params": count_parameters(shapellm_encoder),
                        "macs": 0,
                        "flops": 0,
                        "latency_ms": "",
                        "status": "oom",
                        "error_message": msg[:4000],
                        "mac_breakdown": "{}",
                    }
                )
                gc.collect()
                empty_cuda_caches()
            else:
                raise

        if shapellm_dev.type == "cuda":
            sync_if_needed(shapellm_dev)

        prepare_three_encoders(
            shapellm_encoder,
            sparse_encoder,
            pointllm_encoder,
            shapellm_dev,
            sparse_dev,
            pointllm_dev,
            active="sparse",
            offload_idle=offload_idle,
        )
        sync_dev_sp = module_device(sparse_encoder)
        sparse_batch = sparse_sdf_from_surface(
            surface,
            res,
            args.sparse_threshold_voxels,
            args.max_sparse_points,
        )
        sparse_tensor = make_sparse_tensor(sparse_batch, sparse_dev)

        def sparse_fn() -> Any:
            with torch.inference_mode(), cuda_device_ctx(sparse_dev):
                return sparse_encoder(sparse_tensor)

        try:
            sparse_out, sparse_stats, sparse_latency = profile_forward(
                sparse_encoder,
                sparse_fn,
                args.warmup,
                args.repeats,
                sync_dev_sp,
                suppress_logs=True,
            )
            rows.append(
                {
                    "model": "SPARC",
                    "resolution": res,
                    "input_points_or_voxels": int(sparse_batch["sparse_sdf"].shape[0]),
                    "encoder_output_elements": output_token_count(sparse_out),
                    "params": count_parameters(sparse_encoder),
                    "macs": int(sparse_stats.macs),
                    "flops": int(sparse_stats.macs * 2),
                    "latency_ms": f"{sparse_latency:.4f}",
                    "status": "ok",
                    "error_message": "",
                    "mac_breakdown": json.dumps(sparse_stats.details, ensure_ascii=False),
                }
            )
        except RuntimeError as exc:
            msg = str(exc)
            is_oom = "out of memory" in msg.lower()
            if args.continue_on_oom and is_oom:
                rows.append(
                    {
                        "model": "SPARC",
                        "resolution": res,
                        "input_points_or_voxels": int(sparse_batch["sparse_sdf"].shape[0]),
                        "encoder_output_elements": 0,
                        "params": count_parameters(sparse_encoder),
                        "macs": 0,
                        "flops": 0,
                        "latency_ms": "",
                        "status": "oom",
                        "error_message": msg[:4000],
                        "mac_breakdown": "{}",
                    }
                )
                gc.collect()
                empty_cuda_caches()
            else:
                raise

        if enable_pointllm and pointllm_encoder is not None:
            n_pts = n_shape_occupied
            prepare_three_encoders(
                shapellm_encoder,
                sparse_encoder,
                pointllm_encoder,
                shapellm_dev,
                sparse_dev,
                pointllm_dev,
                active="pointllm",
                offload_idle=offload_idle,
            )
            sync_dev_pl = module_device(pointllm_encoder)
            if n_pts < 1:
                rows.append(
                    {
                        "model": "PointLLM",
                        "resolution": res,
                        "input_points_or_voxels": 0,
                        "encoder_output_elements": 0,
                        "params": count_parameters(pointllm_encoder),
                        "macs": 0,
                        "flops": 0,
                        "latency_ms": "",
                        "status": "skipped",
                        "error_message": "no occupied voxels in ShapeLLM dense grid for this resolution",
                        "mac_breakdown": "{}",
                    }
                )
            else:
                ng = min(int(args.pointllm_max_groups), n_pts)
                gs = min(int(args.pointllm_group_size), n_pts)
                gen = torch.Generator()
                gen.manual_seed(int(args.seed) + int(res) * 1_000_003)
                pts = random_pointllm_points(1, n_pts, point_dims, sync_dev_pl, gen)

                def pointllm_fn() -> Any:
                    with torch.inference_mode(), cuda_device_ctx(pointllm_dev):
                        return pointllm_encoder(pts, num_group=ng, group_size=gs)

                try:
                    pl_out, pl_stats, pl_latency = profile_forward(
                        pointllm_encoder,
                        pointllm_fn,
                        args.warmup,
                        args.repeats,
                        sync_dev_pl,
                    )
                    rows.append(
                        {
                            "model": "PointLLM",
                            "resolution": res,
                            "input_points_or_voxels": n_pts,
                            "encoder_output_elements": output_token_count(pl_out),
                            "params": count_parameters(pointllm_encoder),
                            "macs": int(pl_stats.macs),
                            "flops": int(pl_stats.macs * 2),
                            "latency_ms": f"{pl_latency:.4f}",
                            "status": "ok",
                            "error_message": "",
                            "mac_breakdown": json.dumps(pl_stats.details, ensure_ascii=False),
                        }
                    )
                except RuntimeError as exc:
                    msg = str(exc)
                    is_oom = "out of memory" in msg.lower()
                    if args.continue_on_oom and is_oom:
                        rows.append(
                            {
                                "model": "PointLLM",
                                "resolution": res,
                                "input_points_or_voxels": n_pts,
                                "encoder_output_elements": 0,
                                "params": count_parameters(pointllm_encoder),
                                "macs": 0,
                                "flops": 0,
                                "latency_ms": "",
                                "status": "oom",
                                "error_message": msg[:4000],
                                "mac_breakdown": "{}",
                            }
                        )
                        gc.collect()
                        empty_cuda_caches()
                    else:
                        raise

        shape_row = rows[-3] if enable_pointllm else rows[-2]
        sparse_row = rows[-2] if enable_pointllm else rows[-1]
        pl_row: Optional[Dict[str, Any]] = rows[-1] if enable_pointllm else None
        if shape_row.get("status") == "ok" and sparse_row.get("status") == "ok":
            line = (
                f"[R={res}] ShapeLLM FLOPs={format_num(float(shape_row['flops']))} "
                f"({shape_row['latency_ms']} ms), Sparse FLOPs={format_num(float(sparse_row['flops']))} "
                f"({sparse_row['latency_ms']} ms)"
            )
            if pl_row is not None:
                line += (
                    f", PointLLM FLOPs={format_num(float(pl_row.get('flops') or 0))} "
                    f"({pl_row.get('latency_ms')!s} ms, status={pl_row.get('status')!r})"
                )
            print(line)
        else:
            line = (
                f"[R={res}] ShapeLLM status={shape_row.get('status')!r}, "
                f"Sparse status={sparse_row.get('status')!r}"
            )
            if pl_row is not None:
                line += f", PointLLM status={pl_row.get('status')!r}"
            print(line)

    if args.continue_on_oom:
        n_fill = extrapolate_oom_rows_cubic_volume(rows)
        if n_fill:
            print(f"[Extrapolation] total rows filled: {n_fill}")

    csv_path = args.out_dir / "encoder_compute.csv"
    pdf_path = args.out_dir / "encoder_compute.pdf"
    write_csv(rows, csv_path)
    plot_csv(csv_path, pdf_path)
    print(f"[Done] CSV saved to: {csv_path}")
    print(f"[Done] Plot saved to: {pdf_path}")


if __name__ == "__main__":
    main()
