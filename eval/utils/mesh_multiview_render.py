"""Multi-view mesh renders: CUDA normals, or RGB mesh renders for image metrics."""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class _RasterMeshView:
    """
    Duck-type compatible with ``MeshRenderer`` for ``return_types=['normal']``:
    ``vertices``, ``faces`` (long), ``face_normal`` (F*3, 3) as in Trellis ``MeshExtractResult``.
    """

    __slots__ = ("vertices", "faces", "face_normal", "vertex_attrs", "success")

    def __init__(self, vertices: "torch.Tensor", faces: "torch.Tensor"):
        import torch

        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = None
        self.face_normal = self._face_normals_per_corner(vertices, self.faces)
        self.success = vertices.shape[0] != 0 and faces.shape[0] != 0

    @staticmethod
    def _face_normals_per_corner(verts: "torch.Tensor", faces: "torch.Tensor") -> "torch.Tensor":
        import torch

        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()
        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        return face_normals[:, None, :].repeat(1, 3, 1)


_DEFAULT_RENDER_FOV_DEG = 40.0
_DEFAULT_CAMERA_MARGIN = 1.35
_FIXED_VIEW_DIRECTIONS: Tuple[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]], ...] = (
    ("front", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("right", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("top", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("axis", (1.0, -1.0, 0.75), (0.0, 0.0, 1.0)),
)


def _yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws: List[float],
    pitchs: List[float],
    rs: float | List[float],
    fovs: float | List[float],
    device: "torch.device",
):
    """Same geometry as ``trellis.utils.render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics`` but on ``device``."""
    import torch
    import utils3d

    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics: List["torch.Tensor"] = []
    intrinsics: List["torch.Tensor"] = []
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        z_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        fov_t = torch.deg2rad(torch.tensor(float(fov), dtype=torch.float32, device=device))
        yaw_t = torch.tensor(float(yaw), dtype=torch.float32, device=device)
        pitch_t = torch.tensor(float(pitch), dtype=torch.float32, device=device)
        orig = (
            torch.stack(
                [
                    torch.sin(yaw_t) * torch.cos(pitch_t),
                    torch.cos(yaw_t) * torch.cos(pitch_t),
                    torch.sin(pitch_t),
                ]
            )
            * float(r)
        )
        if torch.linalg.norm(torch.cross(orig, z_up, dim=0)) < 1e-5:
            z_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        extr = utils3d.torch.extrinsics_look_at(orig, target, z_up)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t)
        extrinsics.append(extr)
        intrinsics.append(intr)
    return extrinsics, intrinsics


def _render_backend_pref() -> str:
    """
    ``EVAL_MESH_RENDER_BACKEND``: ``auto`` | ``cuda`` | ``matplotlib`` | ``pyvista``.

    ``auto`` (default): **CUDA** (Trellis ``MeshRenderer``) if ``torch.cuda.is_available()``;
    else if ``DISPLAY`` is unset → **Matplotlib**; else try **PyVista**.
    """
    v = (os.environ.get("EVAL_MESH_RENDER_BACKEND") or "auto").strip().lower()
    if v in ("auto", "cuda", "matplotlib", "pyvista", "utils3d"):
        return v
    return "auto"


def _mesh_render_device() -> "torch.device":
    import torch

    d = (os.environ.get("EVAL_MESH_RENDER_DEVICE") or "cuda").strip()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.is_available() and d.startswith("cuda"):
        return torch.device(d)
    return torch.device("cpu")


def _trimesh_to_raster_mesh(mesh: "trimesh.Trimesh", device: "torch.device") -> _RasterMeshView:
    """Build raster input on ``device`` (same normalization as other backends)."""
    import torch
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    m = _normalize_mesh_center_scale(mesh)
    v = torch.as_tensor(np.asarray(m.vertices, dtype=np.float32), device=device)
    f = torch.as_tensor(np.asarray(m.faces, dtype=np.int64), device=device)
    return _RasterMeshView(v, f)


def _trimesh_to_colored_raster_mesh(mesh: "trimesh.Trimesh", device: "torch.device") -> _RasterMeshView:
    """Build Trellis raster input with vertex RGB attributes on ``device``."""
    import torch

    m = _normalize_mesh_center_scale(_mesh_with_baked_vertex_colors(mesh))
    sample = _trimesh_to_raster_mesh(m, device)
    colors = _mesh_vertex_rgb_u8(m).astype(np.float32) / 255.0
    sample.vertex_attrs = torch.as_tensor(colors, dtype=torch.float32, device=device)
    return sample


def _render_trimesh_multiview_cuda(
    mesh: "trimesh.Trimesh",
    *,
    device: "torch.device",
    nviews: int,
    resolution: int,
    camera_radius: float,
    fov_deg: float,
) -> List[np.ndarray]:
    """Rasterize mesh normals with Trellis ``MeshRenderer`` (nvdiffrast CUDA). No mesh decimation."""
    from trellis.renderers.mesh_renderer import MeshRenderer

    sample = _trimesh_to_raster_mesh(mesh, device)
    if not sample.success:
        return [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(nviews)]

    renderer = MeshRenderer(
        rendering_options={
            "resolution": int(resolution),
            "near": 1,
            "far": 100,
            "ssaa": 1,
        },
        device=str(device),
    )

    fitted_radius = _fit_camera_radius(_normalize_mesh_center_scale(mesh), camera_radius, fov_deg=fov_deg)
    views = _camera_views(nviews, fitted_radius)
    yaw_pitch = [_view_to_yaw_pitch(np.asarray(v["position"], dtype=np.float64)) for v in views]
    yaws = [c[0] for c in yaw_pitch]
    pitchs = [c[1] for c in yaw_pitch]
    extrinsics, intrinsics = _yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, fitted_radius, fov_deg, device
    )

    nv = int(sample.vertices.shape[0])
    nf = int(sample.faces.shape[0])
    print(
        f"[mesh_multiview] CUDA render (nvdiffrast): nviews={nviews} res={resolution} "
        f"verts={nv} faces={nf} device={device}",
        flush=True,
    )

    images: List[np.ndarray] = []
    for extr, intr in zip(extrinsics, intrinsics):
        res = renderer.render(sample, extr, intr, return_types=["normal"])
        n = res["normal"].detach().float().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(n * 255.0, 0, 255).astype(np.uint8)
        images.append(img)
    return images


def _render_colored_trimesh_multiview_cuda(
    mesh: "trimesh.Trimesh",
    *,
    device: "torch.device",
    nviews: int,
    resolution: int,
    camera_radius: float,
    fov_deg: float,
    background: str,
) -> List[np.ndarray]:
    """Rasterize baked vertex RGB colors with Trellis ``MeshRenderer`` (nvdiffrast CUDA)."""
    import torch
    from trellis.renderers.mesh_renderer import MeshRenderer

    sample = _trimesh_to_colored_raster_mesh(mesh, device)
    if not sample.success:
        return [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(nviews)]

    ssaa = int((os.environ.get("EVAL_MESH_RENDER_SSAA") or "1").strip() or "1")
    renderer = MeshRenderer(
        rendering_options={
            "resolution": int(resolution),
            "near": 1,
            "far": 100,
            "ssaa": max(1, ssaa),
        },
        device=str(device),
    )

    m = _normalize_mesh_center_scale(mesh)
    fitted_radius = _fit_camera_radius(m, camera_radius, fov_deg=fov_deg)
    views = _camera_views(nviews, fitted_radius)
    yaw_pitch = [_view_to_yaw_pitch(np.asarray(v["position"], dtype=np.float64)) for v in views]
    yaws = [c[0] for c in yaw_pitch]
    pitchs = [c[1] for c in yaw_pitch]
    extrinsics, intrinsics = _yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, fitted_radius, fov_deg, device
    )

    bg_lower = str(background).lower()
    bg_rgb = (1.0, 1.0, 1.0) if bg_lower in ("white", "w", "#fff", "#ffffff") else (0.0, 0.0, 0.0)
    bg_t = torch.tensor(bg_rgb, dtype=torch.float32, device=device).view(3, 1, 1)

    nv = int(sample.vertices.shape[0])
    nf = int(sample.faces.shape[0])
    print(
        f"[mesh_multiview] CUDA RGB render (nvdiffrast): nviews={nviews} res={resolution} "
        f"verts={nv} faces={nf} device={device}",
        flush=True,
    )

    images: List[np.ndarray] = []
    with torch.inference_mode():
        for extr, intr in zip(extrinsics, intrinsics):
            res = renderer.render(sample, extr, intr, return_types=["color", "mask"])
            color = res["color"].detach().float().clamp(0.0, 1.0)
            mask = res.get("mask")
            if torch.is_tensor(mask):
                mask = mask.detach().float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                color = color * mask + bg_t * (1.0 - mask)
            img = color.mul(255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            images.append(np.ascontiguousarray(img))
    return images


def _try_render_colored_multiview_cuda(
    mesh: "trimesh.Trimesh",
    *,
    nviews: int,
    resolution: int,
    camera_radius: float,
    background: str,
) -> Optional[List[np.ndarray]]:
    import torch

    if not torch.cuda.is_available():
        return None
    dev = _mesh_render_device()
    if dev.type != "cuda":
        return None
    try:
        torch.cuda.set_device(dev)
    except Exception:
        pass
    fov_deg = float((os.environ.get("EVAL_MESH_RENDER_FOV") or str(_DEFAULT_RENDER_FOV_DEG)).strip())
    try:
        return _render_colored_trimesh_multiview_cuda(
            mesh,
            device=dev,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
            fov_deg=fov_deg,
            background=background,
        )
    except Exception as exc:
        warnings.warn(
            f"CUDA RGB 多视角渲染失败（{type(exc).__name__}: {exc}），将尝试其他后端。",
            stacklevel=3,
        )
        return None


def _try_render_multiview_cuda(
    mesh: "trimesh.Trimesh",
    *,
    nviews: int,
    resolution: int,
    camera_radius: float,
) -> Optional[List[np.ndarray]]:
    import torch

    if not torch.cuda.is_available():
        return None
    dev = _mesh_render_device()
    if dev.type != "cuda":
        return None
    fov_deg = float((os.environ.get("EVAL_MESH_RENDER_FOV") or str(_DEFAULT_RENDER_FOV_DEG)).strip())
    try:
        return _render_trimesh_multiview_cuda(
            mesh,
            device=dev,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
            fov_deg=fov_deg,
        )
    except Exception as exc:
        warnings.warn(
            f"CUDA 多视角渲染失败（{type(exc).__name__}: {exc}），将尝试其他后端。",
            stacklevel=3,
        )
        return None


def _configure_pyvista_headless_env() -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")


def _render_trimesh_multiview_matplotlib(
    mesh: "trimesh.Trimesh",
    *,
    nviews: int,
    resolution: int,
    camera_radius: float,
    background: str,
) -> List[np.ndarray]:
    """Head-safe: Agg backend + ``plot_trisurf`` (no VTK / X11)."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from PIL import Image

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    m = _normalize_mesh_center_scale(mesh)
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return [
            np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(nviews)
        ]

    dpi = 100
    fig_in = resolution / float(dpi)
    bg = (0, 0, 0) if str(background).lower() in ("black", "k", "#000", "#000000") else (1, 1, 1)

    radius = _mesh_bounding_radius(m)
    lim = max(radius * _DEFAULT_CAMERA_MARGIN, 1e-3)
    views = _camera_views(nviews, _fit_camera_radius(m, camera_radius))

    images: List[np.ndarray] = []
    for view in views:
        azim, elev = _view_to_azim_elev(np.asarray(view["position"], dtype=np.float64))

        fig = plt.figure(figsize=(fig_in, fig_in), dpi=dpi, facecolor=bg)
        ax = fig.add_subplot(projection="3d")
        ax.set_facecolor(bg)
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            color="0.95" if bg[0] < 0.5 else "0.15",
            linewidth=0.08,
            antialiased=True,
            shade=True,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=dpi, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = np.asarray(Image.open(buf).convert("RGB"))
        if img.shape[0] != resolution or img.shape[1] != resolution:
            pil = Image.fromarray(img)
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except AttributeError:
                resample = Image.LANCZOS  # type: ignore[attr-defined]
            img = np.asarray(pil.resize((resolution, resolution), resample))
        images.append(img.astype(np.uint8))
    return images


def _render_trimesh_multiview_pyvista_impl(
    mesh: "trimesh.Trimesh",
    *,
    nviews: int,
    resolution: int,
    camera_radius: float,
    background: str,
) -> List[np.ndarray]:
    _configure_pyvista_headless_env()
    import pyvista as pv
    import trimesh

    try:
        pv.start_xvfb(wait=0.5)
    except Exception:
        pass

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    m = _normalize_mesh_center_scale(mesh)
    pv_mesh = pv.wrap(m)
    fov_deg = float((os.environ.get("EVAL_MESH_RENDER_FOV") or str(_DEFAULT_RENDER_FOV_DEG)).strip())
    fitted_radius = _fit_camera_radius(m, camera_radius, fov_deg=fov_deg)
    views = _camera_views(nviews, fitted_radius)

    images: List[np.ndarray] = []
    for view in views:
        plotter = pv.Plotter(off_screen=True, window_size=(resolution, resolution))
        plotter.set_background(background)
        plotter.add_mesh(pv_mesh, color="white", smooth_shading=True)
        plotter.camera.view_angle = fov_deg
        plotter.camera_position = [
            tuple(np.asarray(view["position"], dtype=np.float64).tolist()),
            (0.0, 0.0, 0.0),
            tuple(view["up"]),
        ]
        plotter.camera.clipping_range = (0.01, max(100.0, fitted_radius * 4.0))
        img = plotter.screenshot(return_img=True, transparent_background=False)
        plotter.close()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        images.append(img)
    return images


def _mesh_vertex_rgb_u8(mesh: "trimesh.Trimesh") -> np.ndarray:
    import trimesh

    n = int(len(mesh.vertices))
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    visual = getattr(mesh, "visual", None)
    colors: Optional[np.ndarray] = None
    if visual is not None:
        try:
            color_visual = visual.to_color() if hasattr(visual, "to_color") else visual
            vc = getattr(color_visual, "vertex_colors", None)
            if vc is not None:
                arr = np.asarray(vc)
                if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] >= 3:
                    colors = arr[:, :3]
        except Exception:
            colors = None
    if colors is None:
        try:
            vc = np.asarray(mesh.visual.vertex_colors)
            if vc.ndim == 2 and vc.shape[0] == n and vc.shape[1] >= 3:
                colors = vc[:, :3]
        except Exception:
            colors = None
    if colors is None:
        return np.full((n, 3), 230, dtype=np.uint8)
    return np.clip(colors, 0, 255).astype(np.uint8)


def _mesh_with_baked_vertex_colors(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    import trimesh

    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=_mesh_vertex_rgb_u8(mesh))
    return m


def _as_colored_trimesh(scene_or_mesh: Any) -> "trimesh.Trimesh":
    import trimesh

    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return _mesh_with_baked_vertex_colors(scene_or_mesh)
    if isinstance(scene_or_mesh, trimesh.Scene):
        try:
            meshes = scene_or_mesh.dump(concatenate=False)
        except TypeError:
            meshes = scene_or_mesh.dump()
        meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh)]
    else:
        meshes = [m for m in getattr(scene_or_mesh, "geometry", {}).values() if isinstance(m, trimesh.Trimesh)]

    if not meshes:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    vertices: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        m = _mesh_with_baked_vertex_colors(mesh)
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.faces, dtype=np.int64)
        c = _mesh_vertex_rgb_u8(m)
        if v.size == 0 or f.size == 0:
            continue
        vertices.append(v)
        faces.append(f + offset)
        colors.append(c)
        offset += v.shape[0]

    if not vertices:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    out = trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        process=False,
    )
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=np.vstack(colors))
    return out


def load_colored_trimesh_any(path: str) -> "trimesh.Trimesh":
    """Load a mesh/scene and bake texture/material colors to per-vertex RGB."""
    import trimesh

    scene_or_mesh = trimesh.load(path, force="scene")
    return _as_colored_trimesh(scene_or_mesh)


def _render_colored_trimesh_multiview_pyvista_impl(
    mesh: "trimesh.Trimesh",
    *,
    nviews: int,
    resolution: int,
    camera_radius: float,
    background: str,
) -> List[np.ndarray]:
    _configure_pyvista_headless_env()
    import pyvista as pv
    import trimesh

    try:
        pv.start_xvfb(wait=0.5)
    except Exception:
        pass

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    m = _normalize_mesh_center_scale(_mesh_with_baked_vertex_colors(mesh))
    faces = np.asarray(m.faces, dtype=np.int64)
    if len(m.vertices) == 0 or faces.size == 0:
        return [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(nviews)]
    pv_faces = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    pv_mesh = pv.PolyData(np.asarray(m.vertices, dtype=np.float64), pv_faces)
    pv_mesh.point_data["vertex_rgb"] = _mesh_vertex_rgb_u8(m)
    fov_deg = float((os.environ.get("EVAL_MESH_RENDER_FOV") or str(_DEFAULT_RENDER_FOV_DEG)).strip())
    fitted_radius = _fit_camera_radius(m, camera_radius, fov_deg=fov_deg)
    views = _camera_views(nviews, fitted_radius)

    images: List[np.ndarray] = []
    for view in views:
        plotter = pv.Plotter(off_screen=True, window_size=(resolution, resolution))
        plotter.set_background(background)
        plotter.add_mesh(
            pv_mesh,
            scalars="vertex_rgb",
            rgb=True,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        plotter.camera.view_angle = fov_deg
        plotter.camera_position = [
            tuple(np.asarray(view["position"], dtype=np.float64).tolist()),
            (0.0, 0.0, 0.0),
            tuple(view["up"]),
        ]
        plotter.camera.clipping_range = (0.01, max(100.0, fitted_radius * 4.0))
        img = plotter.screenshot(return_img=True, transparent_background=False)
        plotter.close()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        images.append(img[:, :, :3])
    return images


def _render_colored_trimesh_multiview_matplotlib(
    mesh: "trimesh.Trimesh",
    *,
    nviews: int,
    resolution: int,
    camera_radius: float,
    background: str,
) -> List[np.ndarray]:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from PIL import Image

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    m = _normalize_mesh_center_scale(_mesh_with_baked_vertex_colors(mesh))
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(nviews)]

    vertex_rgb = _mesh_vertex_rgb_u8(m).astype(np.float32) / 255.0
    face_rgb = np.clip(vertex_rgb[faces].mean(axis=1), 0.0, 1.0)
    dpi = 100
    fig_in = resolution / float(dpi)
    bg = (0, 0, 0) if str(background).lower() in ("black", "k", "#000", "#000000") else (1, 1, 1)
    radius = _mesh_bounding_radius(m)
    lim = max(radius * _DEFAULT_CAMERA_MARGIN, 1e-3)
    views = _camera_views(nviews, _fit_camera_radius(m, camera_radius))

    images: List[np.ndarray] = []
    for view in views:
        azim, elev = _view_to_azim_elev(np.asarray(view["position"], dtype=np.float64))

        fig = plt.figure(figsize=(fig_in, fig_in), dpi=dpi, facecolor=bg)
        ax = fig.add_subplot(projection="3d")
        ax.set_facecolor(bg)
        surf = ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            linewidth=0.08,
            antialiased=True,
            shade=False,
        )
        surf.set_facecolors(face_rgb)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=dpi, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = np.asarray(Image.open(buf).convert("RGB"))
        if img.shape[0] != resolution or img.shape[1] != resolution:
            pil = Image.fromarray(img)
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except AttributeError:
                resample = Image.LANCZOS  # type: ignore[attr-defined]
            img = np.asarray(pil.resize((resolution, resolution), resample))
        images.append(img.astype(np.uint8))
    return images

# Trellis-style Hammersley on sphere (yaw=phi, pitch=theta), see ShapeLLM-Omni trellis/utils/random_utils.py
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def _radical_inverse(base: int, n: int) -> float:
    val = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def _halton_sequence(dim: int, n: int) -> float:
    return _radical_inverse(_PRIMES[dim], n)


def _hammersley_sequence(dim: int, n: int, num_samples: int) -> List[float]:
    return [n / num_samples] + [_halton_sequence(dim - 1, n)]


def sphere_hammersley_sequence(n: int, num_samples: int, offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    u, v = _hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    theta = float(np.arccos(1 - 2 * u) - np.pi / 2)
    phi = float(v * 2 * np.pi)
    return phi, theta


def fixed_metric_view_names(nviews: int) -> List[str]:
    """Stable names for the first four metric render views."""
    if int(nviews) == len(_FIXED_VIEW_DIRECTIONS):
        return [name for name, _, _ in _FIXED_VIEW_DIRECTIONS]
    return [f"view_{i:03d}" for i in range(int(nviews))]


def _unit_vector(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError(f"Zero camera direction: {vec!r}")
    return arr / norm


def _camera_views(nviews: int, camera_radius: float) -> List[Dict[str, Any]]:
    """Return camera position/up specs; nviews=4 is front/right/top/axis."""
    if int(nviews) == len(_FIXED_VIEW_DIRECTIONS):
        return [
            {
                "name": name,
                "position": _unit_vector(direction) * float(camera_radius),
                "up": tuple(_unit_vector(up).tolist()),
            }
            for name, direction, up in _FIXED_VIEW_DIRECTIONS
        ]

    views: List[Dict[str, Any]] = []
    for i in range(int(nviews)):
        yaw, pitch = sphere_hammersley_sequence(i, int(nviews))
        position = np.array(
            [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)],
            dtype=np.float64,
        ) * float(camera_radius)
        views.append({"name": f"view_{i:03d}", "position": position, "up": (0.0, 0.0, 1.0)})
    return views


def _mesh_bounding_radius(mesh: "trimesh.Trimesh") -> float:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return 1.0
    return max(float(np.linalg.norm(verts, axis=1).max()), 1e-6)


def _fit_camera_radius(
    mesh: "trimesh.Trimesh",
    requested_radius: float,
    *,
    fov_deg: float = _DEFAULT_RENDER_FOV_DEG,
    margin: float = _DEFAULT_CAMERA_MARGIN,
) -> float:
    radius = _mesh_bounding_radius(mesh)
    half_fov = np.deg2rad(max(1.0, min(float(fov_deg), 170.0)) * 0.5)
    fitted = radius * float(margin) / max(float(np.sin(half_fov)), 1e-6)
    return max(float(requested_radius), float(fitted))


def _view_to_azim_elev(position: np.ndarray) -> Tuple[float, float]:
    u = position / (np.linalg.norm(position) + 1e-12)
    azim = float(np.degrees(np.arctan2(u[1], u[0])))
    elev = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
    return azim, elev


def _view_to_yaw_pitch(position: np.ndarray) -> Tuple[float, float]:
    u = position / (np.linalg.norm(position) + 1e-12)
    yaw = float(np.arctan2(u[0], u[1]))
    pitch = float(np.arcsin(np.clip(u[2], -1.0, 1.0)))
    return yaw, pitch


def _normalize_mesh_center_scale(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    import trimesh

    m = mesh.copy()
    if len(m.vertices) == 0:
        return m
    verts = np.asarray(m.vertices, dtype=np.float64)
    finite_mask = np.isfinite(verts).all(axis=1)
    if not finite_mask.all():
        m.update_vertices(finite_mask)
        verts = np.asarray(m.vertices, dtype=np.float64)
    if verts.size == 0:
        return m
    bounds_min = verts.min(axis=0)
    bounds_max = verts.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    m.vertices = verts - center
    radius = _mesh_bounding_radius(m)
    if radius > 1e-8:
        m.vertices = np.asarray(m.vertices, dtype=np.float64) / radius
    return m


def render_trimesh_multiview_pyvista(
    mesh: "trimesh.Trimesh",
    *,
    nviews: int = 30,
    resolution: int = 512,
    camera_radius: float = 2.0,
    background: str = "black",
) -> List[np.ndarray]:
    """
    Return list of uint8 (H, W, 3) RGB images, same camera family as Trellis ``render_multiview`` (approx.).

    **Backends** (``EVAL_MESH_RENDER_BACKEND``):

    - ``auto`` (default): **CUDA** (Trellis ``MeshRenderer`` + nvdiffrast) if ``torch.cuda.is_available()``;
      else without ``DISPLAY`` use **Matplotlib**; else **PyVista** (with Matplotlib fallback on error).
    - ``cuda`` / ``utils3d``: force CUDA path (falls back with warning if unavailable).
    - ``matplotlib`` / ``pyvista``: CPU / VTK as before.

    Optional: ``EVAL_MESH_RENDER_DEVICE`` (default ``cuda``), ``EVAL_MESH_RENDER_FOV`` (default ``40``).
    """
    pref = _render_backend_pref()

    if pref in ("cuda", "utils3d"):
        out = _try_render_multiview_cuda(
            mesh,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
        )
        if out is not None:
            return out
        warnings.warn(
            "EVAL_MESH_RENDER_BACKEND 指定了 CUDA 但不可用或初始化失败，改用 Matplotlib/PyVista。",
            stacklevel=2,
        )

    if pref == "auto":
        out = _try_render_multiview_cuda(
            mesh,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
        )
        if out is not None:
            return out

    display_set = bool((os.environ.get("DISPLAY") or "").strip())
    # Headless: prefer Matplotlib after failed CUDA, including when user forced ``cuda`` but GPU path broke.
    use_mpl = pref == "matplotlib" or (not display_set and pref != "pyvista")
    if use_mpl:
        warnings.warn(
            "多视角渲染使用 Matplotlib（无 DISPLAY 或 EVAL_MESH_RENDER_BACKEND=matplotlib）。"
            "若有 GPU，可设置 EVAL_MESH_RENDER_BACKEND=auto（默认）以优先使用 CUDA（nvdiffrast）；"
            "或设置 DISPLAY 并 export EVAL_MESH_RENDER_BACKEND=pyvista。",
            stacklevel=2,
        )
        return _render_trimesh_multiview_matplotlib(
            mesh,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
            background=background,
        )


def render_colored_trimesh_multiview(
    mesh: "trimesh.Trimesh",
    *,
    nviews: int = 30,
    resolution: int = 512,
    camera_radius: float = 2.0,
    background: str = "black",
) -> List[np.ndarray]:
    """
    Render RGB views from baked vertex colors/textures.

    This path uses a CUDA RGB backend when available; unlike
    ``render_trimesh_multiview_pyvista``'s normal render path, it interpolates baked
    vertex colors and is suitable for color-sensitive CLIP/FID/KID evaluation.
    """
    pref = _render_backend_pref()
    if pref in ("cuda", "utils3d"):
        out = _try_render_colored_multiview_cuda(
            mesh,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
            background=background,
        )
        if out is not None:
            return out
        warnings.warn(
            "EVAL_MESH_RENDER_BACKEND 指定了 CUDA RGB 但不可用或初始化失败，改用 PyVista/Matplotlib。",
            stacklevel=2,
        )

    if pref == "auto":
        out = _try_render_colored_multiview_cuda(
            mesh,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
            background=background,
        )
        if out is not None:
            return out

    use_mpl = pref == "matplotlib"
    if not use_mpl:
        try:
            return _render_colored_trimesh_multiview_pyvista_impl(
                mesh,
                nviews=nviews,
                resolution=resolution,
                camera_radius=camera_radius,
                background=background,
            )
        except Exception as exc:
            warnings.warn(
                f"彩色 PyVista 渲染失败 ({exc!r})，改用 Matplotlib 彩色渲染。",
                stacklevel=2,
            )

    return _render_colored_trimesh_multiview_matplotlib(
        mesh,
        nviews=nviews,
        resolution=resolution,
        camera_radius=camera_radius,
        background=background,
    )
    if pref == "pyvista":
        return _render_trimesh_multiview_pyvista_impl(
            mesh,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
            background=background,
        )
    # auto + DISPLAY set: try PyVista, fall back on error
    try:
        return _render_trimesh_multiview_pyvista_impl(
            mesh,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
            background=background,
        )
    except Exception as exc:
        warnings.warn(
            f"PyVista 渲染失败 ({exc!r})，改用 Matplotlib。",
            stacklevel=2,
        )
        return _render_trimesh_multiview_matplotlib(
            mesh,
            nviews=nviews,
            resolution=resolution,
            camera_radius=camera_radius,
            background=background,
        )


def load_trimesh_any(path: str) -> "trimesh.Trimesh":
    """Load OBJ/GLB/PLY etc. as a single ``Trimesh`` (concatenate scene geometries)."""
    import trimesh

    scene_or_mesh = trimesh.load(path, force="scene")
    if isinstance(scene_or_mesh, trimesh.Scene):
        if not scene_or_mesh.geometry:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
        return trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    return trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))


def caption_for_clip_from_record_prompt(prompt: str) -> str:
    """Use first line of stored prompt as caption (兼容历史 ``caption\\nreconstruct`` 记录)."""
    if not prompt:
        return ""
    return str(prompt).strip().split("\n", 1)[0].strip()
