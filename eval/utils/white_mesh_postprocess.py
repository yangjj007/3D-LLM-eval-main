"""VQ-VAE white mesh cleanup: voxel remesh, component bridging, morphology, fill holes, smooth/decimate."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "on"}


def _subcfg(cfg: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
    if not cfg or not isinstance(cfg, dict):
        return {}
    v = cfg.get(key)
    return v if isinstance(v, dict) else {}


def _marching_cubes_volume(vol_f: np.ndarray, *, iso: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(vol_f, level=float(iso), method="lewiner")
    return verts, faces


def _norm_coords_from_mc(
    verts: np.ndarray,
    grid_shape: Tuple[int, int, int],
    *,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> np.ndarray:
    """Map marching-cubes vertex indices to world coords in [min_bound, max_bound]."""
    gx, gy, gz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    span = max_bound - min_bound
    # skimage uses node positions on a regular grid spanning array indices [0, nx-1] etc.
    tx = verts[:, 0] / max(gx - 1, 1)
    ty = verts[:, 1] / max(gy - 1, 1)
    tz = verts[:, 2] / max(gz - 1, 1)
    out = np.stack(
        [
            min_bound[0] + tx * span[0],
            min_bound[1] + ty * span[1],
            min_bound[2] + tz * span[2],
        ],
        axis=1,
    )
    return out.astype(np.float32)


def _mesh_to_volume_open3d(
    mesh: trimesh.Trimesh,
    resolution: int,
    *,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> np.ndarray:
    import open3d as o3d

    r = int(resolution)
    if r < 8:
        raise ValueError(f"voxel resolution must be >= 8, got {r}")
    span = max_bound - min_bound
    voxel_size = float(np.max(span) / r)
    tm = mesh.copy()
    tm.remove_unreferenced_vertices()
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
    )
    o3d_mesh.compute_vertex_normals()
    vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=voxel_size,
        min_bound=min_bound.tolist(),
        max_bound=max_bound.tolist(),
    )
    vol = np.zeros((r, r, r), dtype=np.float32)
    for vk in vox.get_voxels():
        i, j, k = int(vk.grid_index[0]), int(vk.grid_index[1]), int(vk.grid_index[2])
        if 0 <= i < r and 0 <= j < r and 0 <= k < r:
            vol[i, j, k] = 1.0
    return vol


def _morphology_gpu(
    vol: np.ndarray,
    *,
    closing_iters: int,
    opening_iters: int,
    kernel_size: int,
    device: Any,
) -> np.ndarray:
    import torch

    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    t = torch.from_numpy(vol.astype(np.float32)).to(device=device, dtype=torch.float32)
    t = t.unsqueeze(0).unsqueeze(0)

    def dilate(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool3d(x, kernel_size=k, stride=1, padding=pad)

    def erode(x: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.nn.functional.max_pool3d(1.0 - x, kernel_size=k, stride=1, padding=pad)

    for _ in range(int(closing_iters)):
        t = dilate(t)
    for _ in range(int(closing_iters)):
        t = erode(t)
    for _ in range(int(opening_iters)):
        t = erode(t)
    for _ in range(int(opening_iters)):
        t = dilate(t)
    out = t.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return out


def _morphology_scipy(
    vol: np.ndarray,
    *,
    closing_iters: int,
    opening_iters: int,
    kernel_size: int,
) -> np.ndarray:
    from scipy import ndimage

    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    struct = np.ones((k, k, k), dtype=bool)
    v = vol.astype(bool)
    for _ in range(int(closing_iters)):
        v = ndimage.binary_closing(v, structure=struct)
    for _ in range(int(opening_iters)):
        v = ndimage.binary_opening(v, structure=struct)
    return v.astype(np.float32)


def _paint_voxel_line(mask: np.ndarray, start: np.ndarray, end: np.ndarray, *, radius: int) -> None:
    steps = int(np.max(np.abs(end - start))) + 1
    if steps <= 1:
        coords = np.round(start[None, :]).astype(np.int64)
    else:
        coords = np.round(np.linspace(start, end, steps)).astype(np.int64)

    r = max(0, int(radius))
    sx, sy, sz = mask.shape
    for x, y, z in coords:
        x0, x1 = max(0, x - r), min(sx, x + r + 1)
        y0, y1 = max(0, y - r), min(sy, y + r + 1)
        z0, z1 = max(0, z - r), min(sz, z + r + 1)
        mask[x0:x1, y0:y1, z0:z1] = True


def _connect_large_volume_components(
    vol: np.ndarray,
    *,
    min_component_ratio: float,
    max_components: int,
    bridge_radius: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from scipy import ndimage

    vb = vol > 0.5
    labels, num_components = ndimage.label(vb, structure=np.ones((3, 3, 3), dtype=bool))
    if num_components <= 1:
        return vol, {"enabled": True, "num_components": int(num_components), "connected_components": 0}

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    min_size = max(1, int(sizes[largest] * max(0.0, float(min_component_ratio))))
    component_ids = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_size]
    component_ids = component_ids[: max(1, int(max_components))]
    if largest not in component_ids:
        component_ids.insert(0, largest)

    centers = ndimage.center_of_mass(vb, labels, component_ids)
    center_by_id = {
        cid: np.asarray(center, dtype=np.float64)
        for cid, center in zip(component_ids, centers)
        if np.all(np.isfinite(center))
    }
    base = center_by_id.get(largest)
    if base is None:
        return vol, {"enabled": True, "num_components": int(num_components), "connected_components": 0}

    bridged = vb.copy()
    connected = 0
    for cid in component_ids:
        if cid == largest:
            continue
        center = center_by_id.get(cid)
        if center is None:
            continue
        _paint_voxel_line(bridged, base, center, radius=bridge_radius)
        connected += 1

    out = bridged.astype(np.float32)
    rec = {
        "enabled": True,
        "num_components": int(num_components),
        "large_components": int(len(component_ids)),
        "connected_components": int(connected),
        "min_component_ratio": float(min_component_ratio),
        "bridge_radius": int(bridge_radius),
        "occupied_voxels": int(np.sum(out > 0.5)),
    }
    return out, rec


def _keep_largest_component(
    mesh: trimesh.Trimesh,
    cfg: Dict[str, Any],
) -> Tuple[trimesh.Trimesh, Dict[str, Any], Optional[str]]:
    try:
        parts = mesh.split(only_watertight=False)
    except Exception as exc:  # noqa: BLE001
        return mesh, {"enabled": True, "error": repr(exc)}, f"keep_largest_component.split_failed: {exc!r}"

    if len(parts) <= 1:
        return mesh, {"enabled": True, "num_components": 1}, None

    key = str(cfg.get("criterion", "faces")).lower()
    if key == "vertices":

        def _score(m: trimesh.Trimesh) -> int:
            return int(len(m.vertices))
    else:

        def _score(m: trimesh.Trimesh) -> int:
            return int(len(m.faces))

    out = max(parts, key=_score)
    rec = {
        "enabled": True,
        "num_components": len(parts),
        "criterion": key,
        "vertices_out": int(len(out.vertices)),
        "faces_out": int(len(out.faces)),
    }
    return out, rec, None


def postprocess_trimesh_white_mesh(
    mesh: trimesh.Trimesh,
    cfg: Optional[Dict[str, Any]],
    *,
    device: Optional[Any] = None,
    log_prefix: str = "",
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Clean decoder white mesh: voxel solid + component bridging + morphology + fill holes + remesh,
    optional Taubin smooth, quadric decimate, pymeshfix, then optional largest component.

    ``cfg`` uses nested dicts with per-step ``enabled`` flags. See task YAML ``model.white_mesh_postprocess``.
    """
    import torch

    debug: Dict[str, Any] = {"steps": {}, "warnings": []}
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        debug["steps"]["skipped"] = {"reason": "empty_mesh"}
        return mesh, debug

    if not cfg or not _truthy(cfg.get("enabled")):
        debug["steps"]["skipped"] = {"reason": "disabled"}
        return mesh, debug

    log_timings = _truthy(cfg.get("log_timings", True))

    def _log(msg: str) -> None:
        p = (log_prefix + " ").strip()
        print(f"{p}[white_mesh_post] {msg}", flush=True)

    min_b = np.array(cfg.get("bounds_min", [-1.0, -1.0, -1.0]), dtype=np.float64)
    max_b = np.array(cfg.get("bounds_max", [1.0, 1.0, 1.0]), dtype=np.float64)

    out = mesh.copy()
    t_all = time.time()

    # --- voxel remesh pipeline ---
    # Run this before component pruning so morphology can bridge nearby separated solids.
    vr = _subcfg(cfg, "voxel_remesh")
    mc_iso = float(cfg.get("marching_cubes_iso", 0.5))
    if _truthy(vr.get("enabled", True)):
        cc = _subcfg(cfg, "connect_components")
        morph = _subcfg(cfg, "morphology")
        fh = _subcfg(cfg, "fill_holes")
        res = int(vr.get("resolution", 512))
        use_gpu = _truthy(vr.get("use_gpu_morphology", True))
        dev = device
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.time()
        vol = _mesh_to_volume_open3d(out, res, min_bound=min_b, max_bound=max_b)
        t_vox = time.time() - t0
        occupied_before = int(np.sum(vol > 0.5))
        rec_vox = {"resolution": res, "occupied_voxels": occupied_before, "voxelize_sec": round(t_vox, 4)}
        debug["steps"]["voxelize"] = rec_vox
        if log_timings:
            _log(f"voxelize R={res} occupied={occupied_before} ({t_vox:.3f}s)")

        if _truthy(cc.get("enabled", False)):
            t1 = time.time()
            vol, rec_cc = _connect_large_volume_components(
                vol,
                min_component_ratio=float(cc.get("min_component_ratio", 0.05)),
                max_components=int(cc.get("max_components", 8)),
                bridge_radius=int(cc.get("bridge_radius", 1)),
            )
            rec_cc["time_sec"] = round(time.time() - t1, 4)
            debug["steps"]["connect_components"] = rec_cc
            if log_timings:
                _log(
                    "connect_components "
                    f"large={rec_cc.get('large_components', 0)} "
                    f"bridged={rec_cc.get('connected_components', 0)} "
                    f"occupied={rec_cc.get('occupied_voxels', int(np.sum(vol > 0.5)))} "
                    f"({rec_cc['time_sec']:.3f}s)"
                )
        else:
            debug["steps"]["connect_components"] = {"enabled": False}

        if _truthy(morph.get("enabled", True)):
            ci = int(morph.get("closing_iters", 1))
            oi = int(morph.get("opening_iters", 0))
            ks = int(morph.get("kernel_size", 3))
            t1 = time.time()
            if use_gpu and str(dev).startswith("cuda") and torch.cuda.is_available():
                vol = _morphology_gpu(
                    vol, closing_iters=ci, opening_iters=oi, kernel_size=ks, device=dev
                )
                morph_backend = "torch_cuda"
            else:
                vol = _morphology_scipy(vol, closing_iters=ci, opening_iters=oi, kernel_size=ks)
                morph_backend = "scipy_cpu"
            dt = time.time() - t1
            occ_m = int(np.sum(vol > 0.5))
            debug["steps"]["morphology"] = {
                "enabled": True,
                "closing_iters": ci,
                "opening_iters": oi,
                "kernel_size": ks,
                "backend": morph_backend,
                "occupied_voxels": occ_m,
                "time_sec": round(dt, 4),
            }
            if log_timings:
                _log(f"morphology backend={morph_backend} occupied={occ_m} ({dt:.3f}s)")
        else:
            debug["steps"]["morphology"] = {"enabled": False}

        if _truthy(fh.get("enabled", True)):
            from scipy import ndimage

            t1 = time.time()
            vb = (vol > 0.5).astype(bool)
            filled = ndimage.binary_fill_holes(vb)
            vol = filled.astype(np.float32)
            dt = time.time() - t1
            occ_f = int(np.sum(vol > 0.5))
            debug["steps"]["fill_holes"] = {
                "enabled": True,
                "occupied_voxels": occ_f,
                "time_sec": round(dt, 4),
            }
            if log_timings:
                _log(f"fill_holes occupied={occ_f} ({dt:.3f}s)")
        else:
            debug["steps"]["fill_holes"] = {"enabled": False}

        if _truthy(_subcfg(cfg, "marching_cubes").get("enabled", True)):
            t1 = time.time()
            try:
                verts, faces = _marching_cubes_volume(vol, iso=mc_iso)
            except Exception as exc:  # noqa: BLE001
                debug["warnings"].append(f"marching_cubes_failed: {exc!r}")
                debug["steps"]["marching_cubes"] = {"enabled": True, "error": repr(exc)}
                if log_timings:
                    _log(f"marching_cubes failed: {exc!r}")
            else:
                verts_w = _norm_coords_from_mc(verts, vol.shape, min_bound=min_b, max_bound=max_b)
                out = trimesh.Trimesh(vertices=verts_w, faces=faces.astype(np.int64), process=False)
                dt = time.time() - t1
                debug["steps"]["marching_cubes"] = {
                    "enabled": True,
                    "iso": mc_iso,
                    "vertices": int(len(out.vertices)),
                    "faces": int(len(out.faces)),
                    "time_sec": round(dt, 4),
                }
                if log_timings:
                    _log(f"marching_cubes V={len(out.vertices)} F={len(out.faces)} ({dt:.3f}s)")
        else:
            debug["steps"]["marching_cubes"] = {"enabled": False}
            debug["warnings"].append("marching_cubes disabled but voxel_remesh enabled; mesh unchanged after voxel ops")
    else:
        cc = _subcfg(cfg, "connect_components")
        morph = _subcfg(cfg, "morphology")
        fh = _subcfg(cfg, "fill_holes")
        if (
            _truthy(cc.get("enabled", False))
            or _truthy(morph.get("enabled", False))
            or _truthy(fh.get("enabled", False))
        ):
            debug["warnings"].append("connect_components/morphology/fill_holes need voxel_remesh.enabled=true; skipped")

    # --- Taubin ---
    tb = _subcfg(cfg, "taubin")
    if _truthy(tb.get("enabled", False)) and len(out.vertices) > 0 and len(out.faces) > 0:
        t0 = time.time()
        it = int(tb.get("iterations", 5))
        lamb = float(tb.get("lamb", 0.5))
        nu = float(tb.get("nu", -0.53))
        try:
            trimesh.smoothing.filter_taubin(out, lamb=lamb, nu=nu, iterations=it)
            dt = time.time() - t0
            debug["steps"]["taubin"] = {"enabled": True, "iterations": it, "lamb": lamb, "nu": nu, "time_sec": round(dt, 4)}
            if log_timings:
                _log(f"taubin it={it} ({dt:.3f}s)")
        except Exception as exc:  # noqa: BLE001
            debug["warnings"].append(f"taubin_failed: {exc!r}")
            debug["steps"]["taubin"] = {"enabled": True, "error": repr(exc)}

    # --- decimation ---
    dec = _subcfg(cfg, "decimate")
    if _truthy(dec.get("enabled", False)) and len(out.vertices) > 0 and len(out.faces) > 0:
        import open3d as o3d

        t0 = time.time()
        target_tris = dec.get("target_triangles")
        ratio = dec.get("ratio")
        n_face = int(len(out.faces))
        if target_tris is not None:
            target_count = max(4, min(int(target_tris), n_face))
        elif ratio is not None:
            target_count = max(4, int(float(ratio) * n_face))
        else:
            target_count = max(4, n_face // 2)
        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(out.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(out.faces, dtype=np.int32)),
        )
        try:
            simp = om.simplify_quadric_decimation(target_number_of_triangles=target_count)
            v = np.asarray(simp.vertices, dtype=np.float32)
            f = np.asarray(simp.triangles, dtype=np.int64)
            out = trimesh.Trimesh(vertices=v, faces=f, process=False)
            dt = time.time() - t0
            debug["steps"]["decimate"] = {
                "enabled": True,
                "target_triangles": target_count,
                "faces_out": int(len(out.faces)),
                "time_sec": round(dt, 4),
            }
            if log_timings:
                _log(f"decimate -> F={len(out.faces)} ({dt:.3f}s)")
        except Exception as exc:  # noqa: BLE001
            debug["warnings"].append(f"decimate_failed: {exc!r}")
            debug["steps"]["decimate"] = {"enabled": True, "error": repr(exc)}

    # --- pymeshfix ---
    pfx = _subcfg(cfg, "pymeshfix")
    if _truthy(pfx.get("enabled", False)) and len(out.vertices) > 0 and len(out.faces) > 0:
        t0 = time.time()
        try:
            from trellis.utils.mesh_utils import make_watertight

            verbose = _truthy(pfx.get("verbose", False))
            out = make_watertight(out, verbose=verbose)
            dt = time.time() - t0
            debug["steps"]["pymeshfix"] = {
                "enabled": True,
                "vertices_out": int(len(out.vertices)),
                "faces_out": int(len(out.faces)),
                "time_sec": round(dt, 4),
            }
            if log_timings:
                _log(f"pymeshfix V={len(out.vertices)} F={len(out.faces)} ({dt:.3f}s)")
        except Exception as exc:  # noqa: BLE001
            debug["warnings"].append(f"pymeshfix_failed: {exc!r}")
            debug["steps"]["pymeshfix"] = {"enabled": True, "error": repr(exc)}

    # --- largest connected component ---
    lc = _subcfg(cfg, "keep_largest_component")
    if _truthy(lc.get("enabled", True)) and len(out.vertices) > 0 and len(out.faces) > 0:
        t0 = time.time()
        out, rec, warning = _keep_largest_component(out, lc)
        rec["time_sec"] = round(time.time() - t0, 4)
        debug["steps"]["keep_largest_component"] = rec
        if warning:
            debug["warnings"].append(warning)
        if log_timings:
            if rec.get("num_components", 1) == 1:
                _log(f"keep_largest_component single component ({rec['time_sec']:.3f}s)")
            else:
                _log(f"keep_largest_component {rec}")

    debug["total_time_sec"] = round(time.time() - t_all, 4)
    debug["vertices_final"] = int(len(out.vertices))
    debug["faces_final"] = int(len(out.faces))
    if log_timings:
        _log(f"done total={debug['total_time_sec']}s V={debug['vertices_final']} F={debug['faces_final']}")

    return out, debug


def apply_postprocess_from_cfg(
    mesh: Optional[trimesh.Trimesh],
    postprocess_cfg: Optional[Dict[str, Any]],
    *,
    device: Optional[Any] = None,
    log_prefix: str = "",
) -> Tuple[Optional[trimesh.Trimesh], Dict[str, Any]]:
    if mesh is None:
        return None, {"steps": {"skipped": {"reason": "no_mesh"}}, "warnings": []}
    m, dbg = postprocess_trimesh_white_mesh(mesh, postprocess_cfg, device=device, log_prefix=log_prefix)
    return m, dbg
