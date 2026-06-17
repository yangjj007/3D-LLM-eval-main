"""
Trellis coloring: white mesh -> ``run_variant`` (voxelize + SLAT) -> textured GLB.

Mirrors ShapeLLM-Omni ``app.py`` / ``TrellisTextTo3DPipeline.run_variant``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def trellis_mesh_to_textured_glb(
    pipeline: Any,
    mesh_trimesh: "trimesh.Trimesh",
    prompt: str,
    *,
    num_samples: int = 1,
    seed: int = 42,
    simplify: float = 0.95,
    texture_size: int = 1024,
    input_orientation: str = "yup",
    slat_sampler_params: Optional[Dict[str, Any]] = None,
) -> "trimesh.Trimesh":
    """
    Run Trellis SLAT conditioned on ``prompt`` with sparse structure from ``mesh_trimesh``.

    ``slat_sampler_params`` is merged into the pipeline defaults and forwarded to the flow
    sampler (e.g. ``steps`` for Euler integration depth).

    Returns a ``trimesh.Trimesh`` with UV + PBR material suitable for ``.export('x.glb')``.
    """
    import open3d as o3d
    import sys
    import torch
    import trimesh

    from eval.colorization.sparse_quant_compat import apply_sparse_is_quantized_compat
    from trellis.utils import postprocessing_utils

    apply_sparse_is_quantized_compat()

    if not isinstance(mesh_trimesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh_trimesh)}")

    tm = mesh_trimesh.copy()
    orientation = (
        str(input_orientation or "yup").strip().lower().replace("-", "").replace("_", "")
    )
    if orientation not in {"yup", "zup"}:
        raise ValueError(
            "Unsupported Trellis colorization input_orientation="
            f"{input_orientation!r}; expected 'yup' or 'zup'."
        )
    verts = tm.vertices.astype(np.float64)
    if orientation == "yup":
        # Eval white meshes inherit y-up GLB orientation. Trellis expects z-up here,
        # and postprocessing_utils.to_glb rotates z-up back to y-up before export.
        yup_to_zup = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        verts = verts @ yup_to_zup
        tm.vertices = verts
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces.astype(np.int32))
    o3d_mesh.compute_vertex_normals()

    # Trellis 官方 text pipeline 权重与 ``conv_spconv`` 路径一致，要求 **spconv** 稀疏后端。
    # 评测侧 Sparse VAE 往往把 ``SPARSE_BACKEND`` 留在 ``torchsparse``；若不在此强制切换，
    # ``sp.SparseTensor`` 会包装 torchsparse 数据，随后 ``SubMConv3d`` 把该对象当作
    # ``spconv.SparseConvTensor`` 使用，触发 ``'SparseTensor' object has no attribute 'features'``。
    import trellis.modules.sparse as _tsp

    prev_backend = _tsp.BACKEND
    _tsp.set_sparse_backend("spconv")
    try:
        with torch.no_grad():
            # Inline ``run_variant`` so we can align the sampled SLAT dtype with the
            # decoders before ``decode_slat``. Some Trellis checkpoints load decoders
            # in fp16 while the sampler returns fp32 sparse features.
            cond = pipeline.get_cond([str(prompt)])
            coords = pipeline.voxelize(o3d_mesh)
            coords = torch.cat(
                [
                    torch.arange(int(num_samples), device=coords.device)
                    .repeat_interleave(coords.shape[0], 0)[:, None]
                    .int(),
                    coords.repeat(int(num_samples), 1),
                ],
                1,
            )
            torch.manual_seed(int(seed))
            slat_kw: Dict[str, Any] = dict(slat_sampler_params or {})
            slat = pipeline.sample_slat(cond, coords, slat_kw)

            decoder = (
                pipeline.models["slat_decoder_mesh"]
                if "slat_decoder_mesh" in pipeline.models
                else pipeline.models["slat_decoder_gs"]
                if "slat_decoder_gs" in pipeline.models
                else None
            )
            decoder_param = next(decoder.parameters(), None) if decoder is not None else None
            decoder_dtype = decoder_param.dtype if decoder_param is not None else slat.feats.dtype
            if slat.feats.dtype != decoder_dtype:
                print(
                    "[trellis_colorizer][debug] casting slat dtype before decode "
                    f"from {slat.feats.dtype} to {decoder_dtype}",
                    file=sys.stderr,
                    flush=True,
                )
                slat = slat.to(dtype=decoder_dtype)

            out = pipeline.decode_slat(slat, ["mesh", "gaussian"])

        glb_mesh = postprocessing_utils.to_glb(
            out["gaussian"][0],
            out["mesh"][0],
            simplify=float(simplify),
            texture_size=int(texture_size),
            verbose=False,
        )
    finally:
        _tsp.set_sparse_backend(prev_backend)

    if not isinstance(glb_mesh, trimesh.Trimesh):
        raise TypeError(f"to_glb expected Trimesh, got {type(glb_mesh)}")
    return glb_mesh


def load_trellis_text_pipeline_eval(
    pretrained_path: str,
    device: "torch.device",
    cache_dir: Optional[str] = None,
) -> Any:
    """Load Trellis text pipeline (same as ``eval.model_loader.load_trellis_text_pipeline``)."""
    from eval.model_loader import load_trellis_text_pipeline

    return load_trellis_text_pipeline(pretrained_path, device=device, cache_dir=cache_dir)
