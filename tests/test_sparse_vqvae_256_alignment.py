from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from eval.adapters.base import MeshInput
from eval.adapters.sparse_sdf_adapter import SparseSDFQwen3Adapter
from eval.utils import sdf_processing


def _adapter(model_cfg: dict | None = None) -> SparseSDFQwen3Adapter:
    adapter = SparseSDFQwen3Adapter()
    adapter._cfg = {"model": model_cfg or {}}
    adapter._eval_debug = {"verbose_eval": False}
    adapter._device = torch.device("cpu")
    return adapter


def test_mesh_to_sparse_sdf_tensors_uses_med_256_defaults(monkeypatch):
    calls = {}

    class FakeMesh:
        vertices = [0, 1, 2]
        faces = [0]

    fake_trimesh = types.ModuleType("trimesh")
    fake_trimesh.Trimesh = FakeMesh
    fake_trimesh.load = lambda path, force="mesh": FakeMesh()

    fake_mesh_utils = types.ModuleType("trellis.utils.mesh_utils")

    def fake_mesh2sparse_sdf(mesh, **kwargs):
        calls.update(kwargs)
        return {
            "sparse_sdf": np.array([0.0, 0.2], dtype=np.float32),
            "sparse_index": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            "edge_mask": np.array([True, False], dtype=np.bool_),
            "resolution": kwargs["resolution"],
            "extra_band_factor": kwargs["threshold_factor"],
        }

    fake_mesh_utils.mesh2sparse_sdf = fake_mesh2sparse_sdf
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)
    monkeypatch.setitem(sys.modules, "trellis", types.ModuleType("trellis"))
    monkeypatch.setitem(sys.modules, "trellis.utils", types.ModuleType("trellis.utils"))
    monkeypatch.setitem(sys.modules, "trellis.utils.mesh_utils", fake_mesh_utils)

    out = sdf_processing.mesh_to_sparse_sdf_tensors("shape.glb")

    assert calls == {
        "resolution": 256,
        "threshold_factor": 4.0,
        "normalize": True,
        "scale": 0.95,
        "watertight": False,
        "compute_edge_mask": True,
        "sharp_grad_dev_thresh": 0.5,
    }
    assert out["sparse_sdf"].shape == (2, 1)
    assert out["edge_mask"].tolist() == [True, False]


def test_sdf_cache_rebuilds_when_metadata_mismatches(monkeypatch, tmp_path):
    calls = []

    def fake_builder(
        mesh_path,
        resolution,
        threshold_factor,
        *,
        watertight=False,
        compute_edge_mask=True,
        sharp_grad_dev_thresh=0.5,
    ):
        calls.append((resolution, threshold_factor, compute_edge_mask, sharp_grad_dev_thresh))
        return {
            "sparse_sdf": torch.tensor([[0.0], [0.1]], dtype=torch.float32),
            "sparse_index": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            "edge_mask": torch.tensor([True, False]),
        }

    monkeypatch.setattr(sdf_processing, "mesh_to_sparse_sdf_tensors", fake_builder)
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "sample_r256.npz"

    sdf_processing.get_or_build_sdf_for_sample(
        "sample.glb", None, str(cache_dir), 256, 4.0, sample_id="sample"
    )
    assert cache_path.is_file()
    assert calls == [(256, 4.0, True, 0.5)]

    np.savez_compressed(
        cache_path,
        sparse_sdf=np.array([[9.0]], dtype=np.float32),
        sparse_index=np.array([[9, 9, 9]], dtype=np.int64),
        resolution=np.array(512, dtype=np.int32),
        extra_band_factor=np.array(0.5, dtype=np.float32),
    )

    sdf_processing.get_or_build_sdf_for_sample(
        "sample.glb", None, str(cache_dir), 256, 4.0, sample_id="sample"
    )
    assert calls == [(256, 4.0, True, 0.5), (256, 4.0, True, 0.5)]

    cached = sdf_processing.get_or_build_sdf_for_sample(
        "sample.glb", None, str(cache_dir), 256, 4.0, sample_id="sample"
    )
    assert calls == [(256, 4.0, True, 0.5), (256, 4.0, True, 0.5)]
    assert cached["sparse_sdf"].shape[0] == 2


def test_adapter_get_sdf_passes_256_training_defaults(monkeypatch):
    adapter = _adapter()
    recorded = {}

    def fake_get_or_build(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return {
            "sparse_sdf": torch.tensor([[0.0]], dtype=torch.float32),
            "sparse_index": torch.tensor([[1, 2, 3]], dtype=torch.long),
        }

    monkeypatch.setattr(sdf_processing, "get_or_build_sdf_for_sample", fake_get_or_build)

    adapter._get_sdf(MeshInput(sample_id="sample", mesh_path="sample.glb", sdf_path="old.npz"))

    assert recorded["args"][:5] == ("sample.glb", None, None, 256, 4.0)
    assert recorded["kwargs"] == {
        "sample_id": "sample",
        "watertight": False,
        "compute_edge_mask": True,
        "sharp_grad_dev_thresh": 0.5,
    }


def test_tight_band_filters_encoder_input_to_0125():
    adapter = _adapter(
        {
            "input_band_factor": 0.5,
            "preprocessing_extra_band_factor": 4.0,
        }
    )
    sparse = {
        "sparse_sdf": torch.tensor([[-0.2], [-0.1], [0.0], [0.13]], dtype=torch.float32),
        "sparse_index": torch.tensor(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.long
        ),
    }

    filtered, n_keep, n_total, threshold = adapter._filter_sparse_for_encoder(
        sparse, sample_id="s", mesh_path="s.glb"
    )

    assert threshold == pytest.approx(0.125)
    assert (n_keep, n_total) == (2, 4)
    assert torch.allclose(filtered["sparse_sdf"].squeeze(-1), torch.tensor([-0.1, 0.0]))
    assert filtered["sparse_index"].tolist() == [[1, 1, 1], [2, 2, 2]]


def test_tight_band_empty_raises_readable_error():
    adapter = _adapter()
    sparse = {
        "sparse_sdf": torch.tensor([[0.2], [-0.3]], dtype=torch.float32),
        "sparse_index": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long),
    }

    with pytest.raises(RuntimeError, match="tight encoder band is empty"):
        adapter._filter_sparse_for_encoder(sparse, sample_id="empty", mesh_path="empty.glb")


def test_decode_helper_passes_256_pruning_params():
    adapter = _adapter(
        {
            "sdf_resolution": 256,
            "inference_band_factor": 2.0,
            "inference_occ_resolution": 256,
        }
    )

    class FakeDecoded:
        coords = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    class FakeVAE:
        resolution = 32
        vq_block_side = 1

        def __init__(self):
            self.kwargs = None

        def Decode(self, decoded_sparse, *, band_prune=None, gt_prune=None):
            self.kwargs = {"band_prune": band_prune, "gt_prune": gt_prune}
            return "decoded"

    vae = FakeVAE()

    assert adapter._decode_sparse_with_pruning(vae, FakeDecoded()) == "decoded"
    assert vae.kwargs["band_prune"]["mode"] == "seed"
    assert torch.equal(vae.kwargs["band_prune"]["seed_coords"], FakeDecoded.coords)
    assert vae.kwargs["band_prune"]["seed_resolution"] == 32
    assert vae.kwargs["band_prune"]["output_resolution"] == 256
    assert vae.kwargs["band_prune"]["extra_band_factor"] == 2.0
    assert vae.kwargs["gt_prune"] == {
        "mode": "geometry",
        "extra_band_factor": 2.0,
        "resolution": 256,
        "occ_resolution": 256,
    }
