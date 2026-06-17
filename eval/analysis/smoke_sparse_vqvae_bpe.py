"""Smoke checks for Sparse VQ-VAE BPE token flow and mesh export.

This script avoids loading real checkpoints. It verifies the data contracts that
the eval adapter depends on: BPE id-only serialization for the LLM, coordinate
context reuse from Encode, BPE decode back to a SparseTensor-like object, and
Direct3D-style marching cubes export.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from eval.utils.bpe_3d import BPE3DTokenizer
from eval.utils.bpe_sparse_tokens import parse_bpe_sparse_tokens, serialize_bpe_sparse_tokens
from eval.utils.sparse_mesh_export import sparse_sdf_to_meshes


class SimpleSparseTensor:
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = feats
        self.coords = coords

    def replace(self, new_feats: torch.Tensor) -> "SimpleSparseTensor":
        return SimpleSparseTensor(new_feats, self.coords)


def _dummy_encoded_sparse() -> SimpleSparseTensor:
    coords = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [0, 2, 1, 1],
        ],
        dtype=torch.int32,
    )
    feats = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    return SimpleSparseTensor(feats, coords)


def _dummy_recon_sparse(resolution: int) -> SimpleSparseTensor:
    coords = []
    feats = []
    center = (resolution - 1) / 2.0
    radius = resolution / 4.0
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                d = ((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) ** 0.5
                if abs(d - radius) <= 1.5:
                    coords.append([0, x, y, z])
                    feats.append([float(d - radius)])
    return SimpleSparseTensor(
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(coords, dtype=torch.int32),
    )


def _load_bpe(path: str, base_vocab_size: int) -> BPE3DTokenizer:
    if path:
        return BPE3DTokenizer.load(path)
    return BPE3DTokenizer(base_vocab_size=base_vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_table", type=str, default="")
    parser.add_argument("--base_vocab_size", type=int, default=8192)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--out_obj", type=str, default="")
    args = parser.parse_args()

    tok = _load_bpe(args.merge_table, args.base_vocab_size)
    enc = _dummy_encoded_sparse()
    bpe = tok.encode_sparse(enc, sparse_tensor_cls=SimpleSparseTensor)
    ids = bpe["batches"][0]["ids"]
    anchors = bpe["batches"][0]["anchors"]
    text = serialize_bpe_sparse_tokens(ids)
    parsed_ids = parse_bpe_sparse_tokens(text)
    leaf = tok.decode_to_sparse(
        [{"ids": parsed_ids, "anchors": anchors}],
        device=torch.device("cpu"),
        sparse_tensor_cls=SimpleSparseTensor,
    )
    if leaf.coords.shape[1] != 4 or leaf.feats.shape[1] != 1:
        raise AssertionError("decoded sparse tensor has unexpected shape")

    recon = _dummy_recon_sparse(args.resolution)
    meshes = sparse_sdf_to_meshes(recon, voxel_resolution=args.resolution, mc_threshold=0.0)
    mesh = meshes[0] if meshes else None
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise AssertionError("marching cubes did not produce a mesh")
    if args.out_obj:
        p = Path(args.out_obj)
        p.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(p)

    print(
        "OK sparse BPE smoke: "
        f"macro_tokens={len(ids)} decoded_points={leaf.coords.shape[0]} "
        f"mesh_vertices={len(mesh.vertices)} mesh_faces={len(mesh.faces)}"
    )


if __name__ == "__main__":
    main()
