"""Serialize 3D BPE macro token ids for LLM text."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from eval.utils.bpe_3d import (
    MESH_TOKEN_RE,
    MORTON_MESH_PAIR_RE,
    MORTON_TOKEN_RE,
    parse_morton_mesh_pairs,
    serialize_morton_mesh_pairs,
)


BPE_ANCHORED_TOKEN_RE = re.compile(r"<mesh_(\d+)@(-?\d+),(-?\d+),(-?\d+)>")
BPE_TOKEN_RE = MESH_TOKEN_RE


def serialize_bpe_sparse_tokens(ids: Sequence[int], anchors: Any = None) -> str:
    """Serialize macro ids with explicit Morton-coded block anchors."""
    ids_np = np.asarray(ids, dtype=np.int64)
    if anchors is not None:
        return serialize_morton_mesh_pairs(ids_np, anchors)
    if ids_np.size == 0:
        return "<mesh_start><mesh_end>"
    parts = [f"<mesh_{int(t)}>" for t in ids_np.tolist()]
    return "<mesh_start>" + "".join(parts) + "<mesh_end>"


def bpe_batches_to_mesh_strings(batches: Sequence[Dict[str, Any]]) -> List[str]:
    """Convert BPE3DTokenizer batch outputs into LLM mesh strings."""
    return [serialize_bpe_sparse_tokens(rec["ids"], rec["anchors"]) for rec in batches]


def parse_bpe_sparse_tokens(response_text: str) -> np.ndarray:
    """Parse LLM output into BPE macro ids.

    New Morton+mesh pair outputs are accepted first. A legacy mesh-only fallback
    keeps older checkpoints/eval artifacts readable.
    """
    pair_result = parse_morton_mesh_pairs(response_text)
    if pair_result.ids.size > 0:
        return pair_result.ids
    return np.asarray([int(m.group(1)) for m in BPE_TOKEN_RE.finditer(response_text)], dtype=np.int64)


def parse_bpe_sparse_token_pairs(
    response_text: str,
    max_mesh_id: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse valid ``<morton_*><mesh_*>`` pairs into ids and anchors."""
    result = parse_morton_mesh_pairs(response_text, max_mesh_id=max_mesh_id)
    return result.ids, result.anchors, result.dropped_count


def parse_anchored_bpe_sparse_tokens(response_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility parser for old debug strings containing anchors."""
    ids: List[int] = []
    anchors: List[Tuple[int, int, int]] = []
    for m in BPE_ANCHORED_TOKEN_RE.finditer(response_text):
        ids.append(int(m.group(1)))
        anchors.append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return np.asarray(ids, dtype=np.int64), np.asarray(anchors, dtype=np.int64).reshape(-1, 3)


def strip_bpe_sparse_tokens(response_text: str) -> str:
    text = re.sub(r"<mesh_start>", "", response_text)
    text = re.sub(r"<mesh_end>", "", text)
    text = re.sub(r"<mesh_empty>", "", text)
    text = BPE_ANCHORED_TOKEN_RE.sub("", text)
    text = MORTON_MESH_PAIR_RE.sub("", text)
    text = MORTON_TOKEN_RE.sub("", text)
    text = BPE_TOKEN_RE.sub("", text)
    return text.strip()
