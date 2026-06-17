"""Parse / strip sparse VQVAE mesh tokens."""

from __future__ import annotations

import re
from typing import List, Tuple

from eval.utils.bpe_sparse_tokens import (
    parse_bpe_sparse_tokens,
    strip_bpe_sparse_tokens,
)


MESH_TOKEN_RE = re.compile(r"<mesh_(\d+)>")


def parse_sparse_mesh_tokens(response_text: str) -> List[int]:
    """Extract codebook (or BPE macro) indices from LLM output."""
    return [int(m) for m in MESH_TOKEN_RE.findall(response_text)]


def strip_sparse_mesh_tokens(response_text: str) -> str:
    try:
        return strip_bpe_sparse_tokens(response_text)
    except Exception:
        pass
    text = re.sub(r"<mesh_start>", "", response_text)
    text = re.sub(r"<mesh_end>", "", text)
    text = re.sub(r"<mesh_empty>", "", text)
    text = re.sub(r"<morton_\d+>", "", text)
    text = re.sub(r"<mesh_\d+>", "", text)
    return text.strip()


def split_mesh_block(user_content: str) -> Tuple[str, str]:
    """If content starts with <mesh_start>, split mesh prefix and rest."""
    if "<mesh_start>" in user_content and "<mesh_end>" in user_content:
        end = user_content.find("<mesh_end>") + len("<mesh_end>")
        mesh_part = user_content[:end]
        rest = user_content[end:].lstrip("\n")
        return mesh_part, rest
    return "", user_content
