"""Ensure ``third_party`` (e.g. vox2seq) is on ``sys.path``. Trellis uses repo-root ``trellis/``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DONE = False


def repo_root() -> Path:
    """``3D-LLM-eval-main`` directory (parent of ``eval``)."""
    return Path(__file__).resolve().parents[2]


def ensure_third_party_on_path() -> None:
    """Insert ``<repo>/third_party`` at the front of ``sys.path`` (and keep it first if re-present)."""
    global _DONE
    if _DONE:
        return
    tp = repo_root() / "third_party"
    if tp.is_dir():
        p = str(tp)
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    _DONE = True


def ensure_eval_repo_on_path() -> None:
    """Insert repo root so ``python -m eval.runner`` resolves ``eval`` package."""
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
