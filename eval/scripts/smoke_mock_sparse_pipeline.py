#!/usr/bin/env python3
"""
Server-side smoke checks for mock LLM + sparse VQVAE + Trellis path (no full eval).

Usage (from ``3D-LLM-eval-main`` repo root)::

    pip install -r requirements.txt
    python eval/scripts/smoke_mock_sparse_pipeline.py

Checks:
  - ``third_party`` (optional) and repo-root ``trellis`` import
  - BPE mock string parses
  - Optional: torchmetrics / CLIP imports
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from eval.utils.path_bootstrap import ensure_third_party_on_path

    ensure_third_party_on_path()
    print("[smoke] path bootstrap OK")

    try:
        import trellis  # noqa: F401

        print("[smoke] import trellis OK")
    except Exception as exc:
        print("[smoke] import trellis FAILED (install full requirements on GPU server):", exc)
        # 不阻断：本地 CI 可能未装 easydict / torchsparse 等

    from eval.utils.bpe_sparse_tokens import parse_bpe_sparse_token_pairs

    raw = "<mesh_start><morton_0><mesh_0><mesh_end>"
    ids, anchors, dropped = parse_bpe_sparse_token_pairs(raw, max_mesh_id=8192)
    print("[smoke] parse mock string:", ids.tolist(), anchors.tolist(), "dropped", dropped)

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance  # noqa: F401

        print("[smoke] torchmetrics FID import OK")
    except Exception as exc:
        print("[smoke] torchmetrics import skipped/failed:", exc)

    try:
        from transformers import CLIPModel, CLIPProcessor  # noqa: F401

        print("[smoke] transformers CLIP imports OK")
    except Exception as exc:
        print("[smoke] CLIP import failed (pip install transformers):", exc)

    print("[smoke] all lightweight checks done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
