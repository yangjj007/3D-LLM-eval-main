#!/usr/bin/env python3
"""
Build unified eval JSON from metadata.csv + a directory of .glb meshes.

Outputs (under ``--output_dir``):
  - understanding.json -- mesh-to-caption: ``mesh_path``, ``ground_truths`` (all captions),
    ``ground_truth`` (first caption).
  - generation.json -- caption-to-3D: ``prompt``, ``mesh_path`` / ``reference_mesh_path`` (same glb),
    one row per (object, caption index) selected by ``--gen_caption_indices``.

Expected CSV columns (at minimum): ``file_identifier``, ``captions``.
``captions`` must be a JSON array of strings (e.g. 11 captions from detailed to short).

Usage:
    python -m eval.data.build_eval_from_metadata \\
        --metadata_csv eval_data/metadata.csv \\
        --glb_dir eval_data \\
        --output_dir eval_data \\
        --gen_caption_indices 0

For in-memory loading without writing JSON, set ``metadata_csv`` + ``glb_dir`` in the task YAML
and omit ``data_path`` (see ``eval.data.metadata_glb_samples``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

from .metadata_glb_samples import build_samples, read_metadata_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata.csv")
    ap.add_argument("--glb_dir", type=str, required=True, help="Directory containing {file_identifier}.glb")
    ap.add_argument("--output_dir", type=str, default="eval_data", help="Where to write JSON files")
    ap.add_argument(
        "--gen_caption_indices",
        type=str,
        default="0",
        help="Comma-separated caption indices for generation task (e.g. 0 or 0,5,10)",
    )
    ap.add_argument(
        "--default_prompt",
        type=str,
        default="Give a quick overview of the object represented by this 3D mesh.",
        help="Prompt for understanding task when not stored per-row in CSV",
    )
    args = ap.parse_args()

    meta_path = Path(args.metadata_csv).resolve()
    glb_dir = Path(args.glb_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.is_file():
        print(f"[error] metadata not found: {meta_path}", file=sys.stderr)
        sys.exit(1)
    if not glb_dir.is_dir():
        print(f"[error] glb_dir not a directory: {glb_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        idxs: Tuple[int, ...] = tuple(
            int(x.strip()) for x in args.gen_caption_indices.split(",") if x.strip()
        )
    except ValueError:
        print("[error] --gen_caption_indices must be comma-separated integers", file=sys.stderr)
        sys.exit(1)
    if not idxs:
        print("[error] --gen_caption_indices is empty", file=sys.stderr)
        sys.exit(1)

    rows = read_metadata_rows(meta_path)
    u_list, g_list, stats = build_samples(rows, glb_dir, idxs, args.default_prompt)

    u_path = out_dir / "understanding.json"
    g_path = out_dir / "generation.json"
    with open(u_path, "w", encoding="utf-8") as f:
        json.dump(u_list, f, indent=2, ensure_ascii=False)
    with open(g_path, "w", encoding="utf-8") as f:
        json.dump(g_list, f, indent=2, ensure_ascii=False)

    print(f"[build_eval_from_metadata] Wrote {u_path} ({len(u_list)} samples)")
    print(f"[build_eval_from_metadata] Wrote {g_path} ({len(g_list)} samples)")
    print(
        f"[build_eval_from_metadata] stats: rows={stats['rows_total']}, "
        f"skipped_no_glb_or_bad_id={stats['rows_skipped_no_glb']}, "
        f"skipped_no_captions={stats['rows_skipped_no_captions']}"
    )


if __name__ == "__main__":
    main()
