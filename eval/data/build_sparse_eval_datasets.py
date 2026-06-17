#!/usr/bin/env python3
"""
[Legacy / optional] Augment eval JSON with ``sdf_path`` pointing at precomputed
``{sha256}_r{resolution}.npz`` under ``--sdf_dir``.

**Default Sparse 评测不再需要本脚本**：``sparse_sdf_qwen3`` 在 ``model.sdf_from_mesh_only: true``
（默认）时从 ``mesh_path``（GLB）在线计算稀疏 SDF；仅当你显式关闭该开关并希望读取
训练阶段离线写好的 NPZ 时才使用本工具生成 ``*_sparse.json``。

Outputs ``*_sparse.json`` next to inputs (or under ``--output_dir``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _stem_from_mesh(mesh_path: str) -> Optional[str]:
    """Try to infer Objaverse-like id from filename (e.g. <uid>.glb)."""
    p = Path(mesh_path)
    stem = p.stem
    if len(stem) == 32 and all(c in "0123456789abcdef" for c in stem.lower()):
        return stem.lower()
    return None


def find_npz_for_mesh(mesh_path: str, sdf_dir: Path, resolution: int) -> Optional[str]:
    uid = _stem_from_mesh(mesh_path)
    if uid:
        cand = sdf_dir / f"{uid}_r{resolution}.npz"
        if cand.is_file():
            return str(cand.resolve())
    return None


def augment_list(items: List[Dict[str, Any]], sdf_dir: Path, resolution: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        row = dict(item)
        mp = row.get("mesh_path")
        if mp:
            p = find_npz_for_mesh(str(mp), sdf_dir, resolution)
            if p:
                row["sdf_path"] = p
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_eval_dir", type=str, default="eval_data", help="Dir containing *.json")
    ap.add_argument("--sdf_dir", type=str, required=True, help="Dir with *_r{res}.npz")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = ap.parse_args()

    base = Path(args.input_eval_dir)
    sdf_dir = Path(args.sdf_dir)
    out_root = Path(args.output_dir) if args.output_dir else base
    out_root.mkdir(parents=True, exist_ok=True)

    for name in ("understanding.json", "generation.json", "vqvae_recon.json"):
        src = base / name
        if not src.is_file():
            continue
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        aug = augment_list(data, sdf_dir, args.resolution)
        dst = out_root / name.replace(".json", "_sparse.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(aug, f, indent=2, ensure_ascii=False)
        matched = sum(1 for x in aug if x.get("sdf_path"))
        print(f"[build_sparse] Wrote {dst} ({len(aug)} rows, {matched} with sdf_path)")


if __name__ == "__main__":
    main()
