"""
Dataset for VQVAE reconstruction quality evaluation.

Expected data format (JSON):
[
  {"sample_id": "vq_001", "mesh_path": "/path/to/mesh.glb"},
  ...
]

Or a plain text file with one mesh path per line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base_dataset import EvalDataset


class VQVAEDataset(EvalDataset):
    def _infer_metadata_glb_from_data_path(self) -> tuple[str, str]:
        """
        Best-effort fallback for legacy configs that only set ``data_path``.
        If ``metadata.csv`` is colocated with the JSON, switch to strict
        metadata+glb loading to avoid stale absolute mesh paths in JSON.
        """
        if not self.data_path:
            return "", ""
        dp = Path(self.data_path).resolve()
        candidates = [
            dp.parent / "metadata.csv",
            dp.parent.parent / "metadata.csv",
        ]
        for meta in candidates:
            if meta.is_file():
                # Common flat layout: metadata.csv + *.glb in same directory.
                gdir = meta.parent
                return str(meta), str(gdir)
        return "", ""

    def _load_data(self) -> None:
        meta_csv = (self.config.get("metadata_csv") or "").strip()
        glb_dir = (self.config.get("glb_dir") or "").strip()
        if not (meta_csv and glb_dir):
            # Auto-upgrade old data_path-only setups when possible.
            infer_meta, infer_glb = self._infer_metadata_glb_from_data_path()
            if infer_meta and infer_glb:
                meta_csv, glb_dir = infer_meta, infer_glb
                print(
                    f"[VQVAEDataset] detected metadata mode from data_path: "
                    f"metadata_csv={meta_csv}, glb_dir={glb_dir}"
                )

        if meta_csv and glb_dir:
            from .metadata_glb_samples import load_vqvae_from_metadata

            raw = load_vqvae_from_metadata(
                meta_csv, glb_dir, self.max_samples, self.config.get("sample_seed")
            )
            if not raw:
                raise ValueError(
                    "VQVAEDataset: metadata+glb mode loaded 0 samples. "
                    "Please check that metadata rows resolve to a ``.glb`` under glb_dir "
                    "(prefer ``{sha256}.glb`` when a sha256 column is present) and that "
                    "file_identifier / sha256 columns are set."
                )
            for i, item in enumerate(raw):
                row = {
                    "sample_id": item.get("sample_id", f"vq_{i:04d}"),
                    "mesh_path": item["mesh_path"],
                }
                if item.get("sdf_path"):
                    row["sdf_path"] = item["sdf_path"]
                self.samples.append(row)
            return

        if not self.data_path:
            raise ValueError(
                "VQVAEDataset: set ``data.data_path`` to a JSON/JSONL/TXT file, "
                "or set ``metadata_csv`` and ``glb_dir`` to load from CSV + GLB."
            )
        data_path = Path(self.data_path)

        if data_path.suffix == ".txt":
            with open(data_path, "r", encoding="utf-8") as f:
                paths = [line.strip() for line in f if line.strip()]
            raw = [{"mesh_path": p} for p in paths]
        elif data_path.suffix in (".json", ".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                if data_path.suffix == ".jsonl":
                    raw = [json.loads(line) for line in f if line.strip()]
                else:
                    raw = json.load(f)
                    if isinstance(raw, dict):
                        raw = list(raw.values())
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        from .metadata_glb_samples import shuffle_and_truncate_samples

        raw = shuffle_and_truncate_samples(raw, self.max_samples, self.config.get("sample_seed"))

        for i, item in enumerate(raw):
            row = {
                "sample_id": item.get("sample_id", f"vq_{i:04d}"),
                "mesh_path": item["mesh_path"],
            }
            if item.get("sdf_path"):
                row["sdf_path"] = item["sdf_path"]
            self.samples.append(row)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
