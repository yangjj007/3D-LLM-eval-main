"""
Dataset for 3D generation evaluation (text-to-3D).

Expected data format (JSON):
[
  {
    "sample_id": "gen_001",
    "prompt": "A drone with four propellers and a central body.",
    "reference_mesh_path": "/path/to/reference.glb"  // optional
  },
  ...
]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base_dataset import EvalDataset
from .metadata_glb_samples import load_generation_from_metadata, parse_gen_caption_indices


class GenerationDataset(EvalDataset):

    def _load_data(self) -> None:
        meta_csv = (self.config.get("metadata_csv") or "").strip()
        glb_dir = (self.config.get("glb_dir") or "").strip()
        if meta_csv and glb_dir:
            idxs = parse_gen_caption_indices(self.config.get("gen_caption_indices", "0"))
            if not idxs:
                idxs = (0,)
            raw = load_generation_from_metadata(
                meta_csv,
                glb_dir,
                idxs,
                self.max_samples,
                self.config.get("sample_seed"),
            )
            if not raw:
                raise ValueError(
                    "GenerationDataset: metadata+glb 模式加载到 0 条样本。"
                    "请确认 glb_dir 下存在与 CSV 对应的 ``{sha256}.glb``（推荐）或 ``{file_identifier}.glb``；"
                    "``file_identifier`` 可为 Sketchfab URL，会尝试用 URL 末段匹配文件名。"
                    "理解/生成任务还需至少一列可读文本：``captions``（JSON 列表）、``caption``、``text`` 等。"
                )
            for i, item in enumerate(raw):
                self.samples.append(
                    {
                        "sample_id": item.get("sample_id", f"gen_{i:04d}"),
                        "prompt": item["prompt"],
                        "mesh_path": item.get("mesh_path"),
                        "reference_mesh_path": item.get("reference_mesh_path", item.get("mesh_path")),
                    }
                )
            return

        if not self.data_path:
            raise ValueError(
                "GenerationDataset: set ``data.data_path`` to a JSON/JSONL/TXT file, "
                "or set ``metadata_csv`` and ``glb_dir`` to load from CSV + GLB."
            )
        data_path = Path(self.data_path)

        if data_path.suffix == ".jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                raw = [json.loads(line) for line in f if line.strip()]
        elif data_path.suffix == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict):
                    raw = list(raw.values())
        elif data_path.suffix == ".txt":
            with open(data_path, "r", encoding="utf-8") as f:
                raw = [
                    {"prompt": line.strip()}
                    for line in f
                    if line.strip()
                ]
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        from .metadata_glb_samples import shuffle_and_truncate_samples

        raw = shuffle_and_truncate_samples(raw, self.max_samples, self.config.get("sample_seed"))

        for i, item in enumerate(raw):
            row: Dict[str, Any] = {
                "sample_id": item.get("sample_id", f"gen_{i:04d}"),
                "prompt": item["prompt"],
                "reference_mesh_path": item.get("reference_mesh_path", item.get("mesh_path")),
            }
            if item.get("mesh_path"):
                row["mesh_path"] = item["mesh_path"]
            self.samples.append(row)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
