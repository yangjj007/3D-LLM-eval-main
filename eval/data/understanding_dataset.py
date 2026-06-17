"""
Dataset for 3D understanding evaluation (captioning, QA).

Expected data format (JSON):
[
  {
    "sample_id": "abc123",
    "mesh_path": "/path/to/mesh.glb",
    "prompt": "Describe this 3D object.",
    "ground_truth": "A wooden chair with four legs."
  },
  ...
]

Or JSONL with one JSON object per line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base_dataset import EvalDataset


def _understanding_ref_index_from_data_config(config: Dict[str, Any]) -> int | None:
    """
    When ``data.gen_caption_indices`` is set, use its **first** integer as the
    fixed reference caption index for text metrics (understanding task).

    When the key is absent, return ``None`` and keep ``ground_truths`` as in the
    source JSON (multi-reference compatible).
    """
    if "gen_caption_indices" not in config:
        return None
    from .metadata_glb_samples import parse_gen_caption_indices

    idxs = parse_gen_caption_indices(config.get("gen_caption_indices"))
    return idxs[0] if idxs else None


def _apply_fixed_eval_caption(row: Dict[str, Any], ref_idx: int | None) -> None:
    """Mutate ``row``: single ``ground_truth`` / ``ground_truths`` for text eval."""
    if ref_idx is None:
        return
    gts = row.get("ground_truths")
    if not isinstance(gts, list) or not gts:
        return
    idx = int(ref_idx)
    if idx < 0:
        idx = 0
    elif idx >= len(gts):
        idx = len(gts) - 1
    chosen = gts[idx]
    row["ground_truth"] = chosen
    row["ground_truths"] = [chosen]


class UnderstandingDataset(EvalDataset):

    def _load_data(self) -> None:
        ref_idx = _understanding_ref_index_from_data_config(self.config)
        meta_csv = (self.config.get("metadata_csv") or "").strip()
        glb_dir = (self.config.get("glb_dir") or "").strip()
        if meta_csv and glb_dir:
            from .metadata_glb_samples import load_understanding_from_metadata

            raw = load_understanding_from_metadata(
                meta_csv,
                glb_dir,
                self.config.get(
                    "default_prompt",
                    "Give a quick overview of the object represented by this 3D mesh.",
                ),
                self.max_samples,
                self.config.get("sample_seed"),
                understanding_ref_index=ref_idx,
            )
            self.default_prompt = self.config.get(
                "default_prompt",
                "Give a quick overview of the object represented by this 3D mesh.",
            )
            for i, item in enumerate(raw):
                row = {
                    "sample_id": item.get("sample_id", str(i)),
                    "mesh_path": item["mesh_path"],
                    "prompt": item.get("prompt", self.default_prompt),
                    "ground_truth": item.get("ground_truth", ""),
                    "ground_truths": item.get("ground_truths", [item.get("ground_truth", "")]),
                }
                if item.get("sdf_path"):
                    row["sdf_path"] = item["sdf_path"]
                self.samples.append(row)
            return

        if not self.data_path:
            raise ValueError(
                "UnderstandingDataset: set ``data.data_path`` to a JSON/JSONL/CSV file, "
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
                    raw = list(raw.values()) if all(isinstance(v, dict) for v in raw.values()) else [raw]
        elif data_path.suffix == ".csv":
            from .metadata_glb_samples import load_understanding_from_metadata

            self.default_prompt = self.config.get(
                "default_prompt",
                "Give a quick overview of the object represented by this 3D mesh.",
            )
            glb_dir = (self.config.get("glb_dir") or str(data_path.parent)).strip()
            raw = load_understanding_from_metadata(
                str(data_path),
                glb_dir,
                self.default_prompt,
                self.max_samples,
                self.config.get("sample_seed"),
                understanding_ref_index=ref_idx,
            )
            if not raw:
                raise ValueError(
                    "UnderstandingDataset: data_path CSV 模式加载到 0 条样本。"
                    "请确认 CSV 中存在可定位 GLB 的 ``sha256`` 或 ``file_identifier`` 列，"
                    "且 ``glb_dir``（默认 CSV 所在目录）下存在对应的 ``.glb`` 文件；"
                    "还需至少一列可读文本：``captions``（JSON 列表）、``caption``、``text`` 等。"
                )
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        from .metadata_glb_samples import shuffle_and_truncate_samples

        loaded_via_metadata_csv = data_path.suffix == ".csv"

        if not loaded_via_metadata_csv:
            raw = shuffle_and_truncate_samples(raw, self.max_samples, self.config.get("sample_seed"))

            self.default_prompt = self.config.get(
                "default_prompt",
                "Give a quick overview of the object represented by this 3D mesh.",
            )

        for i, item in enumerate(raw):
            row = {
                "sample_id": item.get("sample_id", str(i)),
                "mesh_path": item["mesh_path"],
                "prompt": item.get("prompt", self.default_prompt),
                "ground_truth": item.get("ground_truth", ""),
                "ground_truths": item.get("ground_truths", [item.get("ground_truth", "")]),
            }
            if item.get("sdf_path"):
                row["sdf_path"] = item["sdf_path"]
            if not loaded_via_metadata_csv:
                _apply_fixed_eval_caption(row, ref_idx)
            self.samples.append(row)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
