"""
Append-only per-sample JSONL + mesh export for resumable evaluation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def scan_done_sample_ids(jsonl_path: str) -> Set[str]:
    """Return set of sample_id already present in per_sample.jsonl (or rank shards)."""
    done: Set[str] = set()
    p = Path(jsonl_path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get("sample_id")
                    if sid:
                        done.add(str(sid))
                except json.JSONDecodeError:
                    continue
        return done

    parent = p.parent
    if parent.is_dir():
        for shard in sorted(parent.glob("per_sample.rank*.jsonl")):
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        sid = obj.get("sample_id")
                        if sid:
                            done.add(str(sid))
                    except json.JSONDecodeError:
                        continue
    return done


class ResultStore:
    """Writes per-sample JSONL lines and optional mesh .obj files."""

    def __init__(
        self,
        output_dir: str,
        adapter_name: str,
        task: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.adapter_name = adapter_name
        self.task = task
        self.rank = rank
        self.world_size = world_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_dir = self.output_dir / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            self.jsonl_path = self.output_dir / f"per_sample.rank{rank}.jsonl"
        else:
            self.jsonl_path = self.output_dir / "per_sample.jsonl"
        self._fp = open(self.jsonl_path, "a", encoding="utf-8")

    def append_record(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("timestamp", int(time.time()))
        record.setdefault("adapter", self.adapter_name)
        record.setdefault("task", self.task)
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def save_mesh_obj(
        self,
        sample_id: str,
        mesh: Any,
        *,
        safe_id: Optional[str] = None,
    ) -> str:
        """
        Export trimesh to meshes/{safe_id}.obj (geometry only).
        Returns relative path from output_dir.
        """
        import re
        import trimesh

        sid = safe_id or re.sub(r"[^\w.\-]+", "_", sample_id)[:200]
        path = self.mesh_dir / f"{sid}.obj"
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
        # trimesh API differs by version: some export_obj implementations
        # don't accept include_attributes.
        try:
            mesh.export(str(path), file_type="obj", include_attributes=False)
        except TypeError:
            mesh.export(str(path), file_type="obj")
        rel = f"meshes/{sid}.obj"
        return rel

    def save_glb(
        self,
        sample_id: str,
        mesh: Any,
        *,
        safe_id: Optional[str] = None,
    ) -> str:
        """Export textured ``trimesh`` (e.g. from Trellis ``to_glb``) to ``meshes/{id}.glb``."""
        import re
        import trimesh

        sid = safe_id or re.sub(r"[^\w.\-]+", "_", sample_id)[:200]
        path = self.mesh_dir / f"{sid}.glb"
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh for GLB export, got {type(mesh)}")
        mesh.export(str(path), file_type="glb")
        return f"meshes/{sid}.glb"

    def close(self) -> None:
        if self._fp and not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def merge_rank_jsonls(output_dir: str, world_size: int, final_name: str = "per_sample.jsonl") -> Path:
    """Merge per_sample.rank*.jsonl into per_sample.jsonl (last write wins by sample_id)."""
    out_dir = Path(output_dir)
    merged: Dict[str, Dict[str, Any]] = {}
    for r in range(world_size):
        shard = out_dir / f"per_sample.rank{r}.jsonl"
        if not shard.is_file():
            continue
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = str(obj.get("sample_id", ""))
                if sid:
                    merged[sid] = obj
    final = out_dir / final_name
    with open(final, "w", encoding="utf-8") as f:
        for sid in sorted(merged.keys()):
            f.write(json.dumps(merged[sid], ensure_ascii=False) + "\n")
    return final
