"""
Shared logic: build eval sample dicts from metadata.csv + a directory of .glb files.

Used by ``build_eval_from_metadata`` CLI and by datasets when ``metadata_csv`` +
``glb_dir`` are set in YAML instead of ``data_path``.

**GLB 文件名**

- 若 CSV 含 **64 位十六进制** ``sha256`` 列且 ``glb_dir/{sha256}.glb`` 存在（如
  ``sample_objaverse_glb_subset`` 扁平输出），优先用其定位网格。
- 否则仍支持旧版 ``{file_identifier}.glb``；``file_identifier`` 为 Sketchfab 等 **URL** 时，
  会尝试用 URL **最后一段**作为 ``.glb`` 文件名 stem。

**文本列（理解 / 生成任务）**

- ``captions``：JSON 字符串数组（与旧逻辑一致）。
- 若无或非 JSON，则依次尝试单列字符串：``caption``、``text``、``name``、``title``、``description``。
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_gen_caption_indices(value: Any) -> Tuple[int, ...]:
    """
    Parse ``gen_caption_indices`` from YAML/CLI: ``0``, ``"0"``, ``"0,5,10"``, or ``[0, 5]``.

    Returns empty tuple only when *value* is empty string / whitespace after strip.
    ``None`` → empty tuple (caller may treat as “unset”).
    """
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, list):
        out: List[int] = []
        for x in value:
            if x is None or (isinstance(x, str) and not str(x).strip()):
                continue
            out.append(int(x))
        return tuple(out)
    s = str(value).strip()
    if not s:
        return ()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def shuffle_and_truncate_samples(
    items: List[Dict[str, Any]],
    max_samples: int | None,
    sample_seed: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Optionally shuffle (reproducible) then apply ``max_samples`` head slice.

    - ``sample_seed`` is ``None``: preserve input order; only ``[:max_samples]`` applies.
    - ``sample_seed`` is set: shuffle a copy with ``random.Random(sample_seed)``, then truncate.
    """
    out = list(items)
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(out)
    if max_samples is not None:
        out = out[:max_samples]
    return out


def _cell_ci(row: Dict[str, str], *names: str) -> str:
    """Case-insensitive lookup for CSV column names (e.g. ``SHA256`` vs ``sha256``)."""
    lower_map = {str(k).strip().lower(): v for k, v in row.items()}
    for n in names:
        v = lower_map.get(n.lower().strip())
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _parse_captions(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import ast

            val = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Cannot parse captions as JSON: {raw[:120]}...") from e
    if not isinstance(val, list):
        raise ValueError(f"captions must be a JSON list, got {type(val)}")
    out: List[str] = []
    for x in val:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def _captions_from_row(row: Dict[str, str]) -> List[str]:
    """
    Text prompts for understanding / generation.

    Priority: ``captions`` (JSON list) → ``caption`` / ``text`` / ``name`` (single string).
    """
    raw_caps = _cell_ci(row, "captions")
    if raw_caps:
        try:
            return _parse_captions(raw_caps)
        except ValueError:
            pass
    for key in ("caption", "text", "name", "title", "description"):
        s = _cell_ci(row, key)
        if s:
            return [s]
    return []


def _resolve_glb_path(row: Dict[str, str], glb_dir: Path) -> Optional[Path]:
    """
    Resolve ``*.glb`` under ``glb_dir`` for one metadata row.

    Supports:

    - **sha256 layout** (e.g. ``sample_objaverse_glb_subset`` flat output): ``{sha256}.glb``.
    - **Legacy**: ``{file_identifier}.glb`` when ``file_identifier`` is the filename stem.
    - **Sketchfab URL** ``file_identifier``: try URL last segment as ``{id}.glb``.
    - **Path-like** ``file_identifier``: basename as stem.
    """
    gdir = glb_dir.resolve()
    sha = _cell_ci(row, "sha256").lower()
    if sha and all(c in "0123456789abcdef" for c in sha) and len(sha) == 64:
        p = gdir / f"{sha}.glb"
        if p.is_file():
            return p.resolve()

    fid = _cell_ci(row, "file_identifier")
    if not fid:
        return None

    candidates: List[Path] = []
    candidates.append(gdir / f"{fid}.glb")

    if fid.startswith("http://") or fid.startswith("https://"):
        tail = fid.rstrip("/").split("/")[-1]
        if tail.endswith(".glb"):
            candidates.append(gdir / tail)
        else:
            candidates.append(gdir / f"{tail}.glb")

    norm = fid.replace("\\", "/").lstrip("/")
    base = norm.split("/")[-1] if norm else fid
    if base and base != fid:
        if base.endswith(".glb"):
            candidates.append(gdir / base)
        else:
            candidates.append(gdir / f"{base}.glb")

    seen: set[str] = set()
    for c in candidates:
        key = str(c.resolve())
        if key in seen:
            continue
        seen.add(key)
        if c.is_file():
            return c.resolve()
    return None


def _sample_id_for_row(row: Dict[str, str], glb_path: Path) -> str:
    sha = _cell_ci(row, "sha256").lower()
    if sha and all(c in "0123456789abcdef" for c in sha) and len(sha) == 64:
        return sha
    return glb_path.stem


def read_metadata_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: (v or "").strip() if v is not None else "" for k, v in row.items()})
        return rows


def _understanding_gt_for_eval(
    caps: List[str],
    understanding_ref_index: Optional[int],
) -> Tuple[str, List[str]]:
    """Ground truth fields for understanding / text metrics."""
    if not caps:
        return "", []
    if understanding_ref_index is None:
        return caps[0], list(caps)
    idx = int(understanding_ref_index)
    if idx < 0:
        idx = 0
    elif idx >= len(caps):
        idx = len(caps) - 1
    chosen = caps[idx]
    return chosen, [chosen]


def build_samples(
    rows: Sequence[Dict[str, str]],
    glb_dir: Path,
    gen_caption_indices: Tuple[int, ...],
    default_prompt: str,
    *,
    understanding_ref_index: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Returns (understanding, generation, stats)."""
    understanding: List[Dict[str, Any]] = []
    generation: List[Dict[str, Any]] = []
    stats = {"rows_total": 0, "rows_skipped_no_glb": 0, "rows_skipped_no_captions": 0}

    for row in rows:
        stats["rows_total"] += 1
        glb_path = _resolve_glb_path(row, glb_dir)
        if glb_path is None:
            stats["rows_skipped_no_glb"] += 1
            continue
        mesh_path = str(glb_path)
        sid = _sample_id_for_row(row, glb_path)
        caps = _captions_from_row(row)
        if not caps:
            stats["rows_skipped_no_captions"] += 1
            continue

        gt, gts = _understanding_gt_for_eval(caps, understanding_ref_index)
        understanding.append(
            {
                "sample_id": sid,
                "mesh_path": mesh_path,
                "prompt": default_prompt,
                "ground_truth": gt,
                "ground_truths": gts,
            }
        )

        for ci in gen_caption_indices:
            if ci < 0 or ci >= len(caps):
                continue
            generation.append(
                {
                    "sample_id": f"{sid}_cap{ci}",
                    "prompt": caps[ci],
                    "mesh_path": mesh_path,
                    "reference_mesh_path": mesh_path,
                }
            )

    return understanding, generation, stats


def build_vqvae_samples(rows: Sequence[Dict[str, str]], glb_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """One row per resolved GLB (``sha256.glb`` or ``{file_identifier}.glb``); captions not required."""
    out: List[Dict[str, Any]] = []
    stats = {"rows_total": 0, "rows_skipped_no_glb": 0}
    for row in rows:
        stats["rows_total"] += 1
        glb_path = _resolve_glb_path(row, glb_dir)
        if glb_path is None:
            stats["rows_skipped_no_glb"] += 1
            continue
        sid = _sample_id_for_row(row, glb_path)
        out.append({"sample_id": sid, "mesh_path": str(glb_path)})
    return out, stats


def load_understanding_from_metadata(
    metadata_csv: str,
    glb_dir: str,
    default_prompt: str,
    max_samples: int | None = None,
    sample_seed: int | None = None,
    understanding_ref_index: int | None = None,
) -> List[Dict[str, Any]]:
    meta_path = Path(metadata_csv).resolve()
    gdir = Path(glb_dir).resolve()
    rows = read_metadata_rows(meta_path)
    u_list, _, _ = build_samples(
        rows,
        gdir,
        (0,),
        default_prompt,
        understanding_ref_index=understanding_ref_index,
    )
    return shuffle_and_truncate_samples(u_list, max_samples, sample_seed)


def load_generation_from_metadata(
    metadata_csv: str,
    glb_dir: str,
    gen_caption_indices: Tuple[int, ...],
    max_samples: int | None = None,
    sample_seed: int | None = None,
) -> List[Dict[str, Any]]:
    meta_path = Path(metadata_csv).resolve()
    gdir = Path(glb_dir).resolve()
    rows = read_metadata_rows(meta_path)
    _, g_list, _ = build_samples(
        rows, gdir, gen_caption_indices, "", understanding_ref_index=None
    )
    return shuffle_and_truncate_samples(g_list, max_samples, sample_seed)


def load_vqvae_from_metadata(
    metadata_csv: str,
    glb_dir: str,
    max_samples: int | None = None,
    sample_seed: int | None = None,
) -> List[Dict[str, Any]]:
    meta_path = Path(metadata_csv).resolve()
    gdir = Path(glb_dir).resolve()
    rows = read_metadata_rows(meta_path)
    v_list, _ = build_vqvae_samples(rows, gdir)
    return shuffle_and_truncate_samples(v_list, max_samples, sample_seed)
