"""
从本地 TRELLIS / ObjaverseXL（hf-objaverse-v1）目录中随机抽取若干已存在的 .glb，
并复制到新目录，同时写出对应的 metadata.csv 子集与 object-paths.json 子集。

默认 **完全扁平**（与 ShapeLLM-Omni 等用法一致）：所有模型为 ``<sha256>.glb``，与
``metadata.csv``、``object-paths.json`` 同在 ``output_dir`` 根目录。

可选 ``--nested`` 恢复原先分层目录（``glbs/000-xxx/...``），便于直接作为
``sdf_voxelize --format trellis500k`` 的 ``--input_dir``（见 SDF 文档）。

**注意：** 扁平模式请使用**专用空目录**（例如 ``./ObjaverseXL_flat_5k``），不要把 ``--output_dir``
指到已存在 ``glbs/000-xxx`` 下载树的项目 ``data/`` 根目录，否则会与旧目录混杂；脚本会检测并拒绝。

使用示例::

 python dataset_toolkits/sample_objaverse_glb_subset.py \
        --input_dir ./TRELLIS-500K/ObjaverseXL/raw/hf-objaverse-v1 \
        --output_dir ./ObjaverseXL_flat_5k \
        --num_samples 5000 \
        --seed 42 \
        --max_workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _find_metadata_csv(input_dir: str) -> Path:
    """与 sdf_voxelize.load_trellis500k_metadata 一致：在上一级或上上一级找 metadata.csv。"""
    p = Path(input_dir).resolve()
    parent = p.parent
    for _ in range(2):
        cand = parent / "metadata.csv"
        if cand.is_file():
            return cand
        parent = parent.parent
    raise FileNotFoundError(
        "未找到 metadata.csv。已尝试:\n"
        f"  {p.parent / 'metadata.csv'}\n"
        f"  {p.parent.parent / 'metadata.csv'}"
    )


def _looks_like_objaverse_bucket_glbs_dir(glbs_dir: Path) -> bool:
    """是否为 Objaverse 常见的 glbs/000-000 分桶目录。"""
    if not glbs_dir.is_dir():
        return False
    try:
        for child in glbs_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            # 例如 000-000、127-255
            if len(name) == 7 and name[3] == "-" and name.replace("-", "").isdigit():
                return True
    except OSError:
        return False
    return False


def _flat_output_dir_guard(output_root: Path) -> None:
    """
    扁平模式下禁止写到已含 ``glbs/000-xxx`` 的目录，避免与旧数据混杂。
    请使用专用子目录，例如 ``--output_dir ./data/ObjaverseXL_flat_5k``。
    """
    glbs_child = output_root / "glbs"
    if _looks_like_objaverse_bucket_glbs_dir(glbs_child):
        raise RuntimeError(
            f"扁平输出目录下已存在分桶目录: {glbs_child}\n"
            "这会导致 <sha256>.glb 与 glbs/000-xxx 混在一起。\n"
            "请换用空目录或新子目录，例如:\n"
            f"  --output_dir {output_root / 'ObjaverseXL_flat_subset'}\n"
            "若你确认要保留该 glbs/，可再加 --allow_mixed_output_dir（不推荐）。"
        )


def _infer_data_relpath_from_input(input_dir: str) -> Path:
    """
    若路径中包含 ``raw``，则从 ``raw`` 起保留后缀（如 raw/hf-objaverse-v1）；
    否则使用 raw/<input最后一级目录名>。
    """
    p = Path(input_dir).resolve()
    parts = p.parts
    try:
        i = parts.index("raw")
        return Path(*parts[i:])
    except ValueError:
        return Path("raw") / p.name


def _normalize_file_identifier(s: Any) -> str:
    return str(s).strip().replace("\\", "/").split("/")[-1]


def _norm_to_sha256_map(metadata_df: pd.DataFrame, fid_norm_series: pd.Series) -> Dict[str, str]:
    """每个归一化 file_identifier -> sha256（小写十六进制字符串）。"""
    t = pd.DataFrame(
        {
            "_fid_norm": fid_norm_series,
            "sha256": metadata_df["sha256"].astype(str).str.strip().str.lower(),
        }
    )
    t = t.drop_duplicates(subset=["_fid_norm"], keep="first")
    return dict(zip(t["_fid_norm"].astype(str), t["sha256"]))


def _default_max_workers() -> int:
    n = os.cpu_count() or 8
    return max(8, min(128, n * 4))


def _build_candidates(
    input_dir: Path,
    object_paths: Dict[str, str],
    fid_set: set[str],
    norm_to_sha256: Mapping[str, str],
    max_workers: int,
    stat_batch_size: int,
) -> List[Tuple[str, str, Path, str, str]]:
    """
    返回 [(norm_file_id, rel_path, abs_glb_path, object_paths_key, sha256), ...]，
    仅包含磁盘上存在且在 metadata 中通过 file_identifier 能匹配的行。

    与原先逻辑一致：按 ``object_paths`` 迭代顺序，同一 ``norm`` 在首次 ``is_file`` 为真时收录；
    ``is_file`` 按批并行以加速大量路径探测。
    """
    rows: List[Tuple[str, str, Path, str, str]] = []
    seen_norm: set[str] = set()
    buffer: List[Tuple[str, str, Path, str]] = []

    def flush(executor: ThreadPoolExecutor) -> None:
        if not buffer:
            return
        paths = [b[2] for b in buffer]
        exists = list(executor.map(Path.is_file, paths))
        for (norm, rel_path, abs_glb, op_key), ok in zip(buffer, exists):
            if not ok:
                continue
            if norm in seen_norm:
                continue
            sha = norm_to_sha256.get(norm)
            if not sha:
                continue
            seen_norm.add(norm)
            rows.append((norm, rel_path, abs_glb.resolve(), op_key, sha))
        buffer.clear()

    workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for file_identifier, rel_path in object_paths.items():
            norm = _normalize_file_identifier(file_identifier)
            if norm in seen_norm or norm not in fid_set:
                continue
            abs_glb = input_dir / rel_path
            buffer.append((norm, rel_path, abs_glb, str(file_identifier)))
            if len(buffer) >= stat_batch_size:
                flush(executor)
        flush(executor)

    return rows


def _parallel_copy(
    jobs: Sequence[Tuple[Path, Path]],
    max_workers: int,
    desc: str,
    show_progress: bool,
) -> None:
    """并行复制 (src, dst)，目标父目录在各自任务内创建。"""

    def one(job: Tuple[Path, Path]) -> None:
        src, dst = job
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    workers = max(1, max_workers)
    if len(jobs) == 0:
        return
    if len(jobs) == 1:
        one(jobs[0])
        return

    use_bar = show_progress and tqdm is not None and len(jobs) > 10
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, j) for j in jobs]
        if use_bar:
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="file"):
                fut.result()
        else:
            for fut in futures:
                fut.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="随机抽取 ObjaverseXL 本地 GLB + 对应 metadata / object-paths 子集"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="含 glbs/ 与 object-paths.json 的目录（如 .../raw/hf-objaverse-v1）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出根目录；默认扁平布局下 *.glb / object-paths.json / metadata.csv 均在此目录",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="随机抽取的样本数量（默认 5000）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，便于复现",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="可选：显式指定 metadata.csv；默认按 sdf_voxelize 规则自动查找",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="并行线程数：用于批量 is_file 与复制文件；0 表示自动（约 min(128, cpu×4)）",
    )
    parser.add_argument(
        "--stat_batch_size",
        type=int,
        default=8192,
        help="每批并行探测 is_file 的路径条数（默认 8192，越大内存占用略增）",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="禁用复制阶段的 tqdm 进度条",
    )
    parser.add_argument(
        "--nested",
        action="store_true",
        help="保留 glbs/000-xxx/ 分层目录；默认关闭，即输出为扁平的 <sha256>.glb",
    )
    parser.add_argument(
        "--allow_mixed_output_dir",
        action="store_true",
        help="扁平模式下跳过「output_dir 下已有 glbs/000-xxx」检查（易混杂，一般勿用）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    op_json = input_dir / "object-paths.json"
    if not op_json.is_file():
        raise FileNotFoundError(f"缺少 object-paths.json: {op_json}")

    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv).resolve()
    else:
        metadata_path = _find_metadata_csv(str(input_dir))
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv 不存在: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path, low_memory=False)
    if "file_identifier" not in metadata_df.columns or "sha256" not in metadata_df.columns:
        raise ValueError("metadata.csv 需包含列: file_identifier, sha256")

    fid_norm_series = metadata_df["file_identifier"].map(_normalize_file_identifier)
    fid_set: set[str] = set(fid_norm_series)
    norm_to_sha256 = _norm_to_sha256_map(metadata_df, fid_norm_series)

    max_workers = args.max_workers if args.max_workers > 0 else _default_max_workers()
    stat_batch = max(256, args.stat_batch_size)

    with open(op_json, "r", encoding="utf-8") as f:
        object_paths = json.load(f)

    print(
        f"并行参数: max_workers={max_workers}, stat_batch_size={stat_batch} "
        f"(可用 --max_workers / --stat_batch_size 调整)"
    )

    candidates = _build_candidates(
        input_dir,
        object_paths,
        fid_set,
        norm_to_sha256,
        max_workers,
        stat_batch,
    )
    n_avail = len(candidates)
    if n_avail == 0:
        raise RuntimeError(
            "没有可用样本：请确认 glb 已下载，且 metadata 与 object-paths 的 file_identifier 可对齐。"
        )

    k = min(args.num_samples, n_avail)
    if k < args.num_samples:
        print(f"警告: 仅找到 {n_avail} 个可用样本，少于请求的 {args.num_samples}，将使用全部 {k} 个。")

    if args.seed is not None:
        random.seed(args.seed)

    chosen = random.sample(candidates, k)
    chosen_norm_fids = {t[0] for t in chosen}

    use_nested = args.nested
    if use_nested:
        data_relpath = _infer_data_relpath_from_input(str(input_dir))
        out_data_dir = output_root / data_relpath
    else:
        if not args.allow_mixed_output_dir:
            _flat_output_dir_guard(output_root)
        out_data_dir = output_root
    out_data_dir.mkdir(parents=True, exist_ok=True)

    subset_paths: Dict[str, str] = {}
    copy_jobs: List[Tuple[Path, Path]] = []
    copied_records: List[Dict[str, str]] = []

    for _norm_fid, rel_path, abs_glb, op_key, sha256 in chosen:
        if use_nested:
            dest_rel = rel_path
            dest = out_data_dir / rel_path
        else:
            dest_name = f"{sha256}.glb"
            dest_rel = dest_name
            dest = out_data_dir / dest_name
        subset_paths[op_key] = dest_rel
        copy_jobs.append((abs_glb, dest))
        if not use_nested:
            copied_records.append(
                {
                    "sha256": sha256,
                    "source_relative": rel_path,
                    "dest_filename": dest.name,
                }
            )

    _parallel_copy(
        copy_jobs,
        max_workers=max_workers,
        desc="复制 GLB",
        show_progress=not args.no_progress,
    )

    out_op = out_data_dir / "object-paths.json"
    with open(out_op, "w", encoding="utf-8") as f:
        json.dump(subset_paths, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if copied_records:
        with open(out_data_dir / "copied_files_record.json", "w", encoding="utf-8") as f:
            json.dump(copied_records, f, indent=2, ensure_ascii=False)
            f.write("\n")

    meta_sub = metadata_df[fid_norm_series.isin(chosen_norm_fids)].copy()
    meta_sub = meta_sub.drop_duplicates(subset=["sha256"], keep="first")
    out_meta = output_root / "metadata.csv"
    meta_sub.to_csv(out_meta, index=False)

    manifest = {
        "input_dir": str(input_dir),
        "source_metadata_csv": str(metadata_path),
        "source_object_paths": str(op_json),
        "output_root": str(output_root),
        "output_data_dir": str(out_data_dir),
        "layout": "nested" if use_nested else "flat",
        "num_requested": args.num_samples,
        "num_copied": k,
        "num_available_matched_on_disk": n_avail,
        "seed": args.seed,
        "max_workers": max_workers,
        "stat_batch_size": stat_batch,
    }
    with open(output_root / "sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"已复制 {k} 个 GLB 到: {out_data_dir} （布局: {'nested' if use_nested else 'flat'}）")
    print(f"object-paths.json: {out_op}")
    print(f"metadata.csv: {out_meta}")
    if use_nested:
        print(
            "后续 SDF 处理示例:\n"
            f"  python dataset_toolkits/sdf_voxelize.py --format trellis500k "
            f"--input_dir {out_data_dir} --output_dir <sdf_out> ..."
        )
    else:
        print(
            "扁平布局：所有 .glb 与 object-paths.json、metadata.csv、sample_manifest.json 均在同一目录。\n"
            "请勿把 --output_dir 指到已含 glbs/000-xxx 下载树的项目目录；应使用专用子目录。"
        )


if __name__ == "__main__":
    main()
