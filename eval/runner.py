"""
Evaluation runner: pluggable adapters, multi-GPU, resumable per-sample JSONL + mesh export.

Usage:
    python -m eval.runner --config eval/configs/tasks/understanding.yaml --adapter shapellm --gpu_ids 0
    python -m eval.runner --config eval/configs/tasks/sparse_understanding.yaml --adapter sparse_sdf_qwen3 --gpu_ids 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Project root (3D-LLM-eval-main): repo root first (so ``import eval`` works), then ``third_party``
# (e.g. ``vox2seq``) ahead of the rest of ``sys.path``. Trellis lives in repo-root ``trellis/``.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from eval.utils.path_bootstrap import ensure_third_party_on_path

ensure_third_party_on_path()
os.environ.setdefault("SPCONV_ALGO", "native")


def load_config(config_path: str) -> Dict[str, Any]:
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "default.yaml"
    )
    config: Dict[str, Any] = {}
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    with open(config_path, "r", encoding="utf-8") as f:
        task_config = yaml.safe_load(f) or {}
    return _deep_merge(config, task_config)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.task:
        cfg["task"] = args.task
    if args.output_dir:
        cfg.setdefault("reporting", {})["output_dir"] = args.output_dir
    if args.max_samples is not None:
        cfg.setdefault("data", {})["max_samples"] = args.max_samples
    if args.sample_seed is not None:
        cfg.setdefault("data", {})["sample_seed"] = args.sample_seed
    if args.adapter:
        cfg["adapter"] = args.adapter
    if args.gpu_ids is not None:
        cfg.setdefault("parallel", {})["gpu_ids"] = args.gpu_ids
    if args.batch_size is not None:
        cfg.setdefault("inference", {})["batch_size"] = args.batch_size
    if args.no_resume:
        cfg["resume"] = False


def _output_dir(cfg: Dict[str, Any], adapter: str, task: str) -> str:
    rep = cfg.get("reporting", {})
    base = rep.get("output_dir")
    if base:
        p = str(base)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(_ROOT, p))
        return p
    return os.path.join(_ROOT, "eval_results", adapter, task)


def _expected_sample_ids(samples: List[Dict[str, Any]]) -> Set[str]:
    return {str(s.get("sample_id", "")) for s in samples}


def _dedupe_records_for_sample_ids(
    records: List[Dict[str, Any]], keep_ids: Set[str]
) -> List[Dict[str, Any]]:
    """
    仅保留 sample_id 属于 keep_ids 的行；同一 sample_id 多行时保留最后一行
    （与 JSONL 追加顺序一致）。用于汇总时对齐「当前 YAML 抽样子集」，避免更换
    sample_seed 后历史 per_sample 行仍混入报告与 aggregate。
    """
    last_by_id: Dict[str, Dict[str, Any]] = {}
    for r in records:
        sid = str(r.get("sample_id") or "")
        if sid in keep_ids:
            last_by_id[sid] = r
    return [last_by_id[sid] for sid in sorted(keep_ids) if sid in last_by_id]


def _is_eval_fully_done(out_dir: str, expected: Set[str]) -> Tuple[bool, Set[str], str]:
    """
    返回 (是否完整, 未出现在 JSONL 中的 expected 子集, 原因/状态码).
    完整 = per_sample 已覆盖全部 expected 且存在有效的 aggregate.json 且 num_samples 一致。
    """
    from eval.io.result_store import scan_done_sample_ids

    # 空集在集合论上「vacuously」算完成，但这里表示数据集未加载到任何样本；
    # 若当作已完成会误跳过评估且打印不存在的输出路径。
    if not expected:
        return False, set(), "empty_expected"

    jsonl_probe = os.path.join(out_dir, "per_sample.jsonl")
    done = scan_done_sample_ids(jsonl_probe)
    missing = expected - done
    if missing:
        return False, missing, "incomplete_jsonl"

    agg = os.path.join(out_dir, "aggregate.json")
    if not os.path.isfile(agg):
        return False, set(), "missing_aggregate"

    try:
        with open(agg, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, set(), f"invalid_aggregate:{e!r}"

    n = int(meta.get("num_samples", -1))
    if n != len(expected):
        return False, set(), f"aggregate_count_mismatch:got_{n}_expect_{len(expected)}"

    return True, set(), ""


def _chunks(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _truthy_cfg_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _verbose_eval_from_cfg(cfg: Dict[str, Any]) -> bool:
    """与 sparse adapter 一致：从 ``debug`` / ``inference`` 读取 ``verbose_eval``。"""
    for sec in (cfg.get("debug") or {}, cfg.get("inference") or {}):
        if "verbose_eval" in sec:
            return _truthy_cfg_value(sec.get("verbose_eval"))
    return False


def worker_fn(
    rank: int,
    world_size: int,
    device: Any,
    cfg: Dict[str, Any],
    task: str,
    samples: List[Dict[str, Any]],
    adapter_name: str,
) -> None:
    from eval.adapters import get_adapter
    from eval.adapters.base import MeshInput
    from eval.io.result_store import ResultStore, scan_done_sample_ids
    from eval.metrics.mesh_metrics import MeshMetrics
    from eval.metrics.text_metrics import TextMetrics

    adapter = get_adapter(adapter_name)
    adapter.load(cfg, device)

    adapter_key = adapter.name
    out_dir = _output_dir(cfg, adapter_key, task)
    os.makedirs(out_dir, exist_ok=True)
    store = ResultStore(out_dir, adapter_key, task, rank=rank, world_size=world_size)
    jsonl_done = scan_done_sample_ids(str(store.jsonl_path))

    sorted_samples = sorted(samples, key=lambda x: str(x.get("sample_id", "")))
    my_samples = [s for i, s in enumerate(sorted_samples) if i % world_size == rank]
    todo = [s for s in my_samples if str(s.get("sample_id", "")) not in jsonl_done]
    if cfg.get("resume", True) is False:
        todo = my_samples

    verbose_runner = _verbose_eval_from_cfg(cfg)
    if verbose_runner:
        print(
            f"[runner][debug] worker rank={rank}/{world_size} adapter={adapter_name} "
            f"task={task} todo={len(todo)} my_samples={len(my_samples)} "
            f"already_done={len(jsonl_done)}",
            flush=True,
        )

    bs = max(1, int(cfg.get("inference", {}).get("batch_size", 1)))
    if adapter_name == "shapellm":
        bs = 1

    save_meshes = cfg.get("save_meshes", True)
    metric_names = list(cfg.get("metrics", []))

    for batch in _chunks(todo, bs):
        if verbose_runner:
            sids = [str(s.get("sample_id", "")) for s in batch]
            if task == "generation":
                plens = [len(str(s.get("prompt", "") or "")) for s in batch]
                print(
                    f"[runner][debug] batch rank={rank} task={task} n={len(batch)} "
                    f"sample_ids={sids} prompt_char_lens={plens}",
                    flush=True,
                )
            else:
                print(
                    f"[runner][debug] batch rank={rank} task={task} n={len(batch)} sample_ids={sids}",
                    flush=True,
                )

        mesh_batch = [MeshInput.from_sample_dict(s) for s in batch]

        if task == "understanding":
            rows = adapter.caption_from_shape(mesh_batch, cfg)
            for s, row in zip(batch, rows):
                preds = [row.get("prediction", "")]
                refs = [row.get("ground_truths") or [row.get("ground_truth", "")]]
                text_names = [m for m in metric_names if m in TextMetrics.METRIC_FNS]
                mets = (
                    TextMetrics.compute(preds, refs, text_names, metrics_config=cfg.get("metrics_config", {}))
                    if text_names
                    else {}
                )
                extra_u: Dict[str, Any] = {"raw_response": row.get("raw_response", "")}
                if row.get("debug") is not None:
                    extra_u["debug"] = row["debug"]
                rec = {
                    "sample_id": row.get("sample_id", s.get("sample_id")),
                    "prompt": row.get("prompt", ""),
                    "prediction": row.get("prediction", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "ground_truths": row.get("ground_truths", []),
                    "metrics": mets,
                    "extra": extra_u,
                }
                store.append_record(rec)

        elif task == "vqvae_recon":
            recon_rows = adapter.reconstruct_mesh(mesh_batch, cfg)
            mesh_names = [m for m in metric_names if m in ("chamfer_distance", "hausdorff_distance", "f_score")]
            metric_points = int(cfg.get("metrics_config", {}).get("num_sample_points", 8192))
            for s, row in zip(batch, recon_rows):
                pred = row.get("pred_mesh")
                gt = row.get("gt_mesh")
                mets: Dict[str, float] = {}
                if pred is not None and gt is not None and mesh_names:
                    print(
                        f"[metrics] sample={row.get('sample_id', s.get('sample_id'))} "
                        f"开始计算 {mesh_names}，num_sample_points={metric_points}",
                        flush=True,
                    )
                    agg, per = MeshMetrics.compute_reconstruction(
                        [{"pred_mesh": pred, "gt_mesh": gt}],
                        mesh_names,
                        num_sample_points=metric_points,
                    )
                    mets = per[0] if per else agg
                rel_mesh = None
                if save_meshes and pred is not None:
                    rel_mesh = store.save_mesh_obj(
                        str(row.get("sample_id", s.get("sample_id"))), pred
                    )
                rec = {
                    "sample_id": row.get("sample_id", s.get("sample_id")),
                    "mesh_path": row.get("mesh_path", s.get("mesh_path")),
                    "metrics": mets,
                    "num_tokens": row.get("num_tokens", 0),
                    "mesh_rel_path": rel_mesh,
                    "extra": dict(row.get("extra") or {}),
                }
                store.append_record(rec)

        elif task == "generation":
            prompts = [s["prompt"] for s in batch]
            sids = [str(s.get("sample_id", "")) for s in batch]
            gens = adapter.generate_from_text(prompts, sids, cfg)
            for s, g in zip(batch, gens):
                rel_mesh = None
                rel_glb = None
                if save_meshes and g.pred_mesh is not None:
                    rel_mesh = store.save_mesh_obj(str(s.get("sample_id")), g.pred_mesh)
                if save_meshes and getattr(g, "extra", None) and g.extra.get("glb_trimesh") is not None:
                    try:
                        rel_glb = store.save_glb(str(s.get("sample_id")), g.extra["glb_trimesh"])
                    except Exception as exc:
                        if isinstance(g.extra, dict):
                            g.extra["glb_save_error"] = repr(exc)
                extra = dict(g.extra) if getattr(g, "extra", None) else {}
                if extra.pop("glb_trimesh", None) is not None:
                    print(
                        f"[runner][debug] saved textured GLB for sample_id={s.get('sample_id')} "
                        f"glb_rel_path={rel_glb}; removed in-memory glb_trimesh from JSON record",
                        flush=True,
                    )
                rec = {
                    "sample_id": s.get("sample_id"),
                    "caption": g.extra.get("caption", s.get("prompt", "")) if getattr(g, "extra", None) else s.get("prompt", ""),
                    "prompt": g.extra.get("prompt", s.get("prompt", "")) if getattr(g, "extra", None) else s.get("prompt", ""),
                    "raw_response": g.raw_response,
                    "num_tokens_generated": len(g.mesh_token_ids),
                    "num_occupied_voxels": g.num_occupied_voxels,
                    "mesh_rel_path": rel_mesh,
                    "glb_rel_path": rel_glb,
                    "reference_mesh_path": s.get("reference_mesh_path"),
                    "metrics": {},
                    "extra": extra,
                }
                mesh_names = [m for m in metric_names if m in ("chamfer_distance", "emd", "f_score")]
                if mesh_names and s.get("reference_mesh_path") and rel_mesh:
                    from eval.metrics.mesh_metrics import (
                        _sample_points_from_mesh,
                        chamfer_distance,
                    )

                    refp = s["reference_mesh_path"]
                    rp = _sample_points_from_mesh(refp)
                    pp = _sample_points_from_mesh(os.path.join(out_dir, rel_mesh))
                    rec["metrics"] = {"chamfer_distance": chamfer_distance(pp, rp)}
                store.append_record(rec)

        elif task == "sparse_mesh":
            mesh_names = [m for m in metric_names if m in ("chamfer_distance", "hausdorff_distance", "f_score")]
            metric_points = int(cfg.get("metrics_config", {}).get("num_sample_points", 8192))
            for s in batch:
                sid = str(s.get("sample_id", ""))
                prompt = str(s.get("prompt") or "").strip()
                if prompt:
                    if s.get("mesh_path") and hasattr(adapter, "generate_from_mesh_context"):
                        g = adapter.generate_from_mesh_context([MeshInput.from_sample_dict(s)], cfg)[0]
                    else:
                        g = adapter.generate_from_text([prompt], [sid], cfg)[0]
                    rel_mesh = None
                    if save_meshes and g.pred_mesh is not None:
                        rel_mesh = store.save_mesh_obj(sid, g.pred_mesh)
                    rel_glb = None
                    if save_meshes and getattr(g, "extra", None) and g.extra.get("glb_trimesh") is not None:
                        try:
                            rel_glb = store.save_glb(sid, g.extra["glb_trimesh"])
                        except Exception as exc:
                            if isinstance(g.extra, dict):
                                g.extra["glb_save_error"] = repr(exc)
                    extra = dict(g.extra) if getattr(g, "extra", None) else {}
                    if extra.pop("glb_trimesh", None) is not None:
                        print(
                            f"[runner][debug] saved textured GLB for sample_id={sid} "
                            f"glb_rel_path={rel_glb}; removed in-memory glb_trimesh from JSON record",
                            flush=True,
                        )
                    mets: Dict[str, float] = {}
                    refp = s.get("reference_mesh_path") or s.get("mesh_path")
                    if g.pred_mesh is not None and refp and mesh_names:
                        import trimesh

                        gt = trimesh.load(refp, force="mesh")
                        if not isinstance(gt, trimesh.Trimesh):
                            gt = list(gt.geometry.values())[0]
                        agg, per = MeshMetrics.compute_reconstruction(
                            [{"pred_mesh": g.pred_mesh, "gt_mesh": gt}],
                            [m for m in mesh_names if m in ("chamfer_distance", "hausdorff_distance", "f_score")],
                            num_sample_points=metric_points,
                        )
                        mets = per[0] if per else agg
                    rec = {
                        "sample_id": sid,
                        "caption": g.extra.get("caption", prompt) if getattr(g, "extra", None) else prompt,
                        "prompt": g.extra.get("prompt", prompt) if getattr(g, "extra", None) else prompt,
                        "raw_response": g.raw_response,
                        "num_tokens_generated": len(g.mesh_token_ids),
                        "num_occupied_voxels": g.num_occupied_voxels,
                        "mesh_rel_path": rel_mesh,
                        "glb_rel_path": rel_glb,
                        "reference_mesh_path": s.get("reference_mesh_path") or s.get("mesh_path"),
                        "metrics": mets,
                        "extra": extra,
                    }
                    store.append_record(rec)
                else:
                    row = adapter.reconstruct_mesh([MeshInput.from_sample_dict(s)], cfg)[0]
                    pred = row.get("pred_mesh")
                    gt = row.get("gt_mesh")
                    mets = {}
                    if pred is not None and gt is not None and mesh_names:
                        agg, per = MeshMetrics.compute_reconstruction(
                            [{"pred_mesh": pred, "gt_mesh": gt}],
                            mesh_names,
                            num_sample_points=metric_points,
                        )
                        mets = per[0] if per else agg
                    rel_mesh = None
                    if save_meshes and pred is not None:
                        rel_mesh = store.save_mesh_obj(sid, pred)
                    store.append_record(
                        {
                            "sample_id": sid,
                            "mesh_path": row.get("mesh_path", s.get("mesh_path")),
                            "metrics": mets,
                            "num_tokens": row.get("num_tokens", 0),
                            "mesh_rel_path": rel_mesh,
                            "extra": dict(row.get("extra") or {}),
                        }
                    )

        else:
            raise ValueError(f"Unknown task: {task}")

    store.close()


def _merge_and_aggregate(
    cfg: Dict[str, Any],
    task: str,
    adapter_name: str,
    *,
    expected_sample_ids: Optional[Set[str]] = None,
) -> None:
    from eval.io.result_store import merge_rank_jsonls
    from eval.metrics.mesh_metrics import MeshMetrics
    from eval.metrics.render_metrics import compute_aggregate_generation_render_metrics
    from eval.metrics.text_metrics import TextMetrics
    from eval.reporting import REPORTER_REGISTRY

    out_dir = _output_dir(cfg, adapter_name, task)
    par = cfg.get("parallel", {})
    gpus = par.get("gpu_ids", [0])
    if isinstance(gpus, str):
        gpus = [int(x.strip()) for x in gpus.split(",") if x.strip()]
    world_size = max(1, len(gpus))
    if world_size > 1:
        merge_rank_jsonls(out_dir, world_size)

    final_jsonl = os.path.join(out_dir, "per_sample.jsonl")
    if not os.path.isfile(final_jsonl):
        # single rank wrote per_sample.jsonl directly
        pass

    records: List[Dict[str, Any]] = []
    if os.path.isfile(final_jsonl):
        with open(final_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if expected_sample_ids:
        n_before = len(records)
        records = _dedupe_records_for_sample_ids(records, expected_sample_ids)
        if n_before != len(records):
            print(
                f"[runner] 汇总 per_sample：按当前配置保留 {len(records)}/{n_before} 条"
                f"（sample_id ∈ 当前数据集，共 {len(expected_sample_ids)} 个期望 id）",
                flush=True,
            )

    aggregate: Dict[str, float] = {}
    metric_names = list(cfg.get("metrics", []))

    if task == "understanding" and records:
        preds = [r.get("prediction", "") for r in records]
        refs = [r.get("ground_truths") or [r.get("ground_truth", "")] for r in records]
        text_names = [m for m in metric_names if m in TextMetrics.METRIC_FNS]
        aggregate.update(TextMetrics.compute(preds, refs, text_names, metrics_config=cfg.get("metrics_config", {})))

    if task == "vqvae_recon" and records:
        mesh_names = [m for m in metric_names if m in ("chamfer_distance", "hausdorff_distance", "f_score")]
        if mesh_names:
            # re-load meshes from disk for aggregate if metrics missing
            recon_rows = []
            for r in records:
                rel = r.get("mesh_rel_path")
                mp = os.path.join(out_dir, rel) if rel else None
                if mp and os.path.isfile(mp):
                    import trimesh

                    pred = trimesh.load(mp, force="mesh")
                    gtp = r.get("mesh_path")
                    if gtp and os.path.isfile(gtp):
                        gt = trimesh.load(gtp, force="mesh")
                        if not isinstance(gt, trimesh.Trimesh):
                            gt = list(gt.geometry.values())[0]
                        recon_rows.append({"pred_mesh": pred, "gt_mesh": gt})
            if recon_rows:
                agg, _ = MeshMetrics.compute_reconstruction(recon_rows, mesh_names)
                aggregate.update(agg)

    if task in ("generation", "sparse_mesh") and records:
        n_tok = sum(1 for r in records if r.get("num_tokens_generated", 0) > 0)
        aggregate["generation_success_rate"] = n_tok / len(records) if records else 0.0
        aggregate["avg_occupied_voxels"] = (
            sum(r.get("num_occupied_voxels", 0) for r in records) / len(records)
            if records
            else 0.0
        )
        render_agg = compute_aggregate_generation_render_metrics(records, out_dir, cfg, metric_names)
        aggregate.update(render_agg)

    data_cfg = cfg.get("data") or {}
    meta: Dict[str, Any] = {
        "task": task,
        "adapter": adapter_name,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(records),
        "aggregate_metrics": aggregate,
        "data_slice": {
            "sample_seed": data_cfg.get("sample_seed"),
            "max_samples": data_cfg.get("max_samples"),
            "metadata_csv": data_cfg.get("metadata_csv"),
            "gen_caption_indices": data_cfg.get("gen_caption_indices"),
        },
    }
    with open(os.path.join(out_dir, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    reporting_cfg = cfg.get("reporting", {})
    formats = reporting_cfg.get("formats", ["json", "csv"])
    for fmt in formats:
        reporter_cls = REPORTER_REGISTRY.get(fmt)
        if reporter_cls is None:
            continue
        reporter_cls.save(
            output_dir=out_dir,
            task=task,
            config=cfg,
            aggregate_metrics=aggregate,
            per_sample_results=records,
        )

    print("\n" + "=" * 60)
    print("  AGGREGATE METRICS")
    print("=" * 60)
    for k, v in sorted(aggregate.items()):
        print(f"  {k:30s} : {float(v):.6f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D-LLM Evaluation Runner")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="数据子集打乱种子（写入 data.sample_seed）；与 max_samples 配合可复现随机抽样",
    )
    parser.add_argument("--gpu_ids", type=str, default=None, help="e.g. 0,1,2,3")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _apply_cli(cfg, args)

    task = cfg.get("task")
    if not task:
        print("[Error] task missing in config")
        sys.exit(1)

    adapter_name = cfg.get("adapter", "shapellm")

    from eval.data import DATASET_REGISTRY

    if task not in DATASET_REGISTRY:
        print(f"[Error] Unknown task dataset: {task}")
        sys.exit(1)
    ds = DATASET_REGISTRY[task](cfg.get("data", {}))
    samples = [ds[i] for i in range(len(ds))]

    _dc = cfg.get("data") or {}
    print(
        f"[runner] 数据子集: sample_seed={_dc.get('sample_seed')!r} max_samples={_dc.get('max_samples')} "
        f"→ 本运行 {len(samples)} 条样本（与 per_sample 汇总范围一致）",
        flush=True,
    )

    from eval.adapters import get_adapter
    from eval.parallel import run_spawned

    if len(samples) == 0:
        print(
            "[Error] 当前数据配置下评估样本数为 0（dataset 为空）。\n"
            "  常见原因：glb_dir 下缺少与 metadata 对应的 ``{sha256}.glb`` 或 ``{file_identifier}.glb``，\n"
            "  或缺少可用文本列（``captions`` / ``caption`` / ``text`` 等）导致无生成样本。\n"
            "  请检查 YAML 中 data.metadata_csv、data.glb_dir、data.gen_caption_indices 等。"
        )
        sys.exit(1)

    if cfg.get("resume", True):
        adapter = get_adapter(adapter_name)
        out_dir = _output_dir(cfg, adapter.name, task)
        expected = _expected_sample_ids(samples)
        done_ok, missing_ids, done_reason = _is_eval_fully_done(out_dir, expected)
        resume_files_missing = False
        if done_ok:
            abspath = os.path.abspath(out_dir)
            jpath = os.path.join(abspath, "per_sample.jsonl")
            apath = os.path.join(abspath, "aggregate.json")
            has_agg = os.path.isfile(apath)

            has_jsonl = os.path.isfile(jpath)
            has_shards = any(Path(abspath).glob("per_sample.rank*.jsonl"))
            if not has_agg or not (has_jsonl or has_shards):
                print(
                    "\n[Resume] 内部判定为已完成，但结果文件在磁盘上不存在或不可读，将重新运行评估。\n"
                    f"  输出目录: {abspath}\n"
                    f"  aggregate.json: {has_agg}, per_sample.jsonl: {has_jsonl}, "
                    f"rank 分片: {has_shards}\n"
                )
                done_ok = False
                resume_files_missing = True
        if done_ok:
            print(
                "\n" + "=" * 60
                + "\n  检测到已有完整评估结果（per_sample 全覆盖且 aggregate 有效），"
                "已跳过重新运行评估。\n"
                "  若需强制重算请使用: --no_resume\n" + "=" * 60
            )
            print(f"  输出目录:\n    {abspath}\n")
            print("  主要结果文件路径:")
            print(f"    - {jpath}")
            print(f"    - {apath}")
            for name in (
                "eval_results.json",
                "eval_results.csv",
                "eval_summary.csv",
                "eval_table.tex",
            ):
                p = os.path.join(abspath, name)
                if os.path.isfile(p):
                    print(f"    - {p}")
            print("=" * 60 + "\n")
            sys.exit(0)
        if missing_ids:
            preview = sorted(missing_ids)[:8]
            more = f" …（共 {len(missing_ids)} 个）" if len(missing_ids) > 8 else ""
            print(
                f"\n[Resume] 结果未完整，将从断点续跑。"
                f" 尚未写入的 sample_id（部分）: {preview}{more}\n"
                f"  输出目录: {os.path.abspath(out_dir)}\n"
            )
        elif resume_files_missing:
            pass  # 已打印原因，继续全量跑
        else:
            if done_reason == "missing_aggregate":
                msg = "分样本已齐但缺少 aggregate.json（将补全合并与报告）"
            else:
                msg = f"结果不完整或汇总与当前配置不一致（{done_reason}）"
            print(
                f"\n[Resume] {msg}，将续跑/补全："
                f"\n  输出目录: {os.path.abspath(out_dir)}\n"
            )

    t0 = time.time()
    run_spawned(cfg, adapter_name, task, samples, worker_fn)
    _merge_and_aggregate(
        cfg, task, adapter_name, expected_sample_ids=_expected_sample_ids(samples)
    )
    print(f"\n[Done] Evaluation completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
