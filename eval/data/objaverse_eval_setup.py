"""
Objaverse / TRELLIS-500K helpers for the evaluation repo.

Subcommands wrap ``dataset_toolkits/download.py`` and
``dataset_toolkits/sample_objaverse_glb_subset.py``, and optionally
``eval.data.build_eval_from_metadata``.

Run from the **3D-LLM-eval-main** repository root (same as ``python -m eval.runner``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> str:
    # eval/data/objaverse_eval_setup.py -> eval -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _dataset_toolkits_dir() -> str:
    return os.path.join(_repo_root(), "dataset_toolkits")


def cmd_fetch_metadata(args: argparse.Namespace) -> None:
    dt = _dataset_toolkits_dir()
    if dt not in sys.path:
        sys.path.insert(0, dt)
    from datasets.ObjaverseXL import get_metadata  # type: ignore

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = get_metadata(args.source)
    dest = out / "metadata.csv"
    meta.to_csv(dest, index=False)
    print(f"[fetch-metadata] Wrote {dest} ({len(meta)} rows)")


def cmd_download_glb(args: argparse.Namespace) -> None:
    dt = _dataset_toolkits_dir()
    cmd = [
        sys.executable,
        os.path.join(dt, "download.py"),
        "ObjaverseXL",
        "--output_dir",
        args.output_dir,
    ]
    if args.source:
        cmd.extend(["--source", args.source])
    if args.filter_low_aesthetic_score is not None:
        cmd.extend(["--filter_low_aesthetic_score", str(args.filter_low_aesthetic_score)])
    if args.instances:
        cmd.extend(["--instances", args.instances])
    if args.rank is not None:
        cmd.extend(["--rank", str(args.rank)])
    if args.world_size is not None:
        cmd.extend(["--world_size", str(args.world_size)])
    print("[download-glb]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=dt)
    if r.returncode != 0:
        sys.exit(r.returncode)


def cmd_sample_subset(args: argparse.Namespace) -> None:
    dt = _dataset_toolkits_dir()
    script = os.path.join(dt, "sample_objaverse_glb_subset.py")
    if not os.path.isfile(script):
        print(f"[sample-subset] Missing {script}", file=sys.stderr)
        sys.exit(1)
    cmd = [
        sys.executable,
        script,
        "--input_dir",
        args.input_dir,
        "--output_dir",
        args.output_dir,
        "--num_samples",
        str(args.num_samples),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.max_workers:
        cmd.extend(["--max_workers", str(args.max_workers)])
    if args.nested:
        cmd.append("--nested")
    if args.metadata_csv:
        cmd.extend(["--metadata_csv", args.metadata_csv])
    if args.allow_mixed_output_dir:
        cmd.append("--allow_mixed_output_dir")
    if args.no_progress:
        cmd.append("--no_progress")
    print("[sample-subset]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=dt)
    if r.returncode != 0:
        sys.exit(r.returncode)


def cmd_build_eval_json(args: argparse.Namespace) -> None:
    root = _repo_root()
    cmd = [
        sys.executable,
        "-m",
        "eval.data.build_eval_from_metadata",
        "--metadata_csv",
        args.metadata_csv,
        "--glb_dir",
        args.glb_dir,
        "--output_dir",
        args.output_dir,
        "--gen_caption_indices",
        args.gen_caption_indices,
    ]
    if getattr(args, "default_prompt", None):
        cmd.extend(["--default_prompt", args.default_prompt])
    print("[build-eval-json]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=root)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_meta = sub.add_parser("fetch-metadata", help="Download ObjaverseXL metadata CSV (TRELLIS-500K listing)")
    p_meta.add_argument("--output_dir", type=str, required=True)
    p_meta.add_argument("--source", type=str, default="sketchfab", choices=("sketchfab", "github"))
    p_meta.set_defaults(func=cmd_fetch_metadata)

    p_dl = sub.add_parser(
        "download-glb",
        help="Run dataset_toolkits/download.py ObjaverseXL (requires metadata.csv in output_dir)",
    )
    p_dl.add_argument("--output_dir", type=str, required=True)
    p_dl.add_argument("--source", type=str, default=None, help="Passed to ObjaverseXL (sketchfab/github)")
    p_dl.add_argument("--filter_low_aesthetic_score", type=float, default=None)
    p_dl.add_argument("--instances", type=str, default=None)
    p_dl.add_argument("--rank", type=int, default=None)
    p_dl.add_argument("--world_size", type=int, default=None)
    p_dl.set_defaults(func=cmd_download_glb)

    p_samp = sub.add_parser(
        "sample-subset",
        help="Random copy of existing local GLBs + metadata subset (see sample_objaverse_glb_subset.py)",
    )
    p_samp.add_argument("--input_dir", type=str, required=True)
    p_samp.add_argument("--output_dir", type=str, required=True)
    p_samp.add_argument("--num_samples", type=int, default=5000)
    p_samp.add_argument("--seed", type=int, default=None)
    p_samp.add_argument("--max_workers", type=int, default=0)
    p_samp.add_argument("--nested", action="store_true")
    p_samp.add_argument("--metadata_csv", type=str, default=None)
    p_samp.add_argument("--allow_mixed_output_dir", action="store_true")
    p_samp.add_argument("--no_progress", action="store_true")
    p_samp.set_defaults(func=cmd_sample_subset)

    p_be = sub.add_parser("build-eval-json", help="Write understanding.json + generation.json from CSV + glb dir")
    p_be.add_argument("--metadata_csv", type=str, required=True)
    p_be.add_argument("--glb_dir", type=str, required=True)
    p_be.add_argument("--output_dir", type=str, default="eval_data")
    p_be.add_argument("--gen_caption_indices", type=str, default="0")
    p_be.add_argument("--default_prompt", type=str, default="")
    p_be.set_defaults(func=cmd_build_eval_json)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
