"""Launch all registered baseline evaluations with all visible GPUs.

Examples:
    python -m eval.scripts.run_all_baselines --max_samples 10 --no_resume
    python -m eval.scripts.run_all_baselines --dry_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from eval.baselines.registry import BaselineSpec, enabled_specs
from eval.utils.path_bootstrap import repo_root


def detect_gpu_ids() -> str:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return ",".join(str(i) for i in range(torch.cuda.device_count()))
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        if proc.returncode == 0:
            ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if ids:
                return ",".join(ids)
    except Exception:
        pass
    return "0"


def _selected_specs(args: argparse.Namespace) -> List[BaselineSpec]:
    specs = list(enabled_specs())
    if not args.include_bridged:
        specs = [spec for spec in specs if spec.status == "enabled"]
    if args.only:
        wanted = set(args.only)
        specs = [spec for spec in specs if spec.name in wanted or spec.adapter in wanted]
    if args.exclude:
        excluded = set(args.exclude)
        specs = [spec for spec in specs if spec.name not in excluded and spec.adapter not in excluded]
    return sorted(specs, key=lambda spec: (spec.task, spec.name))


def _command_for_spec(spec: BaselineSpec, args: argparse.Namespace, gpu_ids: str) -> list[str]:
    if not spec.config_path:
        raise ValueError(f"Baseline {spec.name} has no config_path")
    cmd = [
        sys.executable,
        "-m",
        "eval.runner",
        "--config",
        str(repo_root() / spec.config_path),
        "--gpu_ids",
        gpu_ids,
    ]
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.no_resume:
        cmd.append("--no_resume")
    return cmd


def _print_commands(commands: Iterable[list[str]]) -> None:
    import shlex

    for cmd in commands:
        print(shlex.join(cmd))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated GPU ids. Defaults to all visible GPUs.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--include_bridged", action="store_true", default=True)
    parser.add_argument("--direct_only", dest="include_bridged", action="store_false")
    parser.add_argument("--only", nargs="*", default=None, help="Baseline names/adapters to run.")
    parser.add_argument("--exclude", nargs="*", default=None, help="Baseline names/adapters to skip.")
    args = parser.parse_args()

    gpu_ids = args.gpu_ids or detect_gpu_ids()
    specs = _selected_specs(args)
    commands = [_command_for_spec(spec, args, gpu_ids) for spec in specs]

    print(f"[run_all_baselines] repo={repo_root()}")
    print(f"[run_all_baselines] gpu_ids={gpu_ids}")
    print("[run_all_baselines] baselines=" + ", ".join(f"{s.name}:{s.status}" for s in specs))

    if args.dry_run:
        _print_commands(commands)
        return

    for spec, cmd in zip(specs, commands):
        print(f"\n[run_all_baselines] starting {spec.name} ({spec.status})")
        proc = subprocess.run(cmd, cwd=repo_root())
        if proc.returncode != 0:
            msg = f"[run_all_baselines] {spec.name} failed with exit code {proc.returncode}"
            if args.continue_on_error:
                print(msg)
                continue
            raise SystemExit(msg)


if __name__ == "__main__":
    main()
