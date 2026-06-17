"""Clone official baseline repositories declared in ``eval.baselines.registry``.

The script never overwrites existing directories. It is intended for setup and
auditing; cloned repositories are ignored by the main repository's git config.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

from eval.baselines.registry import BASELINE_SPECS, BaselineSpec


def _iter_specs(names: list[str], include_skipped: bool) -> Iterable[BaselineSpec]:
    selected = set(names)
    for spec in BASELINE_SPECS.values():
        if selected and spec.name not in selected:
            continue
        if spec.status == "skipped" and not include_skipped:
            continue
        yield spec


def clone_spec(spec: BaselineSpec, *, depth: int | None = 1) -> str:
    target = spec.default_repo_dir
    if target.exists():
        return f"[skip-existing] {spec.name}: {target}"
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if depth is not None and depth > 0:
        cmd.extend(["--depth", str(depth)])
    cmd.extend([spec.repo_url, str(target)])
    subprocess.run(cmd, check=True)
    return f"[cloned] {spec.name}: {target}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Optional baseline names to clone.")
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Also clone audited skipped repos for local inspection.",
    )
    parser.add_argument("--full", action="store_true", help="Clone full history instead of --depth 1.")
    args = parser.parse_args()

    for spec in _iter_specs(args.names, args.include_skipped):
        print(clone_spec(spec, depth=None if args.full else 1), flush=True)


if __name__ == "__main__":
    main()
