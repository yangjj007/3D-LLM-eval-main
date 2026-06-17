"""Run LGM text-to-3D through the official Gradio app process function.

The official LGM text path is implemented in ``app.py`` but the file launches a
Gradio UI at import time. This helper executes the official file only up to the
``# gradio UI`` marker, then calls the official ``process(input_image=None, ...)``
function and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--model-size", default="big")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--elevation", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    app_path = repo_dir / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(app_path)

    sys.path.insert(0, str(repo_dir))
    old_argv = sys.argv[:]
    sys.argv = [
        str(app_path),
        args.model_size,
        "--resume",
        args.resume,
        "--workspace",
        args.workspace,
    ]
    try:
        source = app_path.read_text(encoding="utf-8")
        prefix = source.split("# gradio UI", 1)[0]
        ns = {"__name__": "__lgm_official_app_prefix__", "__file__": str(app_path)}
        exec(compile(prefix, str(app_path), "exec"), ns)
        process = ns["process"]
        _, video_path, ply_path = process(
            input_image=None,
            prompt=args.prompt,
            prompt_neg=args.negative_prompt,
            input_elevation=args.elevation,
            input_num_steps=args.num_steps,
            input_seed=args.seed,
        )
        print(f"video={video_path}")
        print(f"ply={ply_path}")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
