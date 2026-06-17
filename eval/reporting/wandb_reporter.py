"""
Weights & Biases reporter (optional integration).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List


class WandbReporter:

    @staticmethod
    def save(
        output_dir: str,
        task: str,
        config: Dict[str, Any],
        aggregate_metrics: Dict[str, float],
        per_sample_results: List[Dict[str, Any]],
        **kwargs,
    ) -> None:
        wandb_cfg = config.get("reporting", {}).get("wandb", {})
        if not wandb_cfg.get("enabled", False):
            return

        try:
            import wandb
        except ImportError:
            warnings.warn(
                "wandb not installed. Install with: pip install wandb"
            )
            return

        project = wandb_cfg.get("project", "shapellm-omni-eval")
        run_name = wandb_cfg.get("run_name", f"eval_{task}")
        entity = wandb_cfg.get("entity", None)

        run = wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            config=config,
            reinit=True,
        )

        wandb.log(aggregate_metrics)

        # Log per-sample results as a table
        columns = []
        if per_sample_results:
            columns = [
                k for k in per_sample_results[0].keys() if not hasattr(per_sample_results[0][k], "shape")
            ]
        if columns:
            table = wandb.Table(columns=columns)
            for r in per_sample_results:
                row = [r.get(k, "") for k in columns if not hasattr(r.get(k, ""), "shape")]
                if len(row) == len(columns):
                    table.add_data(*row)
            wandb.log({"per_sample_results": table})

        run.finish()
        print(f"[Reporter] Results logged to WandB project: {project}")
