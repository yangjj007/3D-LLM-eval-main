"""
JSON reporter: saves detailed per-sample results and aggregate metrics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List


class JsonReporter:

    @staticmethod
    def save(
        output_dir: str,
        task: str,
        config: Dict[str, Any],
        aggregate_metrics: Dict[str, float],
        per_sample_results: List[Dict[str, Any]],
        filename: str = "eval_results.json",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        serializable_results = []
        for r in per_sample_results:
            clean = {}
            for k, v in r.items():
                if hasattr(v, "tolist"):
                    continue
                clean[k] = v
            serializable_results.append(clean)

        report = {
            "task": task,
            "model": config.get("model", {}).get("llm_path", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "config": _make_serializable(config),
            "num_samples": len(per_sample_results),
            "aggregate_metrics": aggregate_metrics,
            "per_sample_results": serializable_results,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[Reporter] JSON results saved to: {filepath}")
        return filepath


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return str(obj)
    return obj
