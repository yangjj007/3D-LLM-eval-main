"""
CSV reporter: saves one row per sample for easy analysis in pandas/Excel.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List


class CsvReporter:

    @staticmethod
    def save(
        output_dir: str,
        task: str,
        config: Dict[str, Any],
        aggregate_metrics: Dict[str, float],
        per_sample_results: List[Dict[str, Any]],
        filename: str = "eval_results.csv",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        if not per_sample_results:
            print("[Reporter] No results to save.")
            return filepath

        all_keys = set()
        for r in per_sample_results:
            for k, v in r.items():
                if not hasattr(v, "shape"):
                    all_keys.add(k)

        fieldnames = sorted(all_keys)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in per_sample_results:
                row = {k: v for k, v in r.items() if not hasattr(v, "shape")}
                writer.writerow(row)

        # Also save aggregate summary
        summary_path = os.path.join(output_dir, "eval_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in sorted(aggregate_metrics.items()):
                writer.writerow([k, f"{v:.6f}"])

        print(f"[Reporter] CSV results saved to: {filepath}")
        print(f"[Reporter] CSV summary saved to: {summary_path}")
        return filepath
