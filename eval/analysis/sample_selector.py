#!/usr/bin/env python3
"""
Select / rank samples from per_sample.jsonl by metric vs optional baseline.

Examples:
  python -m eval.analysis.sample_selector \\
    --ours eval_results/sparse_sdf_qwen3/understanding/per_sample.jsonl \\
    --baseline eval_results/shapellm/understanding/per_sample.jsonl \\
    --metric bleu_1 --higher_is_better --min_gap 0.05 --top_k 50 --out picked.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            sid = str(o.get("sample_id", ""))
            if sid:
                out[sid] = o
    return out


def get_metric(rec: Dict[str, Any], name: str) -> Optional[float]:
    m = rec.get("metrics") or {}
    if name in m:
        v = m[name]
        return float(v) if v is not None else None
    return None


def select_samples(
    ours_path: str,
    baseline_path: Optional[str],
    metric: str,
    higher_is_better: bool,
    min_gap: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    ours = load_jsonl(ours_path)
    base = load_jsonl(baseline_path) if baseline_path and Path(baseline_path).is_file() else None

    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for sid, rec in ours.items():
        vo = get_metric(rec, metric)
        if vo is None:
            continue
        if base is not None:
            if sid not in base:
                continue
            vb = get_metric(base[sid], metric)
            if vb is None:
                continue
            gap = (vo - vb) if higher_is_better else (vb - vo)
            if gap < min_gap:
                continue
        key = vo if higher_is_better else -vo
        row = dict(rec)
        row["_sort_key"] = key
        scored.append((key, sid, row))

    scored.sort(key=lambda x: -x[0])
    return [x[2] for x in scored[:top_k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=str, required=True)
    ap.add_argument("--baseline", type=str, default=None)
    ap.add_argument("--metric", type=str, required=True)
    ap.add_argument("--higher_is_better", action="store_true", default=True)
    ap.add_argument("--lower_is_better", action="store_true")
    ap.add_argument("--min_gap", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    high = not args.lower_is_better
    rows = select_samples(
        args.ours,
        args.baseline,
        args.metric,
        high,
        args.min_gap,
        args.top_k,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            r.pop("_sort_key", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sample_selector] Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
