"""
Multi-GPU data-parallel evaluation via torch.multiprocessing.spawn.
Each rank owns one GPU (via CUDA_VISIBLE_DEVICES) and a disjoint shard of samples.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List

import torch


def _parse_gpu_ids(parallel_cfg: Dict[str, Any]) -> List[int]:
    g = parallel_cfg.get("gpu_ids", [0])
    if isinstance(g, str):
        return [int(x.strip()) for x in g.split(",") if x.strip()]
    if isinstance(g, (list, tuple)):
        return [int(x) for x in g]
    return [int(g)]


def _spawn_fn(
    rank: int,
    world_size: int,
    cfg: Dict[str, Any],
    adapter_name: str,
    task: str,
    samples: List[Dict[str, Any]],
    worker_fn: Callable[..., None],
    gpu_ids: List[int],
) -> None:
    phys = int(gpu_ids[rank % len(gpu_ids)])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    worker_fn(
        rank=rank,
        world_size=world_size,
        device=device,
        cfg=cfg,
        task=task,
        samples=samples,
        adapter_name=adapter_name,
    )


def run_spawned(
    cfg: Dict[str, Any],
    adapter_name: str,
    task: str,
    samples: List[Dict[str, Any]],
    worker_fn: Callable[..., None],
) -> None:
    gpu_ids = _parse_gpu_ids(cfg.get("parallel", {}))
    world_size = len(gpu_ids)
    if (
        world_size <= 1
        or not torch.cuda.is_available()
        or sys.platform == "win32"
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        worker_fn(
            rank=0,
            world_size=1,
            device=device,
            cfg=cfg,
            task=task,
            samples=samples,
            adapter_name=adapter_name,
        )
        return

    import torch.multiprocessing as mp

    args = (world_size, cfg, adapter_name, task, samples, worker_fn, gpu_ids)
    mp.spawn(_spawn_fn, args=args, nprocs=world_size, join=True)
