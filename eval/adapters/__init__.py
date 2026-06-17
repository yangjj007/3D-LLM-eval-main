"""Pluggable model adapters for evaluation."""

from __future__ import annotations

from typing import Dict, Type

from .base import ModelAdapter
from .shapellm_adapter import ShapeLLMAdapter
from .sparse_sdf_adapter import SparseSDFQwen3Adapter

ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    "shapellm": ShapeLLMAdapter,
    "sparse_sdf_qwen3": SparseSDFQwen3Adapter,
}


def get_adapter(name: str) -> ModelAdapter:
    if name not in ADAPTER_REGISTRY:
        raise KeyError(f"Unknown adapter {name!r}; known: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[name]()
