"""Pluggable model adapters for evaluation."""

from __future__ import annotations

from typing import Dict, Type

from .base import ModelAdapter
from .shapellm_adapter import ShapeLLMAdapter
from .sparse_sdf_adapter import SparseSDFQwen3Adapter
from .text_to_3d_baselines import GaussianCubeAdapter, SAR3DAdapter, ShapEAdapter, TrellisAdapter
from .understanding_baselines import PointLLM13BAdapter, ThreeDLLMAdapter

ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    "shapellm": ShapeLLMAdapter,
    "sparse_sdf_qwen3": SparseSDFQwen3Adapter,
    "sar3d": SAR3DAdapter,
    "trellis": TrellisAdapter,
    "gaussiancube": GaussianCubeAdapter,
    "shape_e": ShapEAdapter,
    "three_d_llm": ThreeDLLMAdapter,
    "pointllm_13b": PointLLM13BAdapter,
}


def get_adapter(name: str) -> ModelAdapter:
    if name not in ADAPTER_REGISTRY:
        raise KeyError(f"Unknown adapter {name!r}; known: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[name]()
