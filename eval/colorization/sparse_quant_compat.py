"""
Compatibility shims for sparse backends vs newer PyTorch / Trellis code paths.

Some stacks probe ``tensor.is_quantized`` on objects that are not ``torch.Tensor``
(e.g. ``torchsparse.SparseTensor``). Missing that attribute breaks Trellis coloring.
"""

from __future__ import annotations

from typing import Any

_PATCHED = False


def _ensure_is_quantized_property(cls: type) -> bool:
    """Return True if we attached an ``is_quantized`` property."""
    existing = cls.__dict__.get("is_quantized", None)
    if isinstance(existing, property):
        return False

    def _is_quantized(self: Any) -> bool:  # noqa: ANN401
        feats = getattr(self, "feats", None)
        if feats is None:
            feats = getattr(self, "F", None)
        if feats is None:
            feats = getattr(self, "features", None)
        return bool(getattr(feats, "is_quantized", False))

    cls.is_quantized = property(_is_quantized)  # type: ignore[attr-defined]
    return True


def apply_sparse_is_quantized_compat() -> None:
    """Idempotent: patch sparse tensor classes that may lack ``is_quantized``."""
    global _PATCHED
    if _PATCHED:
        return

    patched: list[str] = []

    # 1) torchsparse native SparseTensor (if installed)
    try:
        import torchsparse  # type: ignore import-not-found

        ts_cls = getattr(torchsparse, "SparseTensor", None)
        if isinstance(ts_cls, type) and _ensure_is_quantized_property(ts_cls):
            patched.append("torchsparse.SparseTensor")
    except Exception:
        pass

    # 2) Trellis wrapper: upstream already defines ``is_quantized`` mirroring ``feats``;
    #    do not attach a second property here (would shadow the correct implementation).
    if patched:
        import warnings

        warnings.warn(
            "[eval.colorization] Patched is_quantized on: " + ", ".join(patched),
            stacklevel=2,
        )

    _PATCHED = True
