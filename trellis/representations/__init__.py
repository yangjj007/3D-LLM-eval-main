from .radiance_field import Strivec
from .octree import DfsOctree as Octree

__all__ = ["Strivec", "Octree", "Gaussian", "MeshExtractResult"]


def __getattr__(name: str):
    """
    Lazy imports so ``import trellis`` does not require optional deps
    (``plyfile`` for Gaussian, ``kaolin`` / flexicubes for mesh extraction).
    """
    if name == "Gaussian":
        from .gaussian import Gaussian as _Gaussian

        globals()["Gaussian"] = _Gaussian
        return _Gaussian
    if name == "MeshExtractResult":
        from .mesh import MeshExtractResult as _MeshExtractResult

        globals()["MeshExtractResult"] = _MeshExtractResult
        return _MeshExtractResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
