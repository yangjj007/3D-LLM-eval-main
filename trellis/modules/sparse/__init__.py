from typing import *

BACKEND = 'torchsparse'
# DEBUG = False
DEBUG = True
ATTN = 'flash_attn'

def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    global ATTN
    
    env_sparse_backend = os.environ.get('SPARSE_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn is None:
        env_sparse_attn = os.environ.get('ATTN_BACKEND')

    if env_sparse_backend is not None and env_sparse_backend in ['spconv', 'torchsparse']:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn is not None and env_sparse_attn in ['xformers', 'flash_attn']:
        ATTN = env_sparse_attn
        
    print(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")
        

__from_env()


def set_sparse_backend(backend: Literal["spconv", "torchsparse"]) -> None:
    """
    Switch sparse conv implementation at runtime (updates env, ``BACKEND``, reloads ``.conv``).

    Trellis official weights expect **spconv**; Med / eval **SparseSDFVQVAE** checkpoints
    expect **torchsparse**. Call ``set_sparse_backend('torchsparse')`` before building the VAE,
    then ``set_sparse_backend('spconv')`` before ``Trellis*Pipeline.from_pretrained`` (or use
    ``eval.model_loader.load_trellis_*`` which snapshots/restores automatically).

    Implementation drops ``sys.modules`` entries under ``...sparse.conv`` so a prior
    ``torchsparse`` load cannot survive as stale ``SparseConv3d`` (``importlib.reload`` alone
    is not always enough when pipelines were imported mid-switch).

    It also resets ``trellis.modules.sparse.basic.SparseTensorData`` (lazy backend binding).
    Without this, the first import may cache ``torchsparse.SparseTensor`` forever, and later
    ``BACKEND='spconv'`` will still build torchsparse internals while ``conv_spconv`` expects
    ``spconv.SparseConvTensor`` (``features`` vs ``F`` mismatch).
    """
    import importlib
    import os
    import sys

    global BACKEND
    if backend not in ("spconv", "torchsparse"):
        raise ValueError(f"backend must be 'spconv' or 'torchsparse', got {backend!r}")
    os.environ["SPARSE_BACKEND"] = backend
    BACKEND = backend
    for _name in ("SparseConv3d", "SparseInverseConv3d", "sparseconv3d_func"):
        globals().pop(_name, None)
    # Drop lazily-cached symbols that bind to the old backend implementation.
    for _name in (
        "SparseTensor",
        "sparse_batch_broadcast",
        "sparse_batch_op",
        "sparse_cat",
        "sparse_unbind",
    ):
        globals().pop(_name, None)

    # Reset lazy backend class cached inside ``basic.SparseTensor`` (module-global).
    basic_pkg = __name__ + ".basic"
    basic_mod = sys.modules.get(basic_pkg)
    if basic_mod is not None:
        try:
            setattr(basic_mod, "SparseTensorData", None)
        except Exception:
            pass
    for key in list(sys.modules):
        if key == basic_pkg or key.startswith(basic_pkg + "."):
            del sys.modules[key]

    conv_pkg = __name__ + ".conv"
    for key in list(sys.modules):
        if key == conv_pkg or key.startswith(conv_pkg + "."):
            del sys.modules[key]

    try:
        importlib.import_module(conv_pkg)
    except ImportError as exc:
        if backend == "spconv":
            raise RuntimeError(
                "Trellis 官方权重需要 spconv（SubMConv 的 state_dict 为 *.conv.weight）。"
                "请安装与 CUDA 匹配的 spconv（例如 spconv-cu118 / spconv-cu120）。"
                f"原始错误: {exc!r}"
            ) from exc
        raise

    conv_mod = sys.modules.get(conv_pkg)
    sc = getattr(conv_mod, "SparseConv3d", None) if conv_mod is not None else None
    if sc is None:
        raise RuntimeError(f"切换 SPARSE 后端为 {backend!r} 后未能加载 SparseConv3d（{conv_pkg}）。")
    impl = getattr(sc, "__module__", "")
    dbg = os.environ.get("EVAL_TRELLIS_SPARSE_DEBUG", "").strip().lower() in ("1", "true", "yes")
    if backend == "spconv":
        print(
            f"[SPARSE] Backend=spconv, Attention={ATTN} | SparseConv3d -> {impl!r}",
            flush=True,
        )
    elif dbg:
        print(
            f"[SPARSE] Backend={BACKEND}, Attention={ATTN} | SparseConv3d -> {impl!r}",
            flush=True,
        )
    if backend == "spconv" and impl and not impl.endswith("conv_spconv"):
        raise RuntimeError(
            f"期望 spconv 后端但 SparseConv3d 来自 {impl!r}。"
            "请确认已安装 spconv，且未在 set_sparse_backend('spconv') 之前 import Trellis pipeline。"
        )


def set_backend(backend: Literal["spconv", "torchsparse"]):
    set_sparse_backend(backend)


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn(attn: Literal['xformers', 'flash_attn']):
    global ATTN
    ATTN = attn
    
    
import importlib

__attributes = {
    'SparseTensor': 'basic',
    'sparse_batch_broadcast': 'basic',
    'sparse_batch_op': 'basic',
    'sparse_cat': 'basic',
    'sparse_unbind': 'basic',
    'SparseGroupNorm': 'norm',
    'SparseLayerNorm': 'norm',
    'SparseGroupNorm32': 'norm',
    'SparseLayerNorm32': 'norm',
    'SparseSigmoid': 'nonlinearity',
    'SparseReLU': 'nonlinearity',
    'SparseSiLU': 'nonlinearity',
    'SparseGELU': 'nonlinearity',
    'SparseTanh': 'nonlinearity',
    'SparseActivation': 'nonlinearity',
    'SparseLinear': 'linear',
    'sparse_scaled_dot_product_attention': 'attention',
    'SerializeMode': 'attention',
    'sparse_serialized_scaled_dot_product_self_attention': 'attention',
    'sparse_windowed_scaled_dot_product_self_attention': 'attention',
    'SparseMultiHeadAttention': 'attention',
    'SparseConv3d': 'conv',
    'SparseInverseConv3d': 'conv',
    'sparseconv3d_func': 'conv',
    'SparseDownsample': 'spatial',
    'SparseUpsample': 'spatial',
    'SparseSubdivide' : 'spatial'
}

__submodules = ['transformer']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .basic import *
    from .norm import *
    from .nonlinearity import *
    from .linear import *
    from .attention import *
    from .conv import *
    from .spatial import *
    import transformer
