import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'ElasticSLatEncoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoder': 'structured_latent_vae',
    'ElasticSLatRadianceFieldDecoder': 'structured_latent_vae',
    'ElasticSLatMeshDecoder': 'structured_latent_vae',
    
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # Sparse SDF VQVAE models
    'SparseSDFVQVAE': 'autoencoders.ss_vqvae',
    'Direct3DS2_VQVAE': 'autoencoders.ss_vqvae',  # 向后兼容的别名
    'SparseVectorQuantizer': 'autoencoders.ss_vqvae',
}

__submodules = []

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


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    state = load_file(model_file)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        msg = str(exc)
        if "conv.kernel" in msg and "conv.weight" in msg:
            import trellis.modules.sparse as _tsp

            _sc = getattr(_tsp, "SparseConv3d", None)
            _impl = getattr(_sc, "__module__", "n/a") if _sc is not None else "n/a"
            raise RuntimeError(
                f"{msg}\n"
                f"[trellis.models] 诊断: trellis.modules.sparse.BACKEND={getattr(_tsp, 'BACKEND', '?')!r}, "
                f"SparseConv3d.__module__={_impl!r}。"
                "官方权重为 spconv（*.conv.weight）；若见 *.conv.kernel 说明仍在用 torchsparse。"
                "请确认 load_trellis_* 在 import pipeline 之前调用了 set_sparse_backend('spconv')，并已安装 spconv。"
            ) from exc
        raise

    # Some Trellis configs set ``use_fp16=True`` and cast activations to
    # ``model.dtype`` during forward. Loading safetensors afterwards can restore
    # fp32 weights, so re-apply the model's dtype conversion just like the
    # training checkpoint loader does.
    if bool(getattr(model, "use_fp16", False)) and hasattr(model, "convert_to_fp16"):
        model.convert_to_fp16()

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import (
        SparseStructureEncoder, 
        SparseStructureDecoder,
    )
    
    from .sparse_structure_flow import SparseStructureFlowModel
    
    from .structured_latent_vae import (
        SLatEncoder,
        SLatGaussianDecoder,
        SLatRadianceFieldDecoder,
        SLatMeshDecoder,
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatRadianceFieldDecoder,
        ElasticSLatMeshDecoder,
    )
    
    from .structured_latent_flow import (
        SLatFlowModel,
        ElasticSLatFlowModel,
    )
