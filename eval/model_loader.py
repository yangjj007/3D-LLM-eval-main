"""
Unified model loading for the ShapeLLM-Omni evaluation framework.

Encapsulates loading of VQVAE, LLM, and Trellis pipelines
into a single ModelBundle, driven by YAML configuration.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
from huggingface_hub import hf_hub_download


def _eval_repo_root() -> Path:
    """``3D-LLM-eval-main`` (parent of ``eval``). ``eval/model_loader.py`` → parents[1]."""
    return Path(__file__).resolve().parents[1]


def resolve_hf_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """
    Resolve ``model.hf_cache_dir`` to an absolute path under the eval repo when relative
    (e.g. ``./eval_data/hf_cache``), so downloads always land in ``3D-LLM-eval-main/eval_data/hf_cache``
    regardless of process cwd.
    """
    if not cache_dir:
        return None
    p = Path(str(cache_dir)).expanduser()
    if not p.is_absolute():
        p = (_eval_repo_root() / p).resolve()
    return str(p)


def _try_reset_hf_hub_http() -> None:
    """Best-effort: avoid stale httpx client after connection errors (hub / Trellis downloads)."""
    try:
        import huggingface_hub.utils._http as hub_http  # type: ignore

        reset = getattr(hub_http, "reset_sessions", None) or getattr(hub_http, "close_session", None)
        if callable(reset):
            reset()
    except Exception:
        pass


def _is_retryable_hub_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "connection reset",
        "remote end closed",
        "broken pipe",
        "client has been closed",
        "cannot send a request",
        "timed out",
        "timeout",
        "503",
        "502",
        "429",
        "eof occurred",
        "ssl",
        "temporary failure",
    )
    if any(n in msg for n in needles):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) is not None:
        import errno as errno_mod

        return exc.errno in (
            errno_mod.ECONNRESET,
            errno_mod.ETIMEDOUT,
            errno_mod.EPIPE,
            errno_mod.ECONNABORTED,
            errno_mod.ECONNREFUSED,
        )
    if isinstance(exc, RuntimeError) and ("client has been closed" in msg or "cannot send a request" in msg):
        return True
    return False


@contextmanager
def _optional_hf_hub_cache(cache_dir: Optional[str]) -> Iterator[None]:
    """
    Trellis ``Pipeline.from_pretrained(path)`` only accepts ``path`` (ShapeLLM-Omni official API).

    Pin all Hugging Face downloads under the eval repo: set ``HF_HOME`` to ``eval_data/hf_cache``,
    ``HF_HUB_CACHE`` to ``<HF_HOME>/hub``, and ``TRANSFORMERS_CACHE`` for Trellis CLIP weights.
    """
    if not cache_dir:
        yield
        return
    root = resolve_hf_cache_dir(cache_dir)
    if not root:
        yield
        return
    os.makedirs(root, exist_ok=True)
    hub = os.path.join(root, "hub")
    os.makedirs(hub, exist_ok=True)
    tx = os.path.join(root, "transformers")
    os.makedirs(tx, exist_ok=True)

    keys = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
    saved: Dict[str, Optional[str]] = {}
    for k in keys:
        saved[k] = os.environ.get(k)
    os.environ["HF_HOME"] = root
    os.environ["HF_HUB_CACHE"] = hub
    os.environ["TRANSFORMERS_CACHE"] = tx
    try:
        yield
    finally:
        for k in keys:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]  # type: ignore[assignment]


class TextProcessor:
    """Small text-only processor compatible with the existing inference engines."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages: Any, tokenize: bool = False, add_generation_prompt: bool = True) -> Any:
        normalized = []
        for message in messages:
            item = dict(message)
            content = item.get("content")
            if isinstance(content, list):
                item["content"] = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) and part.get("type") == "text" else str(part)
                    for part in content
                )
            normalized.append(item)
        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def __call__(self, text: Any, padding: bool = True, return_tensors: str = "pt", **kwargs: Any) -> Any:
        return self.tokenizer(text, padding=padding, return_tensors=return_tensors, **kwargs)


@dataclass
class ModelBundle:
    """Container holding all loaded models needed for evaluation."""

    vqvae: Any = None
    llm: Any = None
    processor: Any = None
    tokenizer: Any = None
    pipeline_text: Any = None
    pipeline_image: Any = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def _looks_like_vlm(config: Any) -> bool:
    """
    Return True if this HF config describes a vision-language model.

    Such checkpoints must be loaded with AutoModelForVision2Seq / Qwen2_5_VLForConditionalGeneration,
    not AutoModelForCausalLM.
    """
    cls_name = type(config).__name__
    if any(tag in cls_name for tag in ("VL", "Llava", "Idefics", "Mllama", "Kosmos", "Fuyu", "Pixtral")):
        return True
    mt = getattr(config, "model_type", None)
    if mt is None:
        return False
    mt_l = str(mt).lower()
    if "_vl" in mt_l or mt_l.endswith("_vl") or "llava" in mt_l:
        return True
    if "vision" in mt_l or "multimodal" in mt_l:
        return True
    return False


def _load_vlm_llm(
    model_path: str,
    dtype: str,
    device_map: str,
    cache_dir: Optional[str],
    trust_remote_code: bool,
    *,
    config: Optional[Any] = None,
) -> tuple:
    """Load a HF VLM (e.g. Qwen2.5-VL) with AutoProcessor + tokenizer."""
    from transformers import AutoProcessor

    load_kw: Dict[str, Any] = dict(
        torch_dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if config is not None:
        load_kw["config"] = config

    model = None
    try:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kw)
    except (ImportError, AttributeError, ValueError, TypeError, KeyError):
        pass
    if model is None:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kw)
        except ImportError as exc:
            raise ImportError(
                "无法加载视觉语言模型：当前 transformers 缺少 AutoModelForVision2Seq 与 "
                "Qwen2_5_VLForConditionalGeneration。请升级 transformers，或在配置中将 "
                "model.llm_loader 设为兼容的加载方式。"
            ) from exc

    processor = AutoProcessor.from_pretrained(
        model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    tokenizer = processor.tokenizer
    return model, processor, tokenizer


def load_vqvae(
    repo_id: str = "yejunliang23/3DVQVAE",
    filename: str = "3DVQVAE.bin",
    num_embeddings: int = 8192,
    device: torch.device = torch.device("cuda"),
    cache_dir: Optional[str] = None,
) -> Any:
    from trellis.models.sparse_structure_vqvae import VQVAE3D

    vqvae = VQVAE3D(num_embeddings=num_embeddings)
    vqvae.eval()
    filepath = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    state_dict = torch.load(filepath, map_location="cpu")
    vqvae.load_state_dict(state_dict)
    vqvae = vqvae.to(device)
    return vqvae


def load_llm(
    model_path: str = "yejunliang23/ShapeLLM-7B-omni",
    dtype: str = "auto",
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
    loader: str = "qwen2_5_vl",
    trust_remote_code: bool = True,
) -> tuple:
    if loader == "qwen2_5_vl":
        return _load_vlm_llm(
            model_path,
            dtype,
            device_map,
            cache_dir,
            trust_remote_code,
            config=None,
        )

    if loader not in {"auto", "causal_lm", "text"}:
        raise ValueError(f"Unsupported model.llm_loader={loader!r}. Expected qwen2_5_vl, auto, causal_lm, or text.")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if _looks_like_vlm(config):
        return _load_vlm_llm(
            model_path,
            dtype,
            device_map,
            cache_dir,
            trust_remote_code,
            config=config,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = TextProcessor(tokenizer)
    return model, processor, tokenizer


def load_trellis_text_pipeline(
    pretrained_path: str = "JeffreyXiang/TRELLIS-text-xlarge",
    device: torch.device = torch.device("cuda"),
    cache_dir: Optional[str] = None,
    *,
    max_retries: int = 6,
) -> Any:
    """Trellis Hub weights match **spconv**; snapshot/restore sparse backend around load."""
    import trellis.modules.sparse as tsp

    prev_backend = tsp.BACKEND
    tsp.set_sparse_backend("spconv")
    try:
        # Import pipeline only after spconv is active (pipelines import ``..modules.sparse``).
        from trellis.pipelines import TrellisTextTo3DPipeline

        for attempt in range(max_retries):
            try:
                with _optional_hf_hub_cache(cache_dir):
                    pipeline = TrellisTextTo3DPipeline.from_pretrained(pretrained_path)
                pipeline.to(device)
                if attempt:
                    print(f"[ModelLoader] Trellis text pipeline loaded after {attempt + 1} attempt(s).", flush=True)
                return pipeline
            except Exception as exc:
                if attempt >= max_retries - 1 or not _is_retryable_hub_error(exc):
                    raise
                delay = min(90.0, (2**attempt) * 2.0 + random.uniform(0.0, 2.0))
                print(
                    f"[ModelLoader] Trellis text pipeline load failed ({exc!r}); "
                    f"retry in {delay:.1f}s ({attempt + 1}/{max_retries - 1})...",
                    flush=True,
                )
                _try_reset_hf_hub_http()
                time.sleep(delay)
    finally:
        tsp.set_sparse_backend(prev_backend)


def load_trellis_image_pipeline(
    pretrained_path: str = "JeffreyXiang/TRELLIS-image-large",
    device: torch.device = torch.device("cuda"),
    cache_dir: Optional[str] = None,
    *,
    max_retries: int = 6,
) -> Any:
    import trellis.modules.sparse as tsp

    prev_backend = tsp.BACKEND
    tsp.set_sparse_backend("spconv")
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline

        for attempt in range(max_retries):
            try:
                with _optional_hf_hub_cache(cache_dir):
                    pipeline = TrellisImageTo3DPipeline.from_pretrained(pretrained_path)
                pipeline.to(device)
                if attempt:
                    print(f"[ModelLoader] Trellis image pipeline loaded after {attempt + 1} attempt(s).", flush=True)
                return pipeline
            except Exception as exc:
                if attempt >= max_retries - 1 or not _is_retryable_hub_error(exc):
                    raise
                delay = min(90.0, (2**attempt) * 2.0 + random.uniform(0.0, 2.0))
                print(
                    f"[ModelLoader] Trellis image pipeline load failed ({exc!r}); "
                    f"retry in {delay:.1f}s ({attempt + 1}/{max_retries - 1})...",
                    flush=True,
                )
                _try_reset_hf_hub_http()
                time.sleep(delay)
    finally:
        tsp.set_sparse_backend(prev_backend)


def load_models(config: Dict[str, Any]) -> ModelBundle:
    """
    Load all models specified in the evaluation config.

    Config keys under `model`:
        llm_path (str): HF repo or local path for LLM
        llm_loader (str): qwen2_5_vl for ShapeLLM-Omni, or auto/causal_lm/text for text-only LMs
        vqvae_repo (str): HF repo for 3DVQVAE weights
        vqvae_filename (str): Weight file name
        vqvae_num_embeddings (int): Codebook size
        dtype (str): Torch dtype string
        device_map (str): Device mapping strategy
        trellis_text_path (str): Trellis text-to-3D model path (optional)
        trellis_image_path (str): Trellis image-to-3D model path (optional)
        load_llm (bool): Whether to load the LLM (default True)
        load_trellis_text (bool): Whether to load Trellis text pipeline (default False)
        load_trellis_image (bool): Whether to load Trellis image pipeline (default False)
    """
    from eval.utils.path_bootstrap import ensure_third_party_on_path

    ensure_third_party_on_path()
    model_cfg = config.get("model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_cache_dir = resolve_hf_cache_dir(model_cfg.get("hf_cache_dir"))
    hf_hub_root = os.path.join(hf_cache_dir, "hub") if hf_cache_dir else None
    transformers_cache = os.path.join(hf_cache_dir, "transformers") if hf_cache_dir else None
    if hf_cache_dir:
        os.makedirs(hf_cache_dir, exist_ok=True)
        if hf_hub_root:
            os.makedirs(hf_hub_root, exist_ok=True)
        if transformers_cache:
            os.makedirs(transformers_cache, exist_ok=True)
        print(
            f"[ModelLoader] Hugging Face cache root: {hf_cache_dir}\n"
            f"  hub (Trellis / hf_hub_download): {hf_hub_root}\n"
            f"  transformers: {transformers_cache}",
            flush=True,
        )

    bundle = ModelBundle(device=device)

    # VQVAE (always loaded)
    print("[ModelLoader] Loading VQVAE...")
    bundle.vqvae = load_vqvae(
        repo_id=model_cfg.get("vqvae_repo", "yejunliang23/3DVQVAE"),
        filename=model_cfg.get("vqvae_filename", "3DVQVAE.bin"),
        num_embeddings=model_cfg.get("vqvae_num_embeddings", 8192),
        device=device,
        cache_dir=hf_hub_root,
    )

    # LLM
    if model_cfg.get("load_llm", True):
        llm_loader = str(model_cfg.get("llm_loader", "qwen2_5_vl"))
        print(f"[ModelLoader] Loading LLM with loader={llm_loader!r}...")
        bundle.llm, bundle.processor, bundle.tokenizer = load_llm(
            model_path=model_cfg.get("llm_path", "yejunliang23/ShapeLLM-7B-omni"),
            dtype=model_cfg.get("dtype", "auto"),
            device_map=model_cfg.get("device_map", "auto"),
            cache_dir=transformers_cache or hf_hub_root,
            loader=llm_loader,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )

    # Trellis Text Pipeline
    if model_cfg.get("load_trellis_text", False):
        print("[ModelLoader] Loading Trellis Text-to-3D pipeline...")
        bundle.pipeline_text = load_trellis_text_pipeline(
            pretrained_path=model_cfg.get(
                "trellis_text_path", "JeffreyXiang/TRELLIS-text-xlarge"
            ),
            device=device,
            cache_dir=hf_cache_dir,
        )

    # Trellis Image Pipeline
    if model_cfg.get("load_trellis_image", False):
        print("[ModelLoader] Loading Trellis Image-to-3D pipeline...")
        bundle.pipeline_image = load_trellis_image_pipeline(
            pretrained_path=model_cfg.get(
                "trellis_image_path", "JeffreyXiang/TRELLIS-image-large"
            ),
            device=device,
            cache_dir=hf_cache_dir,
        )

    print("[ModelLoader] All models loaded successfully.")
    return bundle
