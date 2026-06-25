"""Sparse SDF VQVAE + Qwen3-VL (HF discrete tokens) evaluation adapter."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import GenerationConfig

from .base import GenResult, MeshInput, ModelAdapter, TokenSeq
from eval.utils import sdf_processing

# 与 Med-3D-LLM `build_qwen3vl_sft_3d_jsonl.py` 中 DEFAULT_* 对齐
DEFAULT_CAPTION_PROMPT = "Describe this 3D shape in one sentence:"
DEFAULT_RECONSTRUCT_PROMPT = "Reconstruct this 3D shape in mesh token format:"
DEFAULT_INPUT_BAND_FACTOR = 0.5
DEFAULT_PREPROCESSING_EXTRA_BAND_FACTOR = 4.0
DEFAULT_INFERENCE_BAND_FACTOR = 2.0


@dataclass
class EncodedSparseSample:
    token_ids: List[int]
    coords_block: Any
    mesh_token_string: str
    bpe_ids: Any
    bpe_anchors: Any
    num_tokens_raw: int
    num_tokens_bpe: int


def build_sparse_understanding_user_prompt(mesh_token_string: str, caption_prompt: str) -> str:
    """训练侧: u_desc = f\"{mesh_str}\\n{caption_prompt}\""""
    return f"{mesh_token_string}\n{caption_prompt}"


def build_sparse_generation_user_prompt(caption: str, reconstruct_prompt: str = "") -> str:
    """
    评测/推理侧发给 **LLM** 的用户文本：``caption`` + 换行 + ``reconstruct_prompt``（若后者非空）。

    Trellis 上色等步骤应单独传纯 ``caption``，不要把 reconstruct 指令拼进条件文本。
    """
    cap = str(caption or "").strip()
    rp = str(reconstruct_prompt or "").strip()
    if not rp:
        return cap
    if not cap:
        return rp
    return f"{cap}\n{rp}"


def resolve_reconstruct_prompt_from_inference(inf: Dict[str, Any]) -> str:
    """
    优先 ``inference.reconstruct_prompt``；否则兼容旧键 ``prompt_prefix``（去掉首尾空白）。
    """
    rp = inf.get("reconstruct_prompt")
    if rp is not None and str(rp).strip():
        return str(rp).strip()
    legacy = inf.get("prompt_prefix")
    if legacy is not None and str(legacy).strip():
        return str(legacy).strip()
    return DEFAULT_RECONSTRUCT_PROMPT


def _truthy_cfg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _eval_repo_root() -> Path:
    """本仓库根目录（含 ``eval/``），与 ``SparseSDFQwen3Adapter._repo_root()`` 一致。"""
    return Path(__file__).resolve().parents[2]


def _looks_like_local_filesystem_path(raw: str) -> bool:
    """
    判定「明显是本地路径」以便在目录不存在时报错，而不是交给 HF 误当成 repo id。

    单层 ``org/model`` 仍可能是 Hub，故仅当路径层级较深（≥3 段）、或以 ``./`` ``../`` 开头、
    或形如 Windows 盘符路径时视为本地。
    """
    s = raw.strip().replace("\\", "/")
    if not s:
        return False
    if s.startswith(("./", "../")):
        return True
    if len(s) >= 2 and s[1] == ":":
        return True
    parts = [x for x in s.split("/") if x]
    return len(parts) >= 3


def _resolve_hf_name_or_path(name_or_path: str) -> str:
    """
    解析 ``model.llm_path`` / ``tokenizer_name_or_path``，供 ``transformers`` 使用。

    相对路径仅 ``Path.cwd().resolve()`` 时，常把 ``LLaMA-Factory/...`` 错解到
    ``3D-LLM-eval-main/LLaMA-Factory/...``（目录不存在），HF 会误当作 Hub repo id
    并抛出 ``Repo id must be in the form ...``。此处依次尝试：

    1. 当前工作目录
    2. eval 仓库根目录
    3. 仓库父目录（常见于 LLaMA-Factory 与 3D-LLM-eval-main 同级）

    若明显是本地路径却找不到目录，抛出 ``FileNotFoundError``；否则原样返回（如纯 Hub id）。
    """
    raw = str(name_or_path).strip()
    if not raw:
        raise ValueError("empty model path (llm_path / tokenizer_name_or_path)")
    p = Path(raw).expanduser()
    if p.is_absolute():
        r = p.resolve()
        if r.is_dir() or r.is_file():
            return str(r)
        raise FileNotFoundError(
            f"本地模型路径不存在: {r}\n请确认 llm_path 指向有效的 checkpoint 目录或文件。"
        )
    rel = p
    tried: List[str] = []
    for base in (Path.cwd(), _eval_repo_root(), _eval_repo_root().parent):
        cand = (base / rel).resolve()
        tried.append(str(cand))
        if cand.is_dir():
            return str(cand)
    if _looks_like_local_filesystem_path(raw):
        raise FileNotFoundError(
            f"无法在本地找到 checkpoint 目录: {raw!r}\n"
            "已尝试:\n  - "
            + "\n  - ".join(tried)
            + "\n请使用绝对路径，或将工程放在上述目录之一下，或从含 LLaMA-Factory 的目录启动。"
        )
    return raw


def _parse_eval_debug(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """从 ``cfg['debug']`` 与 ``cfg['inference']`` 读取唯一调试开关 ``verbose_eval``。"""
    dbg = cfg.get("debug") or {}
    inf = cfg.get("inference") or {}
    value = inf.get("verbose_eval", dbg.get("verbose_eval"))
    return {
        "verbose_eval": _truthy_cfg(value, False),
    }


def _white_mesh_postprocess_record(pred_mesh: Any) -> Dict[str, Any]:
    """从 ``pred_mesh.metadata`` 取出 ``sparse_mesh_export`` 写入的后处理调试信息以便写入 JSON。"""
    if pred_mesh is None:
        return {}
    md = getattr(pred_mesh, "metadata", None) or {}
    rec = md.get("white_mesh_postprocess")
    if rec is None:
        return {}
    return {"white_mesh_postprocess": rec}


def _text_preview(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[{len(text)} chars total]"


def _debug_text_preview(text: str) -> str:
    return _text_preview(text, 200)


class _NoopMMPlugin:
    def process_messages(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        return messages

    def process_token_ids(self, token_ids: List[int], labels: Any, *args: Any, **kwargs: Any) -> Tuple[List[int], Any]:
        return token_ids, labels


class _QwenTextTemplate:
    """评测只用纯文本 Qwen chat template。"""

    name = "qwen3_nothink"
    stop_words = ["<|im_end|>"]
    mm_plugin = _NoopMMPlugin()

    def __init__(self, default_system: Optional[str] = None) -> None:
        self.default_system = default_system

    def encode_oneturn(
        self,
        tokenizer: Any,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        del tools
        prompt = self._format_prompt(messages, system)
        return tokenizer.encode(prompt, add_special_tokens=False), []

    def get_stop_token_ids(self, tokenizer: Any) -> List[int]:
        stop_token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            stop_token_ids.add(tokenizer.convert_tokens_to_ids(token))
        return [int(x) for x in stop_token_ids if x is not None and int(x) >= 0]

    def _format_prompt(self, messages: List[Dict[str, str]], system: Optional[str]) -> str:
        parts: List[str] = []
        system_text = system if system is not None else self.default_system
        if system_text:
            parts.append(f"<|im_start|>system\n{system_text}<|im_end|>\n")

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "assistant" and content == "":
                parts.append("<|im_start|>assistant\n")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)


def _require_non_empty_sparse_sdf(
    sparse: Dict[str, "torch.Tensor"],
    *,
    sample_id: str,
    mesh_path: str,
    resolution: int,
    threshold_factor: float,
) -> None:
    t = sparse.get("sparse_sdf") if sparse else None
    if t is None or t.numel() == 0 or t.shape[0] == 0:
        raise RuntimeError(
            "Sparse SDF voxel count is 0; stopping this sample.\n"
            f"  sample_id: {sample_id!r}\n"
            f"  mesh_path: {mesh_path!r}\n"
            f"  current: sdf_resolution={resolution}, sdf_threshold_factor={threshold_factor}\n"
            "Sparse 256 VQ-VAE expects the Med training preprocessing defaults: "
            "sdf_resolution=256 and sdf_threshold_factor=4.0. If a stale cache "
            "was written with older settings, delete model.sdf_cache_dir for this sample."
        )


def _load_tokenizer_for_sparse(llm_path: str, model_cfg: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """
    Load HF tokenizer for Qwen2/3-VL checkpoints.

    - 部分导出的 ``tokenizer_config.json`` 会把 ``extra_special_tokens`` 写成 list，
      与 ``Qwen2TokenizerFast`` 当前实现不兼容（``list`` 无 ``.keys()``）。
    - 若再 ``use_fast=False``，慢速 ``Qwen2Tokenizer`` 需要 ``vocab_file``，而仅含 ``tokenizer.json``
      的目录会报 ``vocab_file`` 为 ``None``。

    策略：若检测到 list 型 ``extra_special_tokens``，复制 tokenizer 相关文件到临时目录、
    从配置中移除该键后用 **fast** 加载；临时目录路径随返回值交给 adapter 在 ``unload`` 时删除。

    可选 ``model.tokenizer_name_or_path``：仅从该路径加载 tokenizer（权重仍在 ``llm_path``）。
    可选 ``model.tokenizer_use_fast``：显式 True/False；默认自动（优先 fast）。
    """
    from transformers import AutoTokenizer

    explicit = model_cfg.get("tokenizer_name_or_path")
    root = Path(_resolve_hf_name_or_path(str(explicit or llm_path)))
    use_fast_opt = model_cfg.get("tokenizer_use_fast", model_cfg.get("use_fast_tokenizer", None))
    trust_remote_code = _truthy_cfg(model_cfg.get("trust_remote_code"), True)

    def _from(path: Path, use_fast: bool) -> Any:
        return AutoTokenizer.from_pretrained(
            str(path),
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )

    if explicit:
        if use_fast_opt is None:
            return _from(root, True), None
        return _from(root, bool(use_fast_opt)), None

    cfg_path = root / "tokenizer_config.json"
    need_patch = False
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            tc = json.load(f)
        if isinstance(tc.get("extra_special_tokens"), list):
            need_patch = True

    if need_patch:
        staging = tempfile.mkdtemp(prefix="sparse_eval_tok_")
        try:
            names = (
                "tokenizer_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "vocab.json",
                "vocab.txt",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template.jinja",
                "chat_template.json",
            )
            sp = Path(staging)
            for name in names:
                src = root / name
                if src.is_file():
                    shutil.copy2(src, sp / name)
            pcfg = sp / "tokenizer_config.json"
            with open(pcfg, encoding="utf-8") as f:
                patched = json.load(f)
            patched.pop("extra_special_tokens", None)
            with open(pcfg, "w", encoding="utf-8") as f:
                json.dump(patched, f, indent=2, ensure_ascii=False)
            return _from(sp, True), staging
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    if use_fast_opt is not None:
        return _from(root, bool(use_fast_opt)), None

    try:
        return _from(root, True), None
    except Exception:
        return _from(root, False), None


class SparseSDFQwen3Adapter(ModelAdapter):
    name = "sparse_sdf_qwen3"
    supported_tasks = frozenset({"understanding", "vqvae_recon", "generation", "sparse_mesh"})
    capabilities = {
        "batched_understanding": True,
        "batched_vqvae_recon": True,
        "batched_generation": True,
        "generation_produces_mesh": True,
    }

    def __init__(self) -> None:
        self._vae: Optional[torch.nn.Module] = None
        self._llm: Optional[torch.nn.Module] = None
        self._tokenizer: Any = None
        self._template: Any = None
        self._processor: Any = None
        self._generating_args: Dict[str, Any] = {}
        self._tokenizer_staging_dir: Optional[str] = None
        self._device = torch.device("cpu")
        self._cfg: Dict[str, Any] = {}
        self._eval_debug: Dict[str, Any] = _parse_eval_debug({})
        self._bpe_tokenizer: Any = None
        self._trellis_text: Any = None
        self._mock_llm_enabled: bool = False
        self._mock_llm_cfg: Dict[str, Any] = {}

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _truthy(value: Any, default: bool = False) -> bool:
        return _truthy_cfg(value, default)

    @staticmethod
    def _resolve_torch_dtype(model_cfg: Dict[str, Any]) -> Any:
        raw = model_cfg.get("llm_dtype", model_cfg.get("infer_dtype", model_cfg.get("torch_dtype", "auto")))
        if raw is None:
            return "auto"
        if isinstance(raw, torch.dtype):
            return raw
        value = str(raw).strip().lower()
        if value in {"", "auto"}:
            return "auto"
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if value not in mapping:
            raise ValueError(f"Unsupported model llm_dtype/infer_dtype: {raw!r}")
        return mapping[value]

    @staticmethod
    def _resolve_attn_implementation(model_cfg: Dict[str, Any]) -> Optional[str]:
        explicit = model_cfg.get("attn_implementation")
        if explicit:
            return str(explicit)
        flash_attn = model_cfg.get("flash_attn")
        if flash_attn is None:
            return None
        value = str(flash_attn).strip().lower()
        if value in {"1", "true", "yes", "y", "on", "fa2", "flash_attention_2"}:
            return "flash_attention_2"
        if value in {"sdpa", "eager"}:
            return value
        return None

    def _load_generation_defaults(self, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        gen_args: Dict[str, Any] = {}
        model_gen = getattr(self._llm, "generation_config", None)
        if model_gen is not None and hasattr(model_gen, "to_dict"):
            gen_args.update({k: v for k, v in model_gen.to_dict().items() if v is not None})

        cfg_gen = model_cfg.get("generation_config")
        if cfg_gen:
            if not isinstance(cfg_gen, dict):
                raise ValueError("model.generation_config must be a mapping when provided")
            gen_args.update(cfg_gen)
        return gen_args

    def _load_transformers_hf(self, model_cfg: Dict[str, Any], llm_path: str, device: torch.device) -> str:
        import transformers

        resolved_llm = _resolve_hf_name_or_path(llm_path)
        self._tokenizer, self._tokenizer_staging_dir = _load_tokenizer_for_sparse(resolved_llm, model_cfg)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._processor = None
        self._template = _QwenTextTemplate(default_system=model_cfg.get("default_system"))

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self._truthy(model_cfg.get("trust_remote_code"), True),
            "torch_dtype": self._resolve_torch_dtype(model_cfg),
        }
        for key in ("cache_dir", "revision", "token"):
            if key in model_cfg:
                model_kwargs[key] = model_cfg[key]
        if "local_files_only" in model_cfg:
            model_kwargs["local_files_only"] = self._truthy(model_cfg.get("local_files_only"))
        if "low_cpu_mem_usage" in model_cfg:
            model_kwargs["low_cpu_mem_usage"] = self._truthy(model_cfg.get("low_cpu_mem_usage"))

        attn_impl = self._resolve_attn_implementation(model_cfg)
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        device_map = model_cfg.get("device_map")
        if device_map is None and device.type == "cuda":
            device_map = {"": int(device.index or 0)}
        if device_map is not None and str(device_map).strip().lower() not in {"", "none", "false"}:
            model_kwargs["device_map"] = device_map

        config_kwargs = {
            k: v
            for k, v in model_kwargs.items()
            if k in {"trust_remote_code", "cache_dir", "revision", "token", "local_files_only"}
        }
        config = transformers.AutoConfig.from_pretrained(resolved_llm, **config_kwargs)
        from eval.model_loader import _looks_like_vlm

        model_cls = transformers.AutoModelForCausalLM
        if _looks_like_vlm(config):
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_cls is None:
                model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
            if model_cls is None:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration

                    model_cls = Qwen2_5_VLForConditionalGeneration
                except ImportError as exc:
                    raise ImportError(
                        "视觉语言 checkpoint 需要 transformers 提供 AutoModelForVision2Seq 或 "
                        "Qwen2_5_VLForConditionalGeneration；当前版本两者皆不可用。"
                    ) from exc

        self._llm = model_cls.from_pretrained(resolved_llm, config=config, **model_kwargs)
        if "device_map" not in model_kwargs:
            self._llm = self._llm.to(device)
        self._llm.eval()
        self._generating_args = self._load_generation_defaults(model_cfg)
        return resolved_llm

    def _debug_log_post_llm_load(self, model_cfg: Dict[str, Any], llm_path: str) -> None:
        """verbose_eval 下打印 tokenizer / template / 默认生成参数。"""
        t = self._tokenizer
        templ = self._template
        tname = getattr(templ, "name", None) or type(templ).__name__
        print(
            f"[sparse_sdf_qwen3][debug] LLM loaded template={tname!r} "
            f"backend='transformers' llm_path={llm_path!r}",
            flush=True,
        )
        vs = getattr(t, "vocab_size", None)
        print(
            f"  tokenizer: {type(t).__name__} vocab_size={vs} "
            f"pad_token_id={getattr(t, 'pad_token_id', None)} "
            f"eos_token_id={getattr(t, 'eos_token_id', None)} "
            f"bos_token_id={getattr(t, 'bos_token_id', None)}",
            flush=True,
        )
        if templ is not None:
            try:
                stops = templ.get_stop_token_ids(t)
            except Exception as exc:  # noqa: BLE001
                stops = f"<err {exc!r}>"
            sw = getattr(templ, "stop_words", None)
            print(f"  template.get_stop_token_ids={stops!r} stop_words={sw!r}", flush=True)
        print(f"  transformers_generation_args={self._generating_args!r}", flush=True)

    def load(self, cfg: Dict[str, Any], device: torch.device) -> None:
        # Med SparseSDFVQVAE checkpoints use torchsparse; Trellis Hub weights use spconv
        # (``load_trellis_*`` switches to spconv only while loading the pipeline).
        os.environ["SPARSE_BACKEND"] = "torchsparse"
        from trellis.modules.sparse import set_sparse_backend

        set_sparse_backend("torchsparse")
        self._cfg = cfg
        self._device = device
        self._eval_debug = _parse_eval_debug(cfg)
        model_cfg = cfg.get("model", {})
        from eval.sparse_backend import load_vae_from_config

        vae_config = model_cfg.get("vae_config")
        vae_ckpt = model_cfg.get("vae_ckpt")
        if not vae_config or not vae_ckpt:
            raise ValueError("model.vae_config and model.vae_ckpt are required for sparse_sdf_qwen3")
        self._vae = load_vae_from_config(vae_config, vae_ckpt, str(device))
        self._vae = self._vae.to(device)
        self._load_bpe_tokenizer(model_cfg)

        if self._eval_debug["verbose_eval"]:
            print(
                f"[sparse_sdf_qwen3][debug] load task={cfg.get('task')!r} device={device!r}\n"
                f"  vae_config={vae_config!r}\n  vae_ckpt={vae_ckpt!r}",
                flush=True,
            )

        self._llm = None
        self._tokenizer = None
        self._trellis_text = None
        task = cfg.get("task", "")
        llm_path = model_cfg.get("llm_path")
        inf = cfg.get("inference") or {}
        self._mock_llm_cfg = dict(inf.get("mock_llm") or {})
        self._mock_llm_enabled = bool(
            _truthy_cfg(self._mock_llm_cfg.get("enabled"), False)
            and task in ("generation", "sparse_mesh")
        )

        col = model_cfg.get("colorization") or {}
        if _truthy_cfg(col.get("enabled"), False):
            from eval.utils.path_bootstrap import ensure_third_party_on_path

            ensure_third_party_on_path()
            from eval.model_loader import load_trellis_text_pipeline, resolve_hf_cache_dir

            tp = str(model_cfg.get("trellis_text_path", "JeffreyXiang/TRELLIS-text-xlarge"))
            hf_raw = model_cfg.get("hf_cache_dir")
            hf_cache = resolve_hf_cache_dir(str(hf_raw)) if hf_raw else None
            if hf_cache:
                print(f"[sparse_sdf_qwen3] Trellis HF cache root: {hf_cache}", flush=True)
            print(f"[sparse_sdf_qwen3] Loading Trellis for colorization: {tp!r}", flush=True)
            self._trellis_text = load_trellis_text_pipeline(tp, device=device, cache_dir=hf_cache)

        if task in ("understanding", "generation", "sparse_mesh"):
            if self._mock_llm_enabled:
                print(
                    "[sparse_sdf_qwen3] inference.mock_llm.enabled=True — skipping HF LLM for "
                    f"task={task!r} (fixed mesh token string).",
                    flush=True,
                )
            else:
                if not llm_path:
                    raise ValueError(
                        "model.llm_path is required for sparse_sdf_qwen3 when task is understanding or generation"
                    )
                resolved_llm = self._load_transformers_hf(model_cfg, str(llm_path), device)
                if self._eval_debug["verbose_eval"]:
                    self._debug_log_post_llm_load(model_cfg, resolved_llm)

    def unload(self) -> None:
        self._vae = None
        self._llm = None
        self._tokenizer = None
        self._template = None
        self._processor = None
        self._generating_args = {}
        self._bpe_tokenizer = None
        self._trellis_text = None
        self._mock_llm_enabled = False
        self._mock_llm_cfg = {}
        if self._tokenizer_staging_dir:
            shutil.rmtree(self._tokenizer_staging_dir, ignore_errors=True)
            self._tokenizer_staging_dir = None

    def _model_cfg(self) -> Dict[str, Any]:
        return self._cfg.get("model", {})

    def _sdf_resolution(self) -> int:
        return int(self._model_cfg().get("sdf_resolution", sdf_processing.DEFAULT_SDF_RESOLUTION))

    def _sdf_threshold_factor(self) -> float:
        return float(
            self._model_cfg().get(
                "sdf_threshold_factor", sdf_processing.DEFAULT_SDF_THRESHOLD_FACTOR
            )
        )

    def _encoder_input_threshold(self) -> Tuple[float, float, float]:
        mc = self._model_cfg()
        input_band = float(mc.get("input_band_factor", DEFAULT_INPUT_BAND_FACTOR))
        extra_band = float(
            mc.get("preprocessing_extra_band_factor", DEFAULT_PREPROCESSING_EXTRA_BAND_FACTOR)
        )
        if extra_band <= 0:
            raise ValueError("model.preprocessing_extra_band_factor must be > 0")
        return input_band / extra_band, input_band, extra_band

    def _filter_sparse_for_encoder(
        self,
        sparse: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        mesh_path: str,
    ) -> Tuple[Dict[str, torch.Tensor], int, int, float]:
        thresh, input_band, extra_band = self._encoder_input_threshold()
        sdf = sparse["sparse_sdf"]
        if sdf.ndim == 1:
            sdf = sdf.unsqueeze(-1)
        mask = sdf.abs().squeeze(-1) <= float(thresh)
        n_total = int(mask.numel())
        n_keep = int(mask.sum().item())
        if n_keep <= 0:
            raise RuntimeError(
                "Sparse SDF tight encoder band is empty.\n"
                f"  sample_id: {sample_id!r}\n"
                f"  mesh_path: {mesh_path!r}\n"
                f"  threshold=abs(sdf)<={thresh:.6f} "
                f"(input_band_factor={input_band}, preprocessing_extra_band_factor={extra_band})\n"
                "This should match Med training: wide SDF preprocessing at 4.0 voxels "
                "and encoder input at 0.5 voxels."
            )
        return (
            {
                "sparse_sdf": sdf[mask],
                "sparse_index": sparse["sparse_index"][mask],
            },
            n_keep,
            n_total,
            float(thresh),
        )

    def _decode_sparse_with_pruning(self, vae: torch.nn.Module, decoded_sparse: Any) -> Any:
        mc = self._model_cfg()
        res = self._sdf_resolution()
        inf_band = float(mc.get("inference_band_factor", DEFAULT_INFERENCE_BAND_FACTOR))
        occ_res = int(mc.get("inference_occ_resolution", res))
        block_side = int(getattr(vae, "vq_block_side", 1) or 1)
        latent_res = int(getattr(vae, "resolution", 32)) // max(block_side, 1)
        band_prune = {
            "mode": "seed",
            "seed_coords": decoded_sparse.coords,
            "seed_resolution": latent_res,
            "output_resolution": res,
            "extra_band_factor": inf_band,
        }
        gt_prune = {
            "mode": "geometry",
            "extra_band_factor": inf_band,
            "resolution": res,
            "occ_resolution": occ_res,
        }
        if self._eval_debug["verbose_eval"]:
            print(
                "[sparse_sdf_qwen3][debug] Decode pruning "
                f"resolution={res} band={inf_band} occ_resolution={occ_res} "
                f"seed_resolution={latent_res}",
                flush=True,
            )
        return vae.Decode(decoded_sparse, band_prune=band_prune, gt_prune=gt_prune)

    def _sparse_recon_to_meshes(self, recon: Any):
        from eval.utils.sparse_mesh_export import sparse_sdf_to_meshes

        mc = self._model_cfg()
        wmp = mc.get("white_mesh_postprocess")
        return sparse_sdf_to_meshes(
            recon,
            voxel_resolution=self._sdf_resolution(),
            mc_threshold=float(mc.get("mc_threshold", 0.0)),
            postprocess_cfg=wmp if isinstance(wmp, dict) else None,
            device=self._device,
        )

    def _load_bpe_tokenizer(self, model_cfg: Dict[str, Any]) -> None:
        from eval.utils.bpe_3d import BPE3DTokenizer

        merge_table = str(model_cfg.get("bpe_merge_table") or "").strip()
        if merge_table:
            p = Path(merge_table).expanduser()
            if not p.is_absolute():
                p = (self._repo_root() / p).resolve()
            self._bpe_tokenizer = BPE3DTokenizer.load(str(p))
            if self._eval_debug["verbose_eval"]:
                print(
                    f"[sparse_sdf_qwen3][debug] loaded 3D BPE merge_table={str(p)!r} "
                    f"vocab_size={self._bpe_tokenizer.vocab_size}",
                    flush=True,
                )
            return

        base_vocab = int(model_cfg.get("bpe_base_vocab_size", model_cfg.get("vqvae_num_embeddings", 8192)))
        self._bpe_tokenizer = BPE3DTokenizer(base_vocab_size=base_vocab)
        if self._eval_debug["verbose_eval"]:
            print(
                "[sparse_sdf_qwen3][debug] model.bpe_merge_table is empty; "
                f"using identity 3D BPE tokenizer base_vocab_size={base_vocab}",
                flush=True,
            )

    def _require_bpe_tokenizer(self) -> Any:
        if self._bpe_tokenizer is None:
            raise RuntimeError("3D BPE tokenizer is not loaded.")
        return self._bpe_tokenizer

    def _mock_llm_raw_response(self) -> str:
        s = str(self._mock_llm_cfg.get("mesh_token_string") or "").strip()
        if s:
            return s
        return "<mesh_start><morton_0><mesh_0><mesh_end>"

    def _colorize_mesh(self, pred_mesh: Any, caption: str) -> Tuple[Any, Dict[str, Any]]:
        """Return ``(textured_trimesh_or_None, extra_dict)``."""
        extra: Dict[str, Any] = {}
        col = self._model_cfg().get("colorization") or {}
        if not _truthy_cfg(col.get("enabled"), False) or self._trellis_text is None or pred_mesh is None:
            return None, extra
        input_orientation = (
            str(col.get("input_orientation", "yup") or "yup")
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
        )
        extra["glb_input_orientation"] = input_orientation
        extra["glb_output_orientation"] = "yup"
        try:
            from eval.colorization.trellis_colorizer import trellis_mesh_to_textured_glb

            slat_sp = col.get("slat_sampler_params")
            if slat_sp is not None and not isinstance(slat_sp, dict):
                slat_sp = {}
            elif slat_sp is None:
                slat_sp = {}
            glb = trellis_mesh_to_textured_glb(
                self._trellis_text,
                pred_mesh,
                (caption or "3D object").strip() or "3D object",
                simplify=float(col.get("simplify", 0.95)),
                texture_size=int(col.get("texture_size", 1024)),
                input_orientation=input_orientation,
                slat_sampler_params=slat_sp,
            )
            return glb, extra
        except Exception as exc:  # noqa: BLE001
            tb = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True)
            )
            extra["colorize_error"] = repr(exc)
            extra["colorize_traceback"] = tb
            print(
                f"[sparse_sdf_qwen3][error] Trellis colorization failed: {extra['colorize_error']}\n{tb}",
                file=sys.stderr,
                flush=True,
            )
            if self._eval_debug.get("verbose_eval"):
                print(
                    f"[sparse_sdf_qwen3][warn] Trellis colorization failed (summary): {extra['colorize_error']}",
                    flush=True,
                )
            return None, extra

    def _get_sdf(self, m: MeshInput) -> Dict[str, torch.Tensor]:
        mc = self._model_cfg()
        res = self._sdf_resolution()
        th = self._sdf_threshold_factor()
        cache = mc.get("sdf_cache_dir")
        mesh_only = bool(mc.get("sdf_from_mesh_only", True))
        watertight = bool(mc.get("sdf_watertight", False))
        compute_edge_mask = _truthy_cfg(
            mc.get("sdf_compute_edge_mask"), sdf_processing.DEFAULT_SDF_COMPUTE_EDGE_MASK
        )
        sharp_grad_dev_thresh = float(
            mc.get(
                "sdf_sharp_grad_dev_thresh",
                sdf_processing.DEFAULT_SDF_SHARP_GRAD_DEV_THRESH,
            )
        )
        sdf_p = None if mesh_only else m.sdf_path
        return sdf_processing.get_or_build_sdf_for_sample(
            m.mesh_path,
            sdf_p,
            cache,
            res,
            th,
            sample_id=str(m.sample_id),
            watertight=watertight,
            compute_edge_mask=compute_edge_mask,
            sharp_grad_dev_thresh=sharp_grad_dev_thresh,
        )

    @staticmethod
    def _suppress_stdout() -> contextlib.AbstractContextManager:
        return contextlib.redirect_stdout(io.StringIO())

    def _encode_hf_prompt(self, user_text: str, *, log_ctx: Optional[str] = None) -> List[int]:
        if self._template is None or self._tokenizer is None:
            raise RuntimeError("Transformers tokenizer/template is not loaded.")

        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": ""},
        ]
        processed = self._template.mm_plugin.process_messages(
            messages, [], [], [], self._processor
        )
        prompt_ids, _ = self._template.encode_oneturn(self._tokenizer, processed)
        prompt_ids, _ = self._template.mm_plugin.process_token_ids(
            prompt_ids, None, [], [], [], self._tokenizer, self._processor
        )
        if self._eval_debug["verbose_eval"]:
            nt = 32
            head = prompt_ids[:nt] if nt else []
            tail = prompt_ids[-nt:] if nt and len(prompt_ids) > nt else []
            print(
                f"[sparse_sdf_qwen3][debug] _encode_hf_prompt ctx={log_ctx!r} "
                f"user_text_preview={_debug_text_preview(user_text)!r}",
                flush=True,
            )
            print(
                f"  prompt_token_len={len(prompt_ids)} ids_head={head} ids_tail={tail}",
                flush=True,
            )
            try:
                take = min(128, len(prompt_ids))
                dec_prev = self._tokenizer.decode(prompt_ids[:take], skip_special_tokens=False)
                print(
                    f"  prompt_decode_preview={_debug_text_preview(dec_prev)!r}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  prompt_decode_preview failed: {exc!r}", flush=True)
        return prompt_ids

    def _generation_kwargs(self, inf: Dict[str, Any], *, greedy_default: bool) -> Dict[str, Any]:
        if self._template is None or self._tokenizer is None:
            raise RuntimeError("Transformers tokenizer/template is not loaded.")

        gen_args = dict(self._generating_args)
        gen_args.update(
            {
                "eos_token_id": self._template.get_stop_token_ids(self._tokenizer),
                "pad_token_id": self._tokenizer.pad_token_id,
            }
        )

        if "max_new_tokens" in inf:
            gen_args.pop("max_length", None)
            gen_args["max_new_tokens"] = int(inf["max_new_tokens"])
        if "repetition_penalty" in inf:
            gen_args["repetition_penalty"] = float(inf["repetition_penalty"])
        if "length_penalty" in inf:
            gen_args["length_penalty"] = float(inf["length_penalty"])
        if "top_k" in inf:
            gen_args["top_k"] = int(inf["top_k"])
        if "top_p" in inf:
            gen_args["top_p"] = float(inf["top_p"])
        if "skip_special_tokens" in inf:
            gen_args["skip_special_tokens"] = self._truthy(inf["skip_special_tokens"], True)

        if "do_sample" in inf:
            do_sample = self._truthy(inf["do_sample"])
        elif "temperature" in inf:
            do_sample = float(inf["temperature"]) > 0
        else:
            do_sample = not greedy_default
        gen_args["do_sample"] = do_sample

        if "temperature" in inf:
            gen_args["temperature"] = float(inf["temperature"])
        elif greedy_default:
            gen_args["temperature"] = 0.0

        if not gen_args.get("temperature"):
            gen_args["do_sample"] = False
        if not gen_args.get("do_sample"):
            gen_args.pop("temperature", None)
            gen_args.pop("top_p", None)

        return gen_args

    def _generate_one_hf(
        self,
        user_text: str,
        inf: Dict[str, Any],
        *,
        greedy_default: bool,
        log_ctx: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        prompt_ids = self._encode_hf_prompt(user_text, log_ctx=log_ctx)
        prompt_len = len(prompt_ids)
        inputs = torch.tensor([prompt_ids], device=self._device, dtype=torch.long)
        attention_mask = torch.ones_like(inputs, dtype=torch.long)
        gen_kw = self._generation_kwargs(inf, greedy_default=greedy_default)
        if self._eval_debug["verbose_eval"]:
            keys = (
                "max_new_tokens",
                "max_length",
                "do_sample",
                "temperature",
                "top_k",
                "top_p",
                "repetition_penalty",
                "pad_token_id",
                "eos_token_id",
            )
            safe_kw = {k: gen_kw[k] for k in keys if k in gen_kw}
            print(
                f"[sparse_sdf_qwen3][debug] generate() ctx={log_ctx!r} kwargs_subset={safe_kw!r}",
                flush=True,
            )
        gen_config = GenerationConfig(**gen_kw)
        with torch.no_grad():
            out = self._llm.generate(
                inputs=inputs,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        new_ids = out[0, prompt_len:]
        if self._eval_debug["verbose_eval"]:
            nt = 32
            tail = new_ids.tolist()[-nt:] if nt and int(new_ids.shape[0]) > nt else new_ids.tolist()
            print(
                f"  new_token_len={int(new_ids.shape[0])} "
                f"new_ids_head={new_ids.tolist()[:nt]!r} new_ids_tail={tail!r}",
                flush=True,
            )
        return new_ids, prompt_len

    def _strip_template_stops(self, text: str) -> str:
        if self._template is not None:
            for stop in getattr(self._template, "stop_words", []) or []:
                text = text.replace(str(stop), "")
        return text.strip()

    def _bpe_encode_sparse(self, enc: Any) -> Dict[str, Any]:
        tok = self._require_bpe_tokenizer()
        return tok.encode_sparse(enc)

    def _bpe_decode_batches(self, batches: Any) -> Any:
        tok = self._require_bpe_tokenizer()
        return tok.decode_to_sparse(batches, device=self._device)

    def _bpe_vocab_size(self) -> int:
        tok = self._require_bpe_tokenizer()
        return int(tok.vocab_size)

    def _bpe_decode_ids_with_context(self, bpe_ids: Any, encoded: EncodedSparseSample) -> Any:
        """Decode generated BPE ids using anchors saved from the Encode stage."""
        ids = bpe_ids
        if hasattr(ids, "shape") and int(ids.shape[0]) != int(encoded.num_tokens_bpe):
            raise ValueError(
                "generated BPE token count does not match saved coordinate context: "
                f"got {int(ids.shape[0])}, expected {encoded.num_tokens_bpe}"
            )
        return self._bpe_decode_batches(
            [{"ids": ids, "anchors": encoded.bpe_anchors}]
        )

    def _parse_and_decode_pairs(self, raw: str) -> Tuple[Any, List[int], int]:
        """Parse Morton+mesh pairs from LLM output and decode to VQVAE indices."""
        from eval.utils.bpe_sparse_tokens import parse_bpe_sparse_token_pairs

        ids, anchors, dropped = parse_bpe_sparse_token_pairs(
            raw, max_mesh_id=self._bpe_vocab_size()
        )
        toks = [int(x) for x in ids.tolist()]
        if not toks:
            raise ValueError("no valid <morton_*><mesh_*> pairs parsed from LLM output")
        decoded_sparse = self._bpe_decode_batches([{"ids": ids, "anchors": anchors}])
        return decoded_sparse, toks, int(dropped)

    def _encode_mesh_batch(
        self,
        batch: List[MeshInput],
    ) -> Tuple[Any, Any, List[EncodedSparseSample], List[int]]:
        from eval.utils.bpe_sparse_tokens import bpe_batches_to_mesh_strings

        mc = self._model_cfg()
        res = self._sdf_resolution()
        th = self._sdf_threshold_factor()
        batch_for_collate: List[Dict[str, Any]] = []
        sdf_point_counts: List[int] = []
        for i, m in enumerate(batch):
            sparse = self._get_sdf(m)
            _require_non_empty_sparse_sdf(
                sparse,
                sample_id=str(m.sample_id),
                mesh_path=str(m.mesh_path),
                resolution=res,
                threshold_factor=th,
            )
            sparse_enc, n_tight, n_wide, enc_thresh = self._filter_sparse_for_encoder(
                sparse,
                sample_id=str(m.sample_id),
                mesh_path=str(m.mesh_path),
            )
            sdf_point_counts.append(n_wide)
            if self._eval_debug["verbose_eval"]:
                print(
                    f"[sparse_sdf_qwen3][debug] tight-band filter sample_id={m.sample_id} "
                    f"wide_points={n_wide} encoder_points={n_tight} "
                    f"threshold_abs_sdf<={enc_thresh:.6f}",
                    flush=True,
                )
            batch_for_collate.append(
                {"inputs_3d": sdf_processing.sparse_dict_to_inputs_3d(sparse_enc, i)}
            )
        collated = sdf_processing.collate_inputs_3d(batch_for_collate)
        for k in collated:
            if isinstance(collated[k], torch.Tensor):
                collated[k] = collated[k].to(self._device)
                if collated[k].is_floating_point():
                    collated[k] = collated[k].float()

        with torch.no_grad(), self._suppress_stdout():
            vae_f = self._vae.float() if next(self._vae.parameters()).dtype != torch.float32 else self._vae
            enc = vae_f.Encode(collated)
        bpe_out = self._bpe_encode_sparse(enc)
        mesh_strings = bpe_batches_to_mesh_strings(bpe_out["batches"])

        feats = enc.feats.squeeze(-1).detach().cpu().long()
        coords = enc.coords.detach().cpu()
        samples: List[EncodedSparseSample] = []
        for bi, rec in enumerate(bpe_out["batches"]):
            mask = coords[:, 0] == bi
            token_ids = feats[mask].tolist()
            coords_block = coords[mask].numpy()
            samples.append(
                EncodedSparseSample(
                    token_ids=[int(x) for x in token_ids],
                    coords_block=coords_block,
                    mesh_token_string=mesh_strings[bi],
                    bpe_ids=rec["ids"],
                    bpe_anchors=rec["anchors"],
                    num_tokens_raw=int(mask.sum().item()),
                    num_tokens_bpe=int(len(rec["ids"])),
                )
            )
        return vae_f, enc, samples, sdf_point_counts

    def encode_shape_to_tokens(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[TokenSeq]:
        _, _, encoded, sdf_point_counts = self._encode_mesh_batch(batch)
        out: List[TokenSeq] = []
        for bi, enc_sample in enumerate(encoded):
            s = enc_sample.mesh_token_string
            if self._eval_debug["verbose_eval"]:
                m = batch[bi]
                print(
                    f"[sparse_sdf_qwen3][debug] encode_shape sample_id={m.sample_id} "
                    f"sdf_sparse_points={sdf_point_counts[bi]} raw_tokens={enc_sample.num_tokens_raw} "
                    f"bpe_tokens={enc_sample.num_tokens_bpe} "
                    f"mesh_str_preview={_debug_text_preview(s)!r}",
                    flush=True,
                )
            out.append(
                TokenSeq(
                    mesh_token_string=s,
                    token_ids=[int(x) for x in enc_sample.bpe_ids.tolist()],
                    coords_xyz=enc_sample.bpe_anchors,
                    num_tokens=enc_sample.num_tokens_bpe,
                    extra={"raw_num_tokens": enc_sample.num_tokens_raw},
                )
            )
        return out

    def caption_from_shape(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        inf = cfg.get("inference", {})
        token_seqs = self.encode_shape_to_tokens(batch, cfg)
        mesh_strings = [t.mesh_token_string for t in token_seqs]
        default_p = cfg.get("data", {}).get("default_prompt", DEFAULT_CAPTION_PROMPT)
        rows: List[Dict[str, Any]] = []
        for i, m in enumerate(batch):
            pr = m.prompt or default_p
            full_user = build_sparse_understanding_user_prompt(mesh_strings[i], pr)
            log_ctx = f"understanding:{m.sample_id}"
            gen_ids, prompt_len = self._generate_one_hf(
                full_user, inf, greedy_default=True, log_ctx=log_ctx
            )
            raw = self._tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            text = self._strip_template_stops(
                self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            )
            row: Dict[str, Any] = {
                "sample_id": m.sample_id,
                "mesh_path": m.mesh_path,
                "prompt": full_user,
                "prediction": text,
                "ground_truth": m.ground_truth or "",
                "ground_truths": m.ground_truths or [m.ground_truth or ""],
                "raw_response": raw,
                "num_tokens": token_seqs[i].num_tokens,
            }
            if self._eval_debug["verbose_eval"]:
                row["debug"] = {
                    "task": "understanding",
                    "sample_id": str(m.sample_id),
                    "caption_prompt": pr,
                    "final_user_prompt": full_user,
                    "final_user_prompt_char_len": len(full_user),
                    "lm_prompt_token_len": prompt_len,
                    "mesh_token_count": token_seqs[i].num_tokens,
                    "response_token_len": int(gen_ids.shape[0]),
                }
            rows.append(row)
        return rows

    def reconstruct_mesh(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        mc = self._model_cfg()
        res = self._sdf_resolution()
        log_timing = bool(mc.get("log_timing", True))
        debug_dir = mc.get("encoded_debug_dir")
        out: List[Dict[str, Any]] = []
        import time
        import trimesh

        for m in batch:
            t0 = time.time()
            if log_timing:
                print(f"[sparse_sdf] sample={m.sample_id} 开始 SDF -> Encode -> BPE", flush=True)
            t1 = time.time()
            try:
                vae_f, _enc, encoded, sdf_counts = self._encode_mesh_batch([m])
            except Exception as exc:
                print(
                    f"[sparse_sdf] sample={m.sample_id} SDF/Encode 失败，跳过该样本: {exc}",
                    flush=True,
                )
                try:
                    gt_mesh = trimesh.load(m.mesh_path, force="mesh")
                    if not isinstance(gt_mesh, trimesh.Trimesh):
                        gt_mesh = list(gt_mesh.geometry.values())[0]
                except Exception:
                    gt_mesh = None
                out.append(
                    {
                        "sample_id": m.sample_id,
                        "mesh_path": m.mesh_path,
                        "pred_mesh": None,
                        "gt_mesh": gt_mesh,
                        "num_tokens": 0,
                        "extra": {"sdf_error": repr(exc)},
                    }
                )
                continue
            enc_sample = encoded[0]
            decoded_sparse = self._bpe_decode_ids_with_context(enc_sample.bpe_ids, enc_sample)
            with torch.no_grad(), self._suppress_stdout():
                recon = self._decode_sparse_with_pruning(vae_f, decoded_sparse)
            if debug_dir:
                import numpy as np

                p = Path(str(debug_dir))
                if not p.is_absolute():
                    p = self._repo_root() / p
                p.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    p / f"{m.sample_id}.npz",
                    raw_ids=enc_sample.token_ids,
                    raw_coords=enc_sample.coords_block,
                    bpe_ids=enc_sample.bpe_ids,
                    bpe_anchors=enc_sample.bpe_anchors,
                    sample_id=str(m.sample_id),
                    sdf_resolution=res,
                )
            if self._eval_debug["verbose_eval"]:
                try:
                    rf = getattr(recon, "feats", None)
                    rc = getattr(recon, "coords", None)
                    print(
                        f"[sparse_sdf_qwen3][debug] BPE roundtrip raw_tokens={enc_sample.num_tokens_raw} "
                        f"bpe_tokens={enc_sample.num_tokens_bpe} sdf_sparse_points={sdf_counts[0]} "
                        f"recon_feats_shape={tuple(rf.shape) if rf is not None else None} "
                        f"recon_coords_shape={tuple(rc.shape) if rc is not None else None}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[sparse_sdf_qwen3][debug] VAE tensor introspection failed: {exc!r}", flush=True)
            if log_timing:
                print(
                    f"[sparse_sdf] sample={m.sample_id} BPE/VAE 完成 raw_tokens={enc_sample.num_tokens_raw} "
                    f"bpe_tokens={enc_sample.num_tokens_bpe} "
                    f"耗时={time.time() - t1:.1f}s，开始 marching cubes",
                    flush=True,
                )
            t2 = time.time()
            with self._suppress_stdout():
                meshes = self._sparse_recon_to_meshes(recon)
            if log_timing:
                print(
                    f"[sparse_sdf] sample={m.sample_id} marching cubes 完成 "
                    f"耗时={time.time() - t2:.1f}s",
                    flush=True,
                )
            pred_mesh = meshes[0] if meshes else None
            if self._eval_debug["verbose_eval"] and pred_mesh is not None:
                try:
                    nv = int(len(pred_mesh.vertices))
                    nf = int(len(pred_mesh.faces))
                    print(
                        f"[sparse_sdf_qwen3][debug] reconstruct_mesh sample={m.sample_id} "
                        f"pred_vertices={nv} pred_faces={nf}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[sparse_sdf_qwen3][debug] reconstruct_mesh mesh stats failed: {exc!r}",
                        flush=True,
                    )
            gt_mesh = trimesh.load(m.mesh_path, force="mesh")
            if not isinstance(gt_mesh, trimesh.Trimesh):
                gt_mesh = list(gt_mesh.geometry.values())[0]
            row_extra: Dict[str, Any] = {
                "raw_num_tokens": enc_sample.num_tokens_raw,
                "bpe_num_tokens": enc_sample.num_tokens_bpe,
                "sdf_sparse_points": sdf_counts[0],
            }
            row_extra.update(_white_mesh_postprocess_record(pred_mesh))
            out.append(
                {
                    "sample_id": m.sample_id,
                    "mesh_path": m.mesh_path,
                    "pred_mesh": pred_mesh,
                    "gt_mesh": gt_mesh,
                    "num_tokens": enc_sample.num_tokens_bpe,
                    "extra": row_extra,
                }
            )
        return out

    def generate_from_text(
        self, prompts: List[str], sample_ids: List[str], cfg: Dict[str, Any]
    ) -> List[GenResult]:
        inf = cfg.get("inference", {})
        recon_prompt = resolve_reconstruct_prompt_from_inference(inf)
        results: List[GenResult] = []
        for i, sid in enumerate(sample_ids):
            caption = prompts[i]
            full_user = build_sparse_generation_user_prompt(caption, recon_prompt)
            log_ctx = f"generation:{sid}"
            if self._eval_debug["verbose_eval"]:
                print(
                    f"[sparse_sdf_qwen3][debug] generate_from_text caption_preview="
                    f"{_debug_text_preview(caption)!r} reconstruct_prompt={recon_prompt!r}",
                    flush=True,
                )
            if self._mock_llm_enabled:
                raw = self._mock_llm_raw_response()
                gen_ids = torch.tensor([], dtype=torch.long, device=self._device)
                prompt_len = 0
            else:
                gen_ids, prompt_len = self._generate_one_hf(
                    full_user, inf, greedy_default=False, log_ctx=log_ctx
                )
                raw = self._tokenizer.decode(gen_ids, skip_special_tokens=False)
            pred_mesh = None
            decode_error = ""
            toks: List[int] = []
            dropped_pairs = 0
            try:
                decoded_sparse, toks, dropped_pairs = self._parse_and_decode_pairs(raw)
                vae_f = self._vae.float() if next(self._vae.parameters()).dtype != torch.float32 else self._vae
                with torch.no_grad(), self._suppress_stdout():
                    recon = self._decode_sparse_with_pruning(vae_f, decoded_sparse)
                meshes = self._sparse_recon_to_meshes(recon)
                pred_mesh = meshes[0] if meshes else None
                if pred_mesh is None:
                    decode_error = "marching cubes produced no mesh"
            except Exception as exc:  # noqa: BLE001
                decode_error = str(exc)
            glb_tm, cextra = self._colorize_mesh(pred_mesh, str(caption))
            extra: Dict[str, Any] = {
                "sample_id": sid,
                "prompt": full_user,
                "caption": caption,
            }
            extra.update(cextra)
            extra.update(_white_mesh_postprocess_record(pred_mesh))
            if glb_tm is not None:
                extra["glb_trimesh"] = glb_tm
            if dropped_pairs:
                extra["dropped_morton_mesh_pairs"] = dropped_pairs
            if decode_error:
                extra["decode_error"] = decode_error
            if self._eval_debug["verbose_eval"]:
                extra["debug"] = {
                    "task": "generation",
                    "sample_id": sid,
                    "caption": caption,
                    "reconstruct_prompt": recon_prompt,
                    "final_user_prompt": full_user,
                    "final_user_prompt_char_len": len(full_user),
                    "lm_prompt_token_len": prompt_len,
                    "parsed_mesh_token_count": len(toks),
                    "dropped_morton_mesh_pairs": dropped_pairs,
                    "response_token_len": int(gen_ids.shape[0]),
                    "mesh_token_ids_head": toks[:32],
                    "decode_error": decode_error,
                }
            results.append(
                GenResult(
                    raw_response=raw,
                    mesh_token_ids=toks,
                    pred_mesh=pred_mesh,
                    num_occupied_voxels=int(len(pred_mesh.vertices)) if pred_mesh is not None else 0,
                    extra=extra,
                )
            )
        return results

    def generate_from_mesh_context(self, batch: List[MeshInput], cfg: Dict[str, Any]) -> List[GenResult]:
        """Generate BPE ids with the LLM, then decode using coords saved by Encode."""
        inf = cfg.get("inference", {})
        recon_prompt = resolve_reconstruct_prompt_from_inference(inf)
        vae_f, _enc, encoded, _sdf_counts = self._encode_mesh_batch(batch)
        results: List[GenResult] = []
        for i, m in enumerate(batch):
            caption = m.prompt or ""
            full_user = build_sparse_generation_user_prompt(caption, recon_prompt)
            log_ctx = f"sparse_mesh:{m.sample_id}"
            if self._mock_llm_enabled:
                raw = self._mock_llm_raw_response()
                gen_ids = torch.tensor([], dtype=torch.long, device=self._device)
                prompt_len = 0
            else:
                gen_ids, prompt_len = self._generate_one_hf(
                    full_user, inf, greedy_default=False, log_ctx=log_ctx
                )
                raw = self._tokenizer.decode(gen_ids, skip_special_tokens=False)
            pred_mesh = None
            decode_error = ""
            toks: List[int] = []
            dropped_pairs = 0
            try:
                decoded_sparse, toks, dropped_pairs = self._parse_and_decode_pairs(raw)
                with torch.no_grad(), self._suppress_stdout():
                    recon = self._decode_sparse_with_pruning(vae_f, decoded_sparse)
                meshes = self._sparse_recon_to_meshes(recon)
                pred_mesh = meshes[0] if meshes else None
                if pred_mesh is None:
                    decode_error = "marching cubes produced no mesh"
            except Exception as exc:  # noqa: BLE001
                decode_error = str(exc)
            glb_tm, cextra = self._colorize_mesh(pred_mesh, str(caption))
            extra: Dict[str, Any] = {
                "sample_id": m.sample_id,
                "prompt": full_user,
                "caption": caption,
                "raw_num_tokens": encoded[i].num_tokens_raw,
                "coord_context_bpe_tokens": encoded[i].num_tokens_bpe,
            }
            extra.update(cextra)
            extra.update(_white_mesh_postprocess_record(pred_mesh))
            if glb_tm is not None:
                extra["glb_trimesh"] = glb_tm
            if dropped_pairs:
                extra["dropped_morton_mesh_pairs"] = dropped_pairs
            if decode_error:
                extra["decode_error"] = decode_error
            if self._eval_debug["verbose_eval"]:
                extra["debug"] = {
                    "task": "sparse_mesh",
                    "sample_id": m.sample_id,
                    "caption": caption,
                    "reconstruct_prompt": recon_prompt,
                    "lm_prompt_token_len": prompt_len,
                    "parsed_mesh_token_count": len(toks),
                    "dropped_morton_mesh_pairs": dropped_pairs,
                    "response_token_len": int(gen_ids.shape[0]),
                    "decode_error": decode_error,
                }
            results.append(
                GenResult(
                    raw_response=raw,
                    mesh_token_ids=toks,
                    pred_mesh=pred_mesh,
                    num_occupied_voxels=int(len(pred_mesh.vertices)) if pred_mesh is not None else 0,
                    extra=extra,
                )
            )
        return results
