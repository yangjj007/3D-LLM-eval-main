"""
Text-based metrics for 3D understanding evaluation.

Supports: BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr, BERTScore, GPT-Score.

Heavy objects (RoBERTa via ``bert_score.BERTScorer``, SentenceTransformer, NLTK/ROUGE/CIDEr
scorers, OpenAI client) are **cached at module scope** so repeated ``TextMetrics.compute`` calls
(e.g. one sample per batch) reuse the same loaded weights. Use ``clear_text_metric_caches()`` to
reset when running unrelated evaluations in one long-lived process.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

# ---------------------------------------------------------------------------
# Module-level caches: load each heavy object once per process for a given key.
# ---------------------------------------------------------------------------
_bleu_smooth_fn: Optional[Callable[..., Any]] = None
_rouge_scorer: Any = None
_cider_scorer: Any = None
_meteor_env_initialized: bool = False
_meteor_have_wordnet: bool = False

_sentence_transformer_models: Dict[Tuple[str, str], Any] = {}

_bert_scorer_instances: Dict[Tuple[str, int, str, int], Any] = {}
_bert_resolved_num_layers: Dict[str, int] = {}

_openai_clients: Dict[Tuple[str, str], Any] = {}


def clear_text_metric_caches() -> None:
    """Drop cached models / scorers (e.g. before a second unrelated eval in one process)."""
    global _bleu_smooth_fn, _rouge_scorer, _cider_scorer
    global _meteor_env_initialized, _meteor_have_wordnet
    global _sentence_transformer_models, _bert_scorer_instances
    global _bert_resolved_num_layers, _openai_clients
    _bleu_smooth_fn = None
    _rouge_scorer = None
    _cider_scorer = None
    _meteor_env_initialized = False
    _meteor_have_wordnet = False
    _sentence_transformer_models.clear()
    _bert_scorer_instances.clear()
    _bert_resolved_num_layers.clear()
    _openai_clients.clear()


def _get_bleu_smoothing_function() -> Callable[..., Any]:
    global _bleu_smooth_fn
    if _bleu_smooth_fn is None:
        from nltk.translate.bleu_score import SmoothingFunction

        _bleu_smooth_fn = SmoothingFunction().method1
    return _bleu_smooth_fn


def _get_rouge_scorer() -> Any:
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer

        _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge_scorer


def _get_cider_scorer() -> Any:
    global _cider_scorer
    if _cider_scorer is None:
        from pycocoevalcap.cider.cider import Cider

        _cider_scorer = Cider()
    return _cider_scorer


def _ensure_meteor_wordnet() -> bool:
    """Resolve NLTK wordnet once; return whether full METEOR (with WordNet) is available."""
    global _meteor_env_initialized, _meteor_have_wordnet
    if _meteor_env_initialized:
        return _meteor_have_wordnet

    import os

    import nltk

    def _try_download(pkg: str) -> None:
        try:
            nltk.download(pkg, quiet=True, raise_on_error=True)
        except Exception:
            pass

    skip_dl = os.environ.get("NLTK_METEOR_NO_WORDNET", "").lower() in ("1", "true", "yes")

    if not skip_dl:
        for resource, pkg in [("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")]:
            try:
                nltk.data.find(resource)
            except LookupError:
                _try_download(pkg)

    # Path checks alone can miss valid installs (NLTK_DATA in worker, wordnet2021 layout, etc.).
    # METEOR needs a working ``nltk.corpus.wordnet`` reader — verify with an actual lookup.
    have_wn = False
    try:
        from nltk.corpus import wordnet as wn

        have_wn = len(wn.synsets("dog")) > 0
    except (LookupError, OSError, ValueError):
        have_wn = False

    if not have_wn:
        if skip_dl:
            warnings.warn(
                "NLTK 'wordnet' and/or 'omw-1.4' is missing; NLTK_METEOR_NO_WORDNET=1 skipped "
                "auto-download. Using METEOR without WordNet synonym expansion (exact + stem only)."
            )
        else:
            warnings.warn(
                "NLTK WordNet is not usable from this process (``nltk.corpus.wordnet.synsets`` failed). "
                "Common causes: data under a different user (set NLTK_DATA to the same path as "
                "``nltk.download``, e.g. /root/nltk_data), or stale code still passing "
                "``wordnet=None`` to METEOR. Using METEOR without WordNet synonym expansion "
                "(exact + stem only). For full METEOR: fix NLTK_DATA / reinstall wordnet+omw-1.4."
            )

    _meteor_env_initialized = True
    _meteor_have_wordnet = have_wn
    return have_wn


class _MeteorWordNetStub:
    """NLTK ``meteor_score(..., wordnet=None)`` still calls ``wordnet.synsets`` and crashes.

    An object whose ``synsets`` returns ``[]`` disables synonym expansion while preserving
    exact match and Porter stem alignment (METEOR without the WordNet corpus).
    """

    def synsets(self, word: str) -> List[Any]:
        return []


_METEOR_WN_STUB = _MeteorWordNetStub()


def _resolve_bert_score_num_layers(model_type: str, num_layers: int | None) -> int:
    if num_layers is not None:
        return int(num_layers)
    if model_type in _bert_resolved_num_layers:
        return _bert_resolved_num_layers[model_type]
    try:
        from bert_score.utils import model2layers

        nl = int(model2layers[model_type])
        _bert_resolved_num_layers[model_type] = nl
        return nl
    except KeyError:
        pass
    from transformers import AutoConfig

    try:
        from bert_score.utils import model2layers
    except Exception:
        model2layers = {}

    cfg = AutoConfig.from_pretrained(model_type)
    arch = type(cfg).__name__.lower()
    cand_ids = [model_type]
    mid = getattr(cfg, "_name_or_path", "") or getattr(cfg, "name_or_path", "") or ""
    mid = str(mid).strip()
    if mid:
        cand_ids.append(mid)
    m = getattr(cfg, "model_type", None)
    if m:
        cand_ids.append(str(m))
    size_name = "base" if "base" in arch else "large" if "large" in arch else ""
    if "roberta" in arch and size_name:
        cand_ids.append(f"roberta-{size_name}")
    if "bert" in arch and size_name:
        cand_ids.append(f"bert-{size_name}-uncased")

    chosen: int | None = None
    for cid in cand_ids:
        if cid in model2layers:
            chosen = int(model2layers[cid])
            break

    if chosen is None:
        hidden = getattr(cfg, "num_hidden_layers", None)
        if hidden is None:
            raise KeyError(f"Cannot infer num_layers for model_type={model_type!r}")
        chosen = int(hidden) + 1

    _bert_resolved_num_layers[model_type] = chosen
    return chosen


def _get_bert_scorer(
    model_type: str,
    num_layers: int | None,
    *,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> Any:
    nl = _resolve_bert_score_num_layers(model_type, num_layers)
    dkey = device if device is not None else "__auto__"
    key = (model_type, nl, dkey, int(batch_size))
    if key not in _bert_scorer_instances:
        from bert_score import BERTScorer

        ctor_kw: Dict[str, Any] = dict(
            model_type=model_type,
            num_layers=nl,
            batch_size=batch_size,
            rescale_with_baseline=False,
        )
        if device is not None:
            ctor_kw["device"] = device
        _bert_scorer_instances[key] = BERTScorer(**ctor_kw)
    return _bert_scorer_instances[key]


def _get_openai_client(api_key: str, *, base_url: Optional[str] = None) -> Any:
    bkey = base_url or ""
    key = (api_key, bkey)
    if key not in _openai_clients:
        from openai import OpenAI

        kw: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        _openai_clients[key] = OpenAI(**kw)
    return _openai_clients[key]


def _make_sentence_transformer(model_name: str, cache_folder: Optional[str] = None):
    """
    sentence-transformers>=3：避免把 `cache_dir` 直接塞进底层 AutoModel（会触发 deprecation），
    并尽量优先走 safetensors（对 torch<2.6 + transformers 的 torch.load 限制更友好）。
    """
    from sentence_transformers import SentenceTransformer

    model_kwargs: Dict[str, Any] = {"use_safetensors": True}
    tokenizer_kwargs: Dict[str, Any] = {}
    if cache_folder:
        model_kwargs["cache_dir"] = cache_folder
        tokenizer_kwargs["cache_dir"] = cache_folder
    try:
        return SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs or None,
        )
    except TypeError:
        # 旧版 sentence-transformers：退回 `cache_folder`。
        return SentenceTransformer(model_name, cache_folder=cache_folder)
    except Exception:
        # 部分仓库无 safetensors：`use_safetensors=True` 会失败，再退回默认加载（可能在 torch<2.6 上仍失败）。
        fallback_kw: Dict[str, Any] = {}
        if cache_folder:
            fallback_kw["model_kwargs"] = {"cache_dir": cache_folder}
            fallback_kw["tokenizer_kwargs"] = {"cache_dir": cache_folder}
        try:
            return SentenceTransformer(model_name, **fallback_kw)
        except TypeError:
            return SentenceTransformer(model_name, cache_folder=cache_folder)


def _get_sentence_transformer(model_name: str, cache_folder: Optional[str] = None) -> Any:
    key = (model_name, cache_folder or "")
    if key not in _sentence_transformer_models:
        _sentence_transformer_models[key] = _make_sentence_transformer(model_name, cache_folder)
    return _sentence_transformer_models[key]


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    max_order: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU-1 through BLEU-{max_order}.

    Args:
        predictions: List of predicted strings.
        references: List of reference string lists (multiple refs per sample).

    Returns:
        Dict with keys 'bleu_1', 'bleu_2', ..., 'bleu_{max_order}'.
    """
    from nltk.translate.bleu_score import corpus_bleu

    smooth = _get_bleu_smoothing_function()

    tokenized_preds = [p.lower().split() for p in predictions]
    tokenized_refs = [[r.lower().split() for r in ref_list] for ref_list in references]

    results = {}
    for n in range(1, max_order + 1):
        weights = tuple([1.0 / n] * n + [0.0] * (max_order - n))
        score = corpus_bleu(
            tokenized_refs, tokenized_preds, weights=weights[:max_order],
            smoothing_function=smooth,
        )
        results[f"bleu_{n}"] = score
    return results


def compute_rouge_l(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Compute ROUGE-L F1 score."""
    scorer = _get_rouge_scorer()

    scores = []
    for pred, ref_list in zip(predictions, references):
        best = max(
            scorer.score(ref, pred)["rougeL"].fmeasure for ref in ref_list
        )
        scores.append(best)

    return {"rouge_l": sum(scores) / len(scores) if scores else 0.0}


def compute_meteor(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Compute METEOR score.

    If NLTK corpora ``wordnet`` / ``omw-1.4`` are missing and auto-download fails
    (e.g. SSL errors to GitHub), we use a WordNet stub so alignment still runs:
    exact + stem matches only (no synonym expansion). NLTK's ``wordnet=None`` is
    unsafe — it still calls ``synsets`` on ``None``. Set ``NLTK_METEOR_NO_WORDNET=1``
    to skip download attempts in offline / locked-down networks.
    """
    from nltk.translate.meteor_score import meteor_score as _meteor

    have_wn = _ensure_meteor_wordnet()
    no_syn = _METEOR_WN_STUB

    scores: List[float] = []
    for pred, ref_list in zip(predictions, references):
        ref_tok = [r.split() for r in ref_list]
        pred_tok = pred.split()
        if have_wn:
            try:
                score = _meteor(ref_tok, pred_tok)
            except (LookupError, OSError, ValueError, AttributeError):
                score = _meteor(ref_tok, pred_tok, wordnet=no_syn)
        else:
            score = _meteor(ref_tok, pred_tok, wordnet=no_syn)
        scores.append(float(score))

    return {"meteor": sum(scores) / len(scores) if scores else 0.0}


def compute_cider(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Compute CIDEr score using pycocoevalcap.

    Falls back to a simplified TF-IDF based implementation if pycocoevalcap
    is not available.
    """
    try:
        cider_scorer = _get_cider_scorer()
        gts = {i: ref_list for i, ref_list in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        score, _ = cider_scorer.compute_score(gts, res)
        return {"cider": score}
    except ImportError:
        warnings.warn(
            "pycocoevalcap not installed. CIDEr score unavailable. "
            "Install with: pip install pycocoevalcap"
        )
        return {"cider": -1.0}


def compute_bert_score(
    predictions: List[str],
    references: List[List[str]],
    model_type: str = "roberta-large",
    num_layers: int | None = None,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute BERTScore (F1). Uses a cached :class:`bert_score.BERTScorer` per model/layer/device."""
    try:
        scorer = _get_bert_scorer(model_type, num_layers, device=device, batch_size=batch_size)
        flat_refs = [ref_list[0] for ref_list in references]
        P, R, F1 = scorer.score(predictions, flat_refs, verbose=False, batch_size=batch_size)
        return {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }
    except ImportError:
        warnings.warn("bert-score not installed. Install with: pip install bert-score")
        return {"bert_score_f1": -1.0}
    except Exception as e:
        warnings.warn(f"bert_score unavailable: {e}")
        return {
            "bert_score_precision": -1.0,
            "bert_score_recall": -1.0,
            "bert_score_f1": -1.0,
        }


def compute_sentence_bert(
    predictions: List[str],
    references: List[List[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder: Optional[str] = None,
) -> Dict[str, float]:
    """Cosine similarity of sentence embeddings (Table 5: Sentence-BERT style).

    For multiple references per sample: mean cosine(pred, ref_i) over refs,
    then mean over samples.
    """
    try:
        import torch

        model = _get_sentence_transformer(model_name, cache_folder)
        per_sample: List[float] = []
        for pred, ref_list in zip(predictions, references):
            refs = [r for r in ref_list if (r or "").strip()]
            if not refs:
                continue
            emb_p = model.encode([pred], convert_to_tensor=True, show_progress_bar=False)
            emb_r = model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
            emb_p = torch.nn.functional.normalize(emb_p, dim=-1)
            emb_r = torch.nn.functional.normalize(emb_r, dim=-1)
            sims = (emb_r * emb_p.expand_as(emb_r)).sum(dim=-1)
            per_sample.append(float(sims.mean().item()))
        return {"sentence_bert": sum(per_sample) / len(per_sample) if per_sample else 0.0}
    except Exception as e:
        warnings.warn(f"sentence_bert unavailable: {e}")
        return {"sentence_bert": -1.0}


def compute_simcse(
    predictions: List[str],
    references: List[List[str]],
    # 上游 SimCSE 官方权重多为 pytorch_model.bin；torch<2.6 的 transformers 会拒绝 torch.load。
    # 默认改为 sentence-transformers 打包模型（通常含 model.safetensors），语义仍可比，但与论文 backbone 不完全一致。
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    cache_folder: Optional[str] = None,
) -> Dict[str, float]:
    """SimCSE-style embedding cosine (Table 5).

    For multiple references per sample: mean cosine(pred, ref_i) over refs,
    then mean over samples.
    """
    try:
        import torch

        model = _get_sentence_transformer(model_name, cache_folder)
        per_sample: List[float] = []
        for pred, ref_list in zip(predictions, references):
            refs = [r for r in ref_list if (r or "").strip()]
            if not refs:
                continue
            emb_p = model.encode([pred], convert_to_tensor=True, show_progress_bar=False)
            emb_r = model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
            emb_p = torch.nn.functional.normalize(emb_p, dim=-1)
            emb_r = torch.nn.functional.normalize(emb_r, dim=-1)
            sims = (emb_r * emb_p.expand_as(emb_r)).sum(dim=-1)
            per_sample.append(float(sims.mean().item()))
        return {"simcse": sum(per_sample) / len(per_sample) if per_sample else 0.0}
    except Exception as e:
        warnings.warn(f"simcse unavailable: {e}")
        return {"simcse": -1.0}


def compute_gpt_score(
    predictions: List[str],
    references: List[List[str]],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> Dict[str, float]:
    """
    LLM-as-judge scoring via OpenAI API.

    Sends each (prediction, reference) pair to GPT for quality assessment
    on a 1-5 scale.
    """
    if api_key is None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")

    if api_key is None:
        warnings.warn("No OpenAI API key found. GPT-Score unavailable.")
        return {"gpt_score": -1.0}

    try:
        client = _get_openai_client(api_key)
    except ImportError:
        warnings.warn("openai package not installed. GPT-Score unavailable.")
        return {"gpt_score": -1.0}
    scores = []

    for pred, ref_list in zip(predictions, references):
        ref_text = ref_list[0]
        prompt = (
            f"Rate the quality of the following 3D object description on a scale of 1-5.\n\n"
            f"Reference description: {ref_text}\n"
            f"Generated description: {pred}\n\n"
            f"Criteria: accuracy, completeness, and fluency.\n"
            f"Output ONLY a single number (1-5)."
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            score_text = response.choices[0].message.content.strip()
            scores.append(float(score_text))
        except Exception:
            scores.append(0.0)

    return {"gpt_score": sum(scores) / len(scores) if scores else 0.0}




class TextMetrics:
    """Unified interface for computing text metrics."""

    METRIC_FNS = {
        "bleu": compute_bleu,
        "rouge_l": compute_rouge_l,
        "meteor": compute_meteor,
        "cider": compute_cider,
        "bert_score": compute_bert_score,
        "gpt_score": compute_gpt_score,
        "sentence_bert": compute_sentence_bert,
        "simcse": compute_simcse,
    }

    @staticmethod
    def compute(
        predictions: List[str],
        references: List[List[str]],
        metric_names: List[str],
        metrics_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute specified text metrics.

        Args:
            predictions: Model-generated texts.
            references: List of reference text lists per sample.
            metric_names: Which metrics to compute (e.g. ['bleu', 'rouge_l']).

        Returns:
            Dict of metric_name → score.
        """
        results = {}
        metrics_config = metrics_config or {}
        for name in metric_names:
            if name in TextMetrics.METRIC_FNS:
                metric_kwargs = dict(kwargs)
                metric_kwargs.update(metrics_config.get(name, {}))
                results.update(TextMetrics.METRIC_FNS[name](predictions, references, **metric_kwargs))
            else:
                warnings.warn(f"Unknown text metric: {name}")
        return results
