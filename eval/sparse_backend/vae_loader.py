"""Load SparseSDFVQVAE from the Med training JSON config and checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_existing_file(path: str | Path | None, *, kind: str) -> Path:
    if path is None or str(path).strip() == "":
        raise ValueError(f"{kind} path is required")

    raw = Path(str(path)).expanduser()
    if raw.is_absolute():
        candidates = [raw]
    else:
        root = _repo_root()
        candidates = [
            Path.cwd() / raw,
            root / raw,
            root.parent / raw,
        ]

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in checked:
            checked.append(resolved)
        if resolved.is_file():
            return resolved

    checked_text = "\n  - ".join(str(p) for p in checked)
    raise FileNotFoundError(
        f"{kind} file not found: {path}\n"
        f"Checked:\n  - {checked_text}\n"
        "Set model.vae_config/model.vae_ckpt to the 256 sparse SDF VQ-VAE files."
    )


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "vae"):
            nested = checkpoint.get(key)
            if isinstance(nested, Mapping):
                if isinstance(nested.get("state_dict"), Mapping):
                    return _strip_module_prefix(nested["state_dict"])
                return _strip_module_prefix(nested)
        return _strip_module_prefix(checkpoint)
    raise TypeError(
        f"Unsupported VAE checkpoint type: {type(checkpoint)!r}; expected a state dict or dict wrapper"
    )


def _align_vae_args_vq_group_size_to_checkpoint(vae_args: dict[str, Any], state_dict: Mapping[str, Any]) -> None:
    emb_key = "vq.embeddings.weight"
    if emb_key not in state_dict or not isinstance(state_dict[emb_key], torch.Tensor):
        return
    ckpt_emb_dim = int(state_dict[emb_key].shape[1])
    embed_dim = int(vae_args.get("embed_dim") or vae_args.get("latent_channels") or 0)
    if embed_dim <= 0 or ckpt_emb_dim % embed_dim != 0:
        return
    inferred = ckpt_emb_dim // embed_dim
    current = int(vae_args.get("vq_group_size", 1))
    if inferred != current:
        print(
            f"[load_vae] vq_group_size mismatch: config={current}, "
            f"checkpoint={inferred}; using checkpoint value",
            flush=True,
        )
        vae_args["vq_group_size"] = inferred


def _split_prefixed_state_dict(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    marker = f"{prefix}."
    return {
        key[len(marker) :]: value
        for key, value in state_dict.items()
        if key.startswith(marker)
    }


def _load_state_dict_with_med_fallback(model: torch.nn.Module, state_dict: Mapping[str, Any]) -> None:
    try:
        incompatible = model.load_state_dict(dict(state_dict), strict=False)
        missing = len(getattr(incompatible, "missing_keys", []) or [])
        unexpected = len(getattr(incompatible, "unexpected_keys", []) or [])
        print(
            f"[load_vae] Loaded checkpoint with torch.load_state_dict "
            f"(missing={missing}, unexpected={unexpected})",
            flush=True,
        )
        return
    except RuntimeError as exc:
        print(
            "[load_vae] Direct state_dict load failed; falling back to "
            f"SparseSDFVQVAE.load_pretrained_vae: {exc}",
            flush=True,
        )

    if not hasattr(model, "load_pretrained_vae"):
        raise RuntimeError("VAE model has no load_pretrained_vae fallback")

    enc = _split_prefixed_state_dict(state_dict, "encoder")
    dec = _split_prefixed_state_dict(state_dict, "decoder")
    vq = _split_prefixed_state_dict(state_dict, "vq")
    if not enc and not dec and not vq:
        raise RuntimeError(
            "Checkpoint does not contain encoder./decoder./vq. keys for fallback loading"
        )
    model.load_pretrained_vae(enc, dec, vq or None)


def load_vae_from_config(
    vae_config_path: str,
    vae_ckpt_path: Optional[str],
    device: torch.device | str,
) -> torch.nn.Module:
    from trellis.models import SparseSDFVQVAE

    config_path = _resolve_existing_file(vae_config_path, kind="VAE config")
    ckpt_path = _resolve_existing_file(vae_ckpt_path, kind="VAE checkpoint")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    vae_args = dict(config["models"]["vqvae"]["args"])

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_for_load = _extract_state_dict(checkpoint)
    _align_vae_args_vq_group_size_to_checkpoint(vae_args, state_for_load)

    model = SparseSDFVQVAE(**vae_args)
    _load_state_dict_with_med_fallback(model, state_for_load)
    print(f"[load_vae] Loaded VAE weights from {ckpt_path}", flush=True)

    model = model.to(device)
    model.eval()
    return model
