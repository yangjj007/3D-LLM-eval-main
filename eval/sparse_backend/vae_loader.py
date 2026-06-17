"""Load SparseSDFVQVAE from JSON config + .pt checkpoint (vendored from Med training toolkit)."""

from __future__ import annotations

import json
import os
from typing import Optional

import torch


def _align_vae_args_vq_group_size_to_checkpoint(vae_args: dict, state_dict: dict) -> None:
    emb_key = "vq.embeddings.weight"
    if emb_key not in state_dict:
        return
    ckpt_emb_dim: int = state_dict[emb_key].shape[1]
    embed_dim: int = int(
        vae_args.get("embed_dim") or vae_args.get("latent_channels") or 0
    )
    if embed_dim <= 0 or ckpt_emb_dim % embed_dim != 0:
        return
    inferred = ckpt_emb_dim // embed_dim
    current = int(vae_args.get("vq_group_size", 8))
    if inferred != current:
        print(
            f"[load_vae] vq_group_size mismatch: config={current},"
            f" checkpoint={inferred} → using checkpoint value",
            flush=True,
        )
        vae_args["vq_group_size"] = inferred


def load_vae_from_config(
    vae_config_path: str,
    vae_ckpt_path: Optional[str],
    device: torch.device,
) -> torch.nn.Module:
    from trellis.models import SparseSDFVQVAE

    with open(vae_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    vae_args = dict(config["models"]["vqvae"]["args"])
    state_for_load = None
    if vae_ckpt_path and os.path.isfile(vae_ckpt_path):
        # 显式 weights_only=False：训练侧 checkpoint 常含非张量元数据；本地受信任权重。
        ckpt_raw = torch.load(
            vae_ckpt_path, map_location="cpu", weights_only=False
        )
        state_for_load = (
            ckpt_raw["state_dict"]
            if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw
            else ckpt_raw
        )
        if isinstance(state_for_load, dict):
            _align_vae_args_vq_group_size_to_checkpoint(vae_args, state_for_load)
    model = SparseSDFVQVAE(**vae_args)
    if state_for_load is not None:
        model.load_state_dict(state_for_load, strict=False)
        print(f"[load_vae] Loaded VAE weights from {vae_ckpt_path}")
    else:
        print("[load_vae] WARNING: no valid vae_ckpt; randomly initialized VAE (debug only).")
    model = model.to(device)
    model.eval()
    return model
