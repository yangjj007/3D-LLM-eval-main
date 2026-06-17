"""Run 3DTopia-XL text-to-3D with official text conditioner components.

The upstream repository ships ``configs/inference_dit_text.yml`` and
``models/conditioner/text.py``. Its README says to use the text config and
update ``inference.py`` from image encoding to text encoding. This helper keeps
that change in the eval wrapper while still using the official model, VAE,
diffusion, text conditioner, and mesh extraction code from the cloned repo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _resolve(path: str, repo_dir: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else repo_dir / p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--config", default="configs/inference_dit_text.yml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--vae-checkpoint-path", default=None)
    parser.add_argument("--text-encoder-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--mc-resolution", type=int, default=256)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    sys.path.insert(0, str(repo_dir))

    import torch
    import open_clip
    from omegaconf import OmegaConf
    from dva.io import load_from_config
    from dva.ray_marcher import RayMarcher
    from inference import extract_texmesh
    from models.diffusion import create_diffusion

    config = OmegaConf.load(_resolve(args.config, repo_dir))
    config.output_dir = str(Path(args.output_dir).resolve())
    config.inference.seed = int(args.seed)
    config.inference.ddim = int(args.ddim)
    config.inference.cfg = float(args.cfg)
    config.inference.export_glb = True
    config.inference.mc_resolution = int(args.mc_resolution)
    config.inference.batch_size = int(config.inference.get("batch_size", 8192))

    if args.checkpoint_path:
        config.checkpoint_path = _resolve(args.checkpoint_path, repo_dir)
    else:
        config.checkpoint_path = _resolve(str(config.checkpoint_path), repo_dir)
    if args.vae_checkpoint_path:
        config.model.vae_checkpoint_path = _resolve(args.vae_checkpoint_path, repo_dir)
    else:
        config.model.vae_checkpoint_path = _resolve(str(config.model.vae_checkpoint_path), repo_dir)
    if args.text_encoder_path:
        config.model.conditioner.encoder_config.pretrained_path = _resolve(args.text_encoder_path, repo_dir)
    else:
        config.model.conditioner.encoder_config.pretrained_path = _resolve(
            str(config.model.conditioner.encoder_config.pretrained_path),
            repo_dir,
        )

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(int(config.inference.seed))

    amp = config.inference.get("precision", "fp16") == "fp16" and device.type == "cuda"
    precision_dtype = torch.float16 if amp else torch.float32

    model = load_from_config(config.model.generator)
    state_dict = torch.load(config.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict.get("ema", state_dict))
    model = model.to(device).eval()

    vae = load_from_config(config.model.vae)
    vae_state_dict = torch.load(config.model.vae_checkpoint_path, map_location="cpu")
    vae.load_state_dict(vae_state_dict.get("model_state_dict", vae_state_dict))
    vae = vae.to(device).eval()

    conditioner = load_from_config(config.model.conditioner).to(device).eval()
    tokenizer = open_clip.get_tokenizer(config.model.conditioner.encoder_config.get("model_spec", "ViT-L-14"))
    caption_token = tokenizer([args.prompt]).to(device)

    rm = RayMarcher(config.image_height, config.image_width, **config.rm).to(device)

    perchannel_norm = "latent_mean" in config.model
    if perchannel_norm:
        latent_mean = torch.tensor(config.model.latent_mean, dtype=torch.float32, device=device)[None, None, :]
        latent_std = torch.tensor(config.model.latent_std, dtype=torch.float32, device=device)[None, None, :]
    latent_nf = float(config.model.latent_nf)

    diffusion_cfg = OmegaConf.to_container(config.diffusion, resolve=True)
    diffusion_cfg.pop("timestep_respacing", None)
    diffusion = create_diffusion(timestep_respacing=f"ddim{int(config.inference.ddim)}", **diffusion_cfg)
    sample_fn = diffusion.ddim_sample_loop_progressive
    fwd_fn = model.forward_with_cfg if float(config.inference.cfg) > 0 else model.forward

    with torch.no_grad():
        latent_shape = (1, int(config.model.num_prims), 1, 4, 4, 4)
        latent = torch.randn(*latent_shape, device=device)
        inf_x = torch.randn(1, int(config.model.num_prims), int(config.model.generator.in_channels), device=device)
        y = conditioner.encoder(caption_token)
        model_kwargs = {"y": y[:1], "precision_dtype": precision_dtype, "enable_amp": amp}
        if float(config.inference.cfg) > 0:
            model_kwargs["cfg_scale"] = float(config.inference.cfg)
        final_samples = None
        for samples in sample_fn(
            fwd_fn,
            inf_x.shape,
            inf_x,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        ):
            final_samples = samples
        if final_samples is None:
            raise RuntimeError("3DTopia-XL diffusion produced no samples")
        recon_param = final_samples["sample"].reshape(1, int(config.model.num_prims), -1)
        if perchannel_norm:
            recon_param = recon_param / latent_nf * latent_std + latent_mean
        recon_srt_param = recon_param[:, :, 0:4]
        recon_feat_param = recon_param[:, :, 4:]
        decoded_chunks = []
        for bidx in range(recon_feat_param.shape[0]):
            feat = recon_feat_param[bidx].reshape(int(config.model.num_prims), *latent.shape[-4:])
            if not perchannel_norm:
                feat = feat / latent_nf
            decoded_chunks.append(vae.decode(feat).detach())
        recon_feat_param = torch.cat(decoded_chunks, dim=0)
        if not perchannel_norm:
            recon_srt_param[:, :, 0:1] = (recon_srt_param[:, :, 0:1] / 10) + 0.05
        recon_feat_param[:, 0:1, ...] /= 5.0
        recon_feat_param[:, 1:, ...] = (recon_feat_param[:, 1:, ...] + 1) / 2.0
        recon_feat_param = recon_feat_param.reshape(1, int(config.model.num_prims), -1)

    prim_params = {
        "srt_param": recon_srt_param[0].detach().cpu(),
        "feat_param": recon_feat_param[0].detach().cpu(),
    }

    prim_cfg = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
    for key in ("vae", "vae_checkpoint_path", "conditioner", "generator", "latent_nf", "latent_mean", "latent_std"):
        if key in prim_cfg:
            prim_cfg.pop(key)
    model_primx = load_from_config(prim_cfg)
    model_primx.load_state_dict(prim_params)
    model_primx.to(device).eval()
    with torch.no_grad():
        model_primx.srt_param[:, 1:4] *= 0.85
        extract_texmesh(config.inference, model_primx, config.output_dir, device)

    print(Path(config.output_dir) / "pbr_mesh.glb")


if __name__ == "__main__":
    main()
