"""Diffusion-wrapper factory.

Both wrappers consume images in [0, 1] and return samples in [0, 1], so the data
pipeline and FID/grid code need no per-formula branching.
"""

from __future__ import annotations

import torch.nn as nn

from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.elucidated_diffusion import ElucidatedDiffusion


def build_diffusion(cfg, model: nn.Module) -> nn.Module:
    if cfg.diffusion == "ddpm":
        diff = GaussianDiffusion(
            model,
            image_size=cfg.image_size,
            timesteps=cfg.timesteps,
            sampling_timesteps=cfg.sampling_timesteps,
            objective=cfg.objective,
            beta_schedule=cfg.beta_schedule,
        )
        # karras/uvit consume a continuous log-SNR time signal; fill the adapter's
        # lookup table from this schedule's alphas_cumprod (no-op for unet).
        if getattr(model, "log_snr_time", False):
            model.attach_log_snr(diff.alphas_cumprod)
        return diff
    elif cfg.diffusion == "edm":
        return ElucidatedDiffusion(
            model,
            image_size=cfg.image_size,
            channels=cfg.channels,
            num_sample_steps=cfg.num_sample_steps,
        )
    else:
        raise ValueError(f"unknown diffusion {cfg.diffusion!r}")
