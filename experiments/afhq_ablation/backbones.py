"""Backbone factory + a thin adapter that unifies the three architectures.

The library's diffusion wrappers expect the denoising network to expose a few
attributes (``channels``, ``out_dim``, ``self_condition``,
``random_or_learned_sinusoidal_cond``) and to be callable as ``net(x, time,
self_cond)``. The three backbones differ:

- ``Unet``      forward(x, time, x_self_cond=None)
- ``KarrasUnet`` forward(x, time, self_cond=None, class_labels=None)
- ``UViT``      forward(x, time)                # only two args

``BackboneAdapter`` hides these differences. Self-conditioning is disabled for
all runs (UViT has no support for it), which keeps the comparison fair and lets
the adapter always call ``backbone(x, time)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from denoising_diffusion_pytorch import KarrasUnet, Unet
from denoising_diffusion_pytorch.simple_diffusion import UViT


class BackboneAdapter(nn.Module):
    """Uniform interface over Unet / KarrasUnet / UViT for the diffusion wrappers.

    Time-embedding contract. KarrasUnet (MP-Fourier) and UViT (learned-sinusoidal)
    embed ``time`` with a ``* 2*pi`` Fourier map that expects a *continuous log-SNR*
    signal (roughly ``[-15, 15]``); this is exactly what the library's native
    ``simple_diffusion.GaussianDiffusion`` feeds them (``model(x, log_snr)``) and
    what EDM feeds via ``0.25*log(sigma)``. But the standard ``GaussianDiffusion``
    (DDPM) passes an *integer* timestep in ``[0, num_timesteps)``. Feeding ~1000
    aliases the signal; feeding ``t/T`` in ``[0,1)`` is too compressed to resolve
    noise levels. Both cripple training.

    Fix: when ``log_snr_time`` is set (DDPM + karras/uvit) map the integer timestep
    to the schedule's log-SNR via a lookup table filled from GaussianDiffusion's
    ``alphas_cumprod`` (see ``attach_log_snr``). Unet under DDPM uses an
    integer-friendly ``SinusoidalPosEmb`` (theta=10000), so it is left untouched.
    """

    def __init__(self, backbone: nn.Module, channels: int, continuous_noise: bool,
                 log_snr_time: bool = False):
        super().__init__()
        self.backbone = backbone
        self.channels = channels
        self.out_dim = channels
        self.self_condition = False
        # GaussianDiffusion asserts this is falsy; ElucidatedDiffusion asserts truthy.
        self.random_or_learned_sinusoidal_cond = bool(continuous_noise)
        # True -> map integer DDPM timestep to schedule log-SNR before the backbone.
        self.log_snr_time = log_snr_time
        # per-timestep log-SNR lookup, filled by attach_log_snr() once the
        # diffusion (and thus the noise schedule) exists. None placeholder buffer.
        self.register_buffer("log_snr_table", None)

    def attach_log_snr(self, alphas_cumprod: torch.Tensor) -> None:
        """Build the integer-timestep -> log-SNR table from the DDPM schedule.

        log_snr(t) = log(alpha_bar_t) - log(1 - alpha_bar_t), matching the SNR the
        continuous-time backbones were designed to consume.
        """
        ac = alphas_cumprod.detach().clamp(1e-8, 1 - 1e-8)
        self.log_snr_table = (ac.log() - (1 - ac).log()).float()

    def forward(self, x, time, x_self_cond=None):
        # self-conditioning disabled everywhere -> ignore x_self_cond
        if self.log_snr_time:
            time = self.log_snr_table[time.long()]
        return self.backbone(x, time)


def build_backbone(cfg) -> BackboneAdapter:
    """Instantiate the backbone named by ``cfg.backbone`` and wrap it."""
    continuous = cfg.diffusion == "edm"
    kw = dict(cfg.backbone_kwargs)

    if cfg.backbone == "unet":
        net = Unet(
            channels=cfg.channels,
            self_condition=False,
            # EDM feeds continuous noise levels; DDPM feeds integer timesteps.
            learned_sinusoidal_cond=continuous,
            # flash off: SDPA has no fp32 kernel on this box, and sampling/FID run
            # in fp32; the einsum fallback works in every dtype. Attention at 32x32
            # is cheap, so the speed cost is negligible.
            flash_attn=False,
            **kw,
        )
    elif cfg.backbone == "karras":
        # KarrasUnet uses MP-Fourier time embedding -> continuous by construction.
        net = KarrasUnet(
            image_size=cfg.image_size,
            channels=cfg.channels,
            self_condition=False,
            attn_flash=False,
            **kw,
        )
    elif cfg.backbone == "uvit":
        # UViT uses learned-sinusoidal time embedding -> continuous by construction.
        net = UViT(
            channels=cfg.channels,
            **kw,
        )
    else:
        raise ValueError(f"unknown backbone {cfg.backbone!r}")

    # Under DDPM, karras/uvit need a continuous log-SNR time signal (their Fourier
    # embeddings expect it); the adapter maps the integer timestep via a log-SNR
    # table attached in build_diffusion. Unet under DDPM uses an integer-friendly
    # SinusoidalPosEmb, and all EDM runs feed continuous log-sigma -> no mapping.
    log_snr_time = cfg.diffusion == "ddpm" and cfg.backbone in ("karras", "uvit")

    return BackboneAdapter(net, cfg.channels, continuous, log_snr_time=log_snr_time)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def human(n: int) -> str:
    return f"{n / 1e6:.2f}M"
