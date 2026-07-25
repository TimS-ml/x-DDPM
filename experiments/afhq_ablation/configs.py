"""Experiment configs for the AFHQv2 cat+dog backbone/diffusion ablation.

Each run is a named ``ExperimentConfig``. Backbone-specific arguments live in
``backbone_kwargs`` so the three architectures (which take different constructor
signatures) stay decoupled from the shared training/eval knobs. First AFHQ
campaign is unconditional; see the autonomy log for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class ExperimentConfig:
    name: str
    backbone: str                 # 'unet' | 'karras' | 'uvit'
    diffusion: str                # 'ddpm' | 'edm'
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)

    # data / model
    image_size: int = 64
    channels: int = 3

    # gaussian diffusion (ddpm)
    timesteps: int = 1000
    sampling_timesteps: int = 100   # DDIM steps for fast sampling
    objective: str = "pred_v"
    beta_schedule: str = "sigmoid"

    # elucidated diffusion (edm)
    num_sample_steps: int = 32

    # training
    train_steps: int = 80_000
    batch_size: int = 64            # 64x64 -> 4x activation vs CIFAR; halved from 128
    lr: float = 2e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    ema_update_every: int = 10
    amp_dtype: str = "bf16"         # 'bf16' | 'fp16' | 'fp32'

    # logging / eval
    log_every: int = 50
    eval_every: int = 10_000
    fid_samples: int = 5_000
    fid_batch_size: int = 64        # sampling spikes VRAM at 64x64; keep modest
    sample_grid: int = 64           # perfect square -> 8x8 grid

    # infra
    seed: int = 0
    wandb_project: str = "x-ddpm-afhq-ablation"

    def smoke(self) -> "ExperimentConfig":
        """Return a fast, low-fidelity variant for pipeline validation."""
        return replace(
            self,
            train_steps=300,
            eval_every=150,
            fid_samples=250,
            fid_batch_size=32,
            sample_grid=16,
        )


# shared knobs applied to every run
_COMMON: dict[str, Any] = dict(
    image_size=64,
    channels=3,
    batch_size=64,
    lr=2e-4,
    train_steps=80_000,
    eval_every=10_000,
)

# per-backbone constructor args, tuned for ~26-29M params at 64x64 (spread 8.8%).
# dims are refined against experiments/.../backbones.py:count_params in the
# param-table step; edit here if the table is off by >10%.
# karras: dropout=0.0 overrides KarrasUnet's default 0.1 to match Unet/UViT (0)
# for a fair fixed-step convergence comparison.
_BACKBONE_KWARGS: dict[str, dict[str, Any]] = {
    "unet": dict(dim=108, dim_mults=(1, 2, 4), attn_dim_head=32, attn_heads=4),
    "karras": dict(dim=72, dim_max=192, num_downsamples=4, num_blocks_per_stage=2,
                   attn_res=(16, 8), attn_dim_head=32, dropout=0.0),
    "uvit": dict(dim=96, dim_mults=(1, 2, 4), vit_depth=8, attn_dim_head=32, attn_heads=4),
}

# per-backbone learning rate. KarrasUnet is magnitude-preserving: every conv/linear
# forces per-row weight norm to sqrt(fan_in), so only filter *direction* updates and
# its functional LR is ~sqrt(fan_in) (~15x) lower than a standard net at the same
# nominal LR. At Unet's 2e-4 it learns ~25x too slowly (FID ~85 vs 20 in both DDPM
# and EDM). The architecture ships InvSqrtDecayLRSched with peak ~1e-2; 3e-3
# (= 2e-4 x median sqrt(fan_in)) is the matched constant LR. Unet/UViT are not MP,
# so they keep 2e-4.
_BACKBONE_LR: dict[str, float] = {
    "unet": 2e-4,
    "karras": 3e-3,
    "uvit": 2e-4,
}


def _make(name: str, backbone: str, diffusion: str) -> ExperimentConfig:
    common = dict(_COMMON)
    common["lr"] = _BACKBONE_LR[backbone]
    return ExperimentConfig(
        name=name,
        backbone=backbone,
        diffusion=diffusion,
        backbone_kwargs=dict(_BACKBONE_KWARGS[backbone]),
        **common,
    )


# ordered by priority: P0 backbone ablation (ddpm) first, then P1 edm.
RUNS: dict[str, ExperimentConfig] = {
    "unet-ddpm": _make("unet-ddpm", "unet", "ddpm"),
    "karras-ddpm": _make("karras-ddpm", "karras", "ddpm"),
    "uvit-ddpm": _make("uvit-ddpm", "uvit", "ddpm"),
    "unet-edm": _make("unet-edm", "unet", "edm"),
    "karras-edm": _make("karras-edm", "karras", "edm"),
    "uvit-edm": _make("uvit-edm", "uvit", "edm"),
}

# priority order for the serial queue runner.
# EDM won the CIFAR ablation (see RESULTS.md); front-load its three cells so the
# first useful AFHQ results land soonest. Within EDM, karras-edm goes first for
# speed (fastest per-step under bs=64 on this box). DDPM triplet follows.
RUN_ORDER: list[str] = [
    "karras-edm",
    "unet-edm",
    "uvit-edm",
    "karras-ddpm",
    "unet-ddpm",
    "uvit-ddpm",
]
