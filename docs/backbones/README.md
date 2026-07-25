# Backbones

Layer-1 in the [three-layer design](../architecture_overview.md): the network
that consumes a noisy image plus a time/noise signal and predicts the quantity
the diffusion process wants (ε, x0, v, or a denoised image D). The backbone
knows no diffusion math.

## Covered here

| Backbone | Doc | Source | Nickname |
|----------|-----|--------|----------|
| `Unet` | [`unet.md`](./unet.md) | `denoising_diffusion_pytorch.py:432` | reference 2D U-Net, GroupNorm-ish RMSNorm, FiLM |
| `KarrasUnet` | [`karras_unet.md`](./karras_unet.md) | `karras_unet.py:1028` | EDM2 "Config G", magnitude-preserving, bias-free |
| `UViT` | [`uvit.md`](./uvit.md) | `simple_diffusion.py:648` | conv encoder/decoder + transformer bottleneck |

These are the three backbones the `experiments/afhq_ablation` harness compares.

## In the codebase but not covered here

- `Unet1D` — 1D twin of `Unet` for waveforms / time series.
  `denoising_diffusion_pytorch_1d.py`.
- `KarrasUnet1D` — 1D magnitude-preserving. `karras_unet_1d.py`.
- `KarrasUnet3D` — 3D magnitude-preserving for video / volumetric data.
  `karras_unet_3d.py`.
- The class-conditional `Unet` inside `classifier_free_guidance.py` —
  same body as `Unet`, plus a null-class embedding and CFG hooks.

Add walkthroughs here if you ever run an ablation on any of them.

## What every backbone must expose

The diffusion wrappers in `../diffusion/` all call the backbone as
`net(x, time, self_cond=None)` and probe a handful of attributes:

| Attribute | Consumer | Meaning |
|-----------|----------|---------|
| `channels` | `GaussianDiffusion`, `ElucidatedDiffusion` | input channel count |
| `out_dim` | `GaussianDiffusion` (asserts `==channels` unless learned variance) | output channel count |
| `self_condition` | both wrappers | whether the backbone accepts a `self_cond` image |
| `random_or_learned_sinusoidal_cond` | both wrappers (see below) | whether time is a continuous scalar |

The `random_or_learned_sinusoidal_cond` flag is a two-way constraint:

- `GaussianDiffusion` **rejects** truthy (asserts falsy at
  `denoising_diffusion_pytorch.py:792`) because it feeds integer timesteps
  and Fourier `× 2π` embeddings would alias.
- `ElucidatedDiffusion` **requires** truthy (asserts at
  `elucidated_diffusion.py:190`) because it feeds `c_noise(σ) = 0.25 log σ`
  which needs a continuous-scalar-friendly time embedding.

`Unet` toggles between the two modes via the `learned_sinusoidal_cond` /
`random_fourier_features` constructor args; `KarrasUnet` and `UViT` are
continuous-only. The ablation harness reconciles all of this in one place —
`BackboneAdapter` at `experiments/afhq_ablation/backbones.py:26`. Cheatsheet
in [`../ablation_walkthrough.md`](../ablation_walkthrough.md#backboneadapter-cheatsheet).

## Time-embedding contract at a glance

| Backbone | Time embed under DDPM | Time embed under EDM |
|----------|------------------------|-----------------------|
| `Unet` | `SinusoidalPosEmb(theta=10000)` on integer `t` | `RandomOrLearnedSinusoidalPosEmb` on `0.25·log σ` |
| `KarrasUnet` | `MPFourierEmbedding` on `log_snr_table[t]` (via adapter) | `MPFourierEmbedding` on `0.25·log σ` |
| `UViT` | `LearnedSinusoidalPosEmb` on `log_snr_table[t]` (via adapter) | `LearnedSinusoidalPosEmb` on `0.25·log σ` |
