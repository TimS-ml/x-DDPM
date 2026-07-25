# Ablation walkthrough index

Code-level walkthroughs for the backbones and diffusion processes used in
`experiments/afhq_ablation`. Each linked file is self-contained and grounded
in `file:line` refs against the corresponding source.

Two folders, one per layer of the [three-layer design](./architecture_overview.md):

- [`backbones/`](./backbones/README.md) — Layer-1 networks (`Unet`,
  `KarrasUnet`, `UViT`).
- [`diffusion/`](./diffusion/README.md) — Layer-2 processes (`GaussianDiffusion`,
  `ElucidatedDiffusion`).

| Kind | Doc | Source file (in `denoising_diffusion_pytorch/`) |
|------|-----|-------------------------------------------------|
| Backbone (reference U-Net, RMSNorm + FiLM) | [`backbones/unet.md`](./backbones/unet.md) | `denoising_diffusion_pytorch.py` — `Unet` (432) |
| Backbone (magnitude-preserving UNet, EDM2 Config G) | [`backbones/karras_unet.md`](./backbones/karras_unet.md) | `karras_unet.py` — `KarrasUnet` (1028) |
| Backbone (conv hull + transformer bottleneck) | [`backbones/uvit.md`](./backbones/uvit.md) | `simple_diffusion.py` — `UViT` (648) |
| Diffusion (discrete t, β-schedule, DDIM sampler) | [`diffusion/ddpm.md`](./diffusion/ddpm.md) | `denoising_diffusion_pytorch.py` — `GaussianDiffusion` (738) |
| Diffusion (continuous σ, preconditioning, Heun-with-churn) | [`diffusion/edm.md`](./diffusion/edm.md) | `elucidated_diffusion.py` — `ElucidatedDiffusion` (129) |

The high-level pieces this index sits under:
[`architecture_overview.md`](./architecture_overview.md),
[`core_components.md`](./core_components.md),
[`model_inference_flow.md`](./model_inference_flow.md).

## What the ablation actually wires

Cross-reference so the walkthroughs don't have to repeat it. All files live
under `experiments/afhq_ablation/`.

### The 3×2 matrix

| run | backbone | diffusion | LR (`_BACKBONE_LR`) | sampler |
|-----|----------|-----------|---------------------|---------|
| `unet-ddpm` | `Unet` | `GaussianDiffusion` | `2e-4` | DDIM, 100 steps |
| `karras-ddpm` | `KarrasUnet` | `GaussianDiffusion` | `3e-3` | DDIM, 100 steps |
| `uvit-ddpm` | `UViT` | `GaussianDiffusion` | `2e-4` | DDIM, 100 steps |
| `unet-edm` | `Unet` | `ElucidatedDiffusion` | `2e-4` | Heun-with-churn, 32 steps |
| `karras-edm` | `KarrasUnet` | `ElucidatedDiffusion` | `3e-3` | Heun-with-churn, 32 steps |
| `uvit-edm` | `UViT` | `ElucidatedDiffusion` | `2e-4` | Heun-with-churn, 32 steps |

### The two glue files

- `backbones.py` — `build_backbone` + `BackboneAdapter`
  ([walkthrough anchors below](#backboneadapter-cheatsheet)).
- `diffusions.py` — `build_diffusion` (26 lines, no branching worth
  describing beyond "call the right constructor and attach the log-SNR table
  if needed").

Everything else in the folder is training-loop infrastructure and doesn't
touch model math.

### `BackboneAdapter` cheatsheet

The single source of truth for reconciling three different backbone signatures
with two different diffusion contracts is `BackboneAdapter` at
`experiments/afhq_ablation/backbones.py:26`.

```
                       time signal fed to net
              ┌─────────────────────────────────────────┐
              │        DDPM (integer t)   │  EDM (σ)     │
──────────────┼───────────────────────────┼──────────────┤
Unet          │ int t → SinusoidalPosEmb  │ 0.25·log σ   │
              │  (theta=10000)            │              │
KarrasUnet    │ int t → log_snr_table[t]  │ 0.25·log σ   │
              │  → MPFourierEmbedding     │              │
UViT          │ int t → log_snr_table[t]  │ 0.25·log σ   │
              │  → LearnedSinusoidalPosEmb│              │
──────────────┼───────────────────────────┼──────────────┤
r_o_l_s_cond  │ False                     │ True         │
```

- `random_or_learned_sinusoidal_cond` is a two-way assertion:
  `GaussianDiffusion` requires falsy (line 792 of the DDPM file);
  `ElucidatedDiffusion` requires truthy (line 190 of the EDM file). The
  adapter sets it from `cfg.diffusion` (`backbones.py:52`).
- The `log_snr_table` is only built for `karras` and `uvit` under DDPM
  (`backbones.py:114`). `attach_log_snr` (line 59) computes
  `log(ᾱ_t) − log(1−ᾱ_t)` from the DDPM schedule and stores it as a buffer;
  `forward` (line 68) does `time = log_snr_table[time.long()]` before calling
  the underlying network.
- Self-conditioning is off everywhere. UViT has no code path for it; turning
  it off in Unet/KarrasUnet keeps `backbone(x, time)` as the single shared
  interface.

### Parameter budgets

`experiments/afhq_ablation/configs.py:_BACKBONE_KWARGS` is tuned so the three
backbones stay within ~10% of each other at ~26–29M parameters at 64×64. Run
`python experiments/afhq_ablation/paramtable.py` to print the table.

### Where the LR asymmetry comes from

The `karras` LR is `3e-3` while the other two are `2e-4`. Reason lives in
[`backbones/karras_unet.md`](./backbones/karras_unet.md#1-normalize_weight-algorithm-1-lines-412439-and-forced-norm-layers)
(the forced weight-norm section) and is summarised in
`experiments/afhq_ablation/README.md:34–40`. Short version: MP layers clamp
weight rows to `sqrt(fan_in)` every step, so the effective LR is `sqrt(fan_in)×`
smaller than a plain net at the same nominal LR.
