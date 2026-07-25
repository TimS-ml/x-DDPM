# Diffusion processes

Layer-2 in the [three-layer design](../architecture_overview.md): the class
that owns the forward noising `q`, the reverse denoising `p`, the training
objective, and the sampler. It calls the backbone once per step.

## Covered here

| Class | Doc | Source | Nickname |
|-------|-----|--------|----------|
| `GaussianDiffusion` | [`ddpm.md`](./ddpm.md) | `denoising_diffusion_pytorch.py:738` | discrete t, β-schedule, DDPM + DDIM sampler |
| `ElucidatedDiffusion` | [`edm.md`](./edm.md) | `elucidated_diffusion.py:129` | continuous σ, preconditioning, Heun-with-churn |

These are the two the `experiments/afhq_ablation` harness compares. See
`experiments/afhq_ablation/diffusions.py` for the two-branch factory.

## In the codebase but not covered here

Every one of these is a Layer-2 process; you can drop it in place of the two
above if the backbone contract matches.

- `ContinuousTimeGaussianDiffusion` — continuous `t ∈ [0, 1]`,
  log-SNR-parameterised, linear/cosine/learned schedules.
  `continuous_time_gaussian_diffusion.py`.
- `VParamContinuousTimeGaussianDiffusion` — same, with v-parameterisation.
  `v_param_continuous_time_gaussian_diffusion.py`.
- `LearnedGaussianDiffusion` — Improved-DDPM learned variance
  (hybrid simple + VLB loss); needs `Unet(out_dim=2*channels)`.
  `learned_gaussian_diffusion.py`.
- `WeightedObjectiveGaussianDiffusion` — per-pixel learned blend between the
  ε- and x0-parameterisations; needs `Unet(out_dim=2*channels + 2)`.
  `weighted_objective_gaussian_diffusion.py`.
- `simple_diffusion.GaussianDiffusion` — log-SNR-only, resolution-aware
  schedule shift. Meant for the UViT backbone but the ablation uses UViT
  paired with the two classes above instead.
- Classifier-free guidance variant — class-conditional `GaussianDiffusion` in
  `classifier_free_guidance.py`.
- Classifier guidance (`guided_diffusion.py`) — uses a separate noise-aware
  classifier gradient during sampling.
- `RePaint` inpainting (`repaint.py`) — mask-blend reverse step + resampling.

Add walkthroughs here as they become relevant to a comparison.

## What every diffusion process needs from the backbone

Both classes call `backbone(x, time, self_cond=None)`. What differs is the
`time` signal and the `random_or_learned_sinusoidal_cond` constraint on the
backbone. See [`../backbones/README.md`](../backbones/README.md#time-embedding-contract-at-a-glance)
for the reconciliation table.

## Sampler at a glance

| Class | Default sampler | Fast-mode sampler | Steps |
|-------|-----------------|-------------------|-------|
| `GaussianDiffusion` | `p_sample_loop` (ancestral DDPM) | `ddim_sample` (auto when `sampling_timesteps < timesteps`) | 1000 / 100 |
| `ElucidatedDiffusion` | `sample` (Heun-with-churn) | `sample_using_dpmpp` (DPM-Solver++) | 32 / ~20 |

More detail across the whole family: [`../model_inference_flow.md`](../model_inference_flow.md).
