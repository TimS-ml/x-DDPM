# Core Components

This document describes the key classes and functions that make up the
reference 2D stack in `denoising_diffusion_pytorch.py` and the shared pieces
reused across the package. See
[`architecture_overview.md`](./architecture_overview.md) for how they fit
together and [`model_inference_flow.md`](./model_inference_flow.md) for the
sampling pipelines.

## 1. `Unet` — the backbone

A time-conditioned U-Net that predicts noise / `x0` / `v`.

- **Input stem.** `init_conv` (7×7) projects the image into feature space; the
  initial activation is cloned and re-used as a final skip connection.
- **Time embedding.** `SinusoidalPosEmb` (or learned/random Fourier features) →
  MLP → a `time_dim`-wide vector fed to every residual block.
- **Residual blocks.** `ResnetBlock` = two `Block`s (`Conv2d → GroupNorm → SiLU`)
  with a residual conv. Time conditioning enters via FiLM-style scale/shift
  (`x * (scale + 1) + shift`).
- **Attention.** `Attention` (full) and `LinearAttention` (O(n)) wrap `Attend`
  (see §6). Each block uses pre-norm + residual; learnable memory key/values are
  prepended to the attention context.
- **Down / Mid / Up paths.** Standard U-Net encoder → bottleneck → decoder with
  channel multipliers `dim_mults`; encoder activations are concatenated into the
  decoder via skip connections.
- **Self-conditioning** (optional). The previous step's predicted `x0` is
  concatenated to the input, following the Bit Diffusion trick.

## 2. `GaussianDiffusion` — the diffusion process

Owns the math. Its responsibilities:

### Schedule and precomputed buffers
At construction it builds `betas` from `linear_beta_schedule` /
`cosine_beta_schedule` / `sigmoid_beta_schedule`, then derives and
`register_buffer`s the constants used everywhere:

- `alphas_cumprod` (ᾱ_t), `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`
- `sqrt_recip_alphas_cumprod`, `sqrt_recipm1_alphas_cumprod`
- posterior terms `posterior_variance`, `posterior_log_variance_clipped`,
  `posterior_mean_coef1/2`

Storing these as buffers means each reverse step is a few `extract()` lookups
rather than recomputation.

### Forward process `q`
`q_sample(x_start, t, noise)` implements the closed form
`x_t = √ᾱ_t · x0 + √(1−ᾱ_t) · ε`. `q_posterior` gives the tractable Gaussian
`q(x_{t-1} | x_t, x0)`.

### The parameterisation adapter — `model_predictions`
This is the hub that decouples the backbone's `objective` from the rest of the
code. It runs the network once and returns a `ModelPrediction(pred_noise,
pred_x_start)` **regardless** of objective, using:

- `predict_start_from_noise` / `predict_noise_from_start`
- `predict_v` / `predict_start_from_v`

So `pred_noise`, `pred_x0`, and `pred_v` objectives all funnel into a uniform
`(ε, x0)` pair that every sampler and the loss consume.

### Loss `p_losses`
Sample `t`, noise `x0` with `q_sample`, run the model, build the target for the
chosen objective, and take MSE (optionally min-SNR weighted, optionally with
offset / immiscible noise).

### Samplers
`p_sample_loop` (ancestral DDPM), `ddim_sample` (DDIM), and the `sample`
dispatcher that picks between them based on `is_ddim_sampling`. Detailed in
[`model_inference_flow.md`](./model_inference_flow.md).

## 3. `Trainer` — the training loop

A batteries-included loop built on HuggingFace `accelerate`:

- Infinite-cycled `DataLoader`, `Adam`, gradient accumulation.
- **EMA** (`ema_pytorch.EMA`) maintains a shadow copy of the weights; samples
  and checkpoints use the EMA model for stability.
- Mixed precision, multi-GPU, periodic `save()` / `sample()`.
- Optional **FID** via `FIDEvaluation`.

## 4. Objective parameterisations

| `objective` | Network outputs | `x0` recovered by | Notes |
|-------------|-----------------|-------------------|-------|
| `pred_noise` | noise `ε` | `predict_start_from_noise` | Original DDPM (Ho et al.). |
| `pred_x0` | clean image `x0` | identity | Direct; can be unstable at high noise. |
| `pred_v` | velocity `v = √ᾱ·ε − √(1−ᾱ)·x0` | `predict_start_from_v` | Progressive-distillation param; well-scaled at all noise levels. Default here. |

## 5. Noise schedules

- `linear_beta_schedule` — original DDPM linear β.
- `cosine_beta_schedule` — Improved-DDPM cosine ᾱ (smoother, better for small
  images).
- `sigmoid_beta_schedule` — sigmoid-shaped, useful at higher resolution.

The continuous-time modules replace these with **log-SNR** schedules
(`beta_linear_log_snr`, `alpha_cosine_log_snr`, or a learned monotonic MLP),
where `α(t) = √sigmoid(logSNR)` and `σ(t) = √sigmoid(−logSNR)`.

## 6. `Attend` — attention backend

`attend.py` provides a single `Attend` module that:

- Uses PyTorch 2.0 `scaled_dot_product_attention` when `flash = True`.
- Detects GPU capability (A100 vs not) and selects flash / mem-efficient / math
  kernels accordingly.
- Falls back to an explicit `einsum` softmax attention otherwise.

All backbones route their attention through it, so flash attention is a single
flag.

## 7. Shared helpers

Small utilities reused across modules: `exists`, `default`, `cast_tuple`,
`extract` (gather schedule constants at timesteps `t` and broadcast to image
shape), `normalize_to_neg_one_to_one` / `unnormalize_to_zero_to_one`,
`num_to_groups`, and `cycle` (infinite dataloader).

## 8. Variant classes at a glance

| Class | Module | Backbone it expects | Headline idea |
|-------|--------|---------------------|---------------|
| `GaussianDiffusion` | `denoising_diffusion_pytorch.py` | `Unet` | Reference DDPM + DDIM. |
| `ElucidatedDiffusion` | `elucidated_diffusion.py` | any net with continuous-σ cond | EDM preconditioning + Heun / DPM++ sampler. |
| `ContinuousTimeGaussianDiffusion` | `continuous_time_gaussian_diffusion.py` | log-SNR-conditioned net | Continuous time, learnable schedule. |
| `VParamContinuousTimeGaussianDiffusion` | `v_param_..._diffusion.py` | log-SNR-conditioned net | Continuous time + v-prediction. |
| `LearnedGaussianDiffusion` | `learned_gaussian_diffusion.py` | `Unet(out_dim = 2·channels)` | Learned variance + hybrid VLB loss. |
| `WeightedObjectiveGaussianDiffusion` | `weighted_objective_..._diffusion.py` | `Unet(out_dim = 2·channels + 2)` | Per-pixel ε/x0 blend. |
| `KarrasUnet` | `karras_unet.py` | (is a backbone) | Magnitude-preserving network. |
| `UViT` | `simple_diffusion.py` | (is a backbone) | Conv hull + Transformer bottleneck. |
| `GaussianDiffusion` (repaint) | `repaint.py` | `Unet` | DDPM + inpainting mask blend. |
