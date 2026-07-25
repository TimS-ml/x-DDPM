# DDPM walkthrough (`GaussianDiffusion`)

`denoising_diffusion_pytorch/denoising_diffusion_pytorch.py:738` (class body
runs to line 1330). Instantiated for every `*-ddpm` cell in the ablation via
`experiments/afhq_ablation/diffusions.py:16`.

The backbone this wrapper expects is documented separately at
[`../backbones/unet.md`](../backbones/unet.md) — `Unet` is what the ablation
uses under DDPM, but any network satisfying the contract below works.

Paper: Ho et al. 2020 (<https://arxiv.org/abs/2006.11239>). DDIM sampler from
Song et al. 2020 (<https://arxiv.org/abs/2010.02502>). `pred_v` from Salimans &
Ho 2022 (progressive distillation).

## The backbone contract (constructor asserts, lines 791–792)

```python
assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
assert not hasattr(model, 'random_or_learned_sinusoidal_cond') \
       or not model.random_or_learned_sinusoidal_cond
```

Two things:

- The backbone's `out_dim` must equal `channels`. Learned-variance variants
  double `out_dim`; they use a subclass.
- The backbone must *not* advertise a continuous-noise time interface —
  `GaussianDiffusion` feeds integer timesteps and Fourier embeddings
  (which multiply by `2π`) would alias badly at `t ~ 1000`. This is why
  `BackboneAdapter` in the ablation sets
  `random_or_learned_sinusoidal_cond=False` under DDPM
  (`experiments/afhq_ablation/backbones.py:52`).

## 1. Schedule and precomputed buffers (`__init__`, lines 773–903)

Three beta schedules are provided (`denoising_diffusion_pytorch.py:690–736`):
`linear_beta_schedule`, `cosine_beta_schedule`, `sigmoid_beta_schedule`. The
ablation picks `sigmoid` (`configs.py:30`). Every constant used at train and
sample time is precomputed once and stored as a buffer so each step is a
handful of `extract()` gathers:

```python
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)                # ᾱ_t
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

# q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
register_buffer('posterior_variance', posterior_variance)
register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
```

The full list of buffers is at `denoising_diffusion_pytorch.py:843–869`.

`sampling_timesteps` (`__init__` arg at `denoising_diffusion_pytorch.py:779`)
switches the sampler: `sampling_timesteps < timesteps` → DDIM. The ablation
sets `timesteps=1000, sampling_timesteps=100` (`configs.py:27–28`), so all
`*-ddpm` cells sample with DDIM.

## 2. Forward process `q` (`denoising_diffusion_pytorch.py:1214–1241`)

The forward noising is the DDPM closed form:

```python
@autocast('cuda', enabled=False)
def q_sample(self, x_start, t, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))
    return (
        extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )
```

`extract` (`denoising_diffusion_pytorch.py:671`) just gathers `a[t]` and
reshapes for broadcasting. `@autocast(enabled=False)` pins this to fp32 so the
noised image is well conditioned even when the rest of training runs in bf16.

## 3. The parameterisation adapter (`denoising_diffusion_pytorch.py:971–1010`)

The single most important function in the file. It calls the backbone once and
returns a `ModelPrediction(pred_noise, pred_x_start)` regardless of the
`objective` string:

```python
model_output = self.model(x, t, x_self_cond)

if self.objective == 'pred_noise':
    pred_noise = model_output
    x_start = self.predict_start_from_noise(x, t, pred_noise)
elif self.objective == 'pred_x0':
    x_start = model_output
    pred_noise = self.predict_noise_from_start(x, t, x_start)
elif self.objective == 'pred_v':
    v = model_output
    x_start = self.predict_start_from_v(x, t, v)
    pred_noise = self.predict_noise_from_start(x, t, x_start)

return ModelPrediction(pred_noise, x_start)
```

The four converters at `denoising_diffusion_pytorch.py:909–948`:

```python
predict_start_from_noise(x_t, t, noise) = sqrt(1/ᾱ_t) · x_t − sqrt(1/ᾱ_t − 1) · noise
predict_noise_from_start(x_t, t, x0)    = (sqrt(1/ᾱ_t) · x_t − x0) / sqrt(1/ᾱ_t − 1)
predict_v(x0, t, noise)                 = √ᾱ_t · noise − √(1−ᾱ_t) · x0
predict_start_from_v(x_t, t, v)         = √ᾱ_t · x_t − √(1−ᾱ_t) · v
```

Loss and every sampler consume `(ε̂, x̂0)`; the backbone can output whichever
one is convenient. The ablation uses `pred_v` (`configs.py:29`), which is
well-scaled at every noise level.

## 4. Loss (`denoising_diffusion_pytorch.py:1243–1311`)

Sample `t ~ U[0, T)`, noise `x0`, run the model, build the target for the
chosen objective, MSE, weight per SNR:

```python
x = self.q_sample(x_start=x_start, t=t, noise=noise)
model_out = self.model(x, t, x_self_cond)

if   self.objective == 'pred_noise': target = noise
elif self.objective == 'pred_x0':    target = x_start
elif self.objective == 'pred_v':     target = self.predict_v(x_start, t, noise)

loss = F.mse_loss(model_out, target, reduction='none')
loss = reduce(loss, 'b ... -> b', 'mean') * extract(self.loss_weight, t, loss.shape)
return loss.mean()
```

`loss_weight` was baked into a buffer at
`denoising_diffusion_pytorch.py:893–898`. Per objective:

| objective | weight buffer |
|-----------|--------------|
| `pred_noise` | `maybe_clipped_snr / snr` (=1 when min-SNR off) |
| `pred_x0`   | `maybe_clipped_snr` |
| `pred_v`    | `maybe_clipped_snr / (snr + 1)` |

Min-SNR (Hang et al. 2023) clips `snr` at `min_snr_gamma`. The ablation leaves
it off.

The `forward` entry point (`denoising_diffusion_pytorch.py:1313–1330`)
normalizes `[0,1] → [-1,1]`, samples `t = torch.randint(0, T, (b,))`, calls
`p_losses`. This is what `train.py` invokes as `diffusion(images)` and
backprops.

## 5. Sampling: DDPM (`p_sample_loop`, lines 1061–1091)

For `t = T−1 … 0`:

```python
model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, t)
noise = torch.randn_like(x) if t > 0 else 0.
pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
```

`p_mean_variance` (line 1012) predicts `x0`, clips it to `[-1, 1]`, then plugs
into `q_posterior` (line 950) which is just the two precomputed coefficient
buffers times `x0` and `x_t`.

## 6. Sampling: DDIM (`ddim_sample`, lines 1093–1151) — what the ablation runs

```python
times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
time_pairs = list(zip(times[:-1], times[1:]))  # (t, t_next)

for time, time_next in time_pairs:
    pred_noise, x_start, *_ = self.model_predictions(
        img, time_cond, self_cond,
        clip_x_start=True, rederive_pred_noise=True,
    )
    alpha      = self.alphas_cumprod[time]
    alpha_next = self.alphas_cumprod[time_next]
    sigma = eta * ((1 - alpha/alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    c = (1 - alpha_next - sigma ** 2).sqrt()
    img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * torch.randn_like(img)
```

`ddim_sampling_eta` defaults to `0.` (`denoising_diffusion_pytorch.py:783`), so
sampling is deterministic. `rederive_pred_noise=True` recomputes `ε̂` from the
*clipped* `x̂0` at every step; without it the two would drift apart.

## The DDPM cell of the ablation, end to end

For `unet-ddpm`:

1. `configs._make('unet-ddpm', 'unet', 'ddpm')` sets `objective='pred_v'`,
   `beta_schedule='sigmoid'`, `timesteps=1000`, `sampling_timesteps=100`,
   `lr=2e-4`.
2. `build_backbone(cfg)` builds a `Unet(dim=108, ...)` and wraps it in
   `BackboneAdapter(continuous_noise=False, log_snr_time=False)`. See
   [`../backbones/unet.md`](../backbones/unet.md) for the backbone side.
3. `build_diffusion(cfg, model)` builds `GaussianDiffusion(model, ...)`.
4. `train.py` calls `diffusion(images)` per step (loss branch above), and
   `diffusion.sample(batch_size=n)` per FID eval, which dispatches to
   `ddim_sample` because `sampling_timesteps=100 < 1000`.

Swapping the backbone (`karras-ddpm`, `uvit-ddpm`) only changes step 2 — steps
1, 3, and 4 are identical. The one extra glue for `karras`/`uvit` under DDPM
is that `BackboneAdapter` attaches a log-SNR lookup table so their Fourier
time embeddings receive a continuous scalar instead of an integer `t`; see
[`../ablation_walkthrough.md`](../ablation_walkthrough.md#backboneadapter-cheatsheet).
