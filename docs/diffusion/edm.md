# EDM walkthrough (`ElucidatedDiffusion`)

Single file: `denoising_diffusion_pytorch/elucidated_diffusion.py` (670 lines).
The class `ElucidatedDiffusion` is at `elucidated_diffusion.py:129` and is
what `experiments/afhq_ablation/diffusions.py:30` instantiates for every
`*-edm` cell.

Paper: Karras et al. 2022, *Elucidating the Design Space of Diffusion-Based
Generative Models* (<https://arxiv.org/abs/2206.00364>). All equation numbers
below reference that paper.

Unlike `GaussianDiffusion`, EDM:

- Works in continuous σ, not integer timesteps.
- Trains the network to predict a *denoised image* — wrapped in a
  preconditioner that mixes network output with a skip of the noisy input.
- Uses a ρ-warped σ schedule, log-normal training noise sampling, and a
  second-order Heun sampler with optional stochastic "churn".

## Constructor (`elucidated_diffusion.py:144–216`)

The full arg list, with the defaults the ablation uses:

```python
ElucidatedDiffusion(
    net,
    image_size,
    channels=3,
    num_sample_steps=32,   # ablation: 32 (configs.py:33)
    sigma_min=0.002, sigma_max=80, sigma_data=0.5, rho=7,
    P_mean=-1.2, P_std=1.2,
    S_churn=80, S_tmin=0.05, S_tmax=50, S_noise=1.003,
)
```

One assertion (`elucidated_diffusion.py:190`):

```python
assert net.random_or_learned_sinusoidal_cond
```

The backbone *must* advertise a continuous-noise time interface. That is why
`BackboneAdapter` toggles `random_or_learned_sinusoidal_cond=True` whenever
`cfg.diffusion == 'edm'` (`backbones.py:52` and `backbones.py:77`). For `Unet`
the ablation flips `learned_sinusoidal_cond=True` in the constructor
(`backbones.py:85`); `KarrasUnet` and `UViT` are continuous by construction.

Nothing is precomputed: no cumulative products, no big buffers. Everything is
derived on the fly from σ.

## Preconditioning (Table 1, Eq. 7)

The heart of EDM. The network `F_θ` is not the denoiser. The denoiser `D_θ` is
built from `F_θ` and four σ-dependent scalars:

```python
def c_skip(σ):  return σ_data² / (σ² + σ_data²)             # line 246
def c_out(σ):   return σ · σ_data · (σ² + σ_data²)^(-1/2)   # line 263
def c_in(σ):    return 1 / sqrt(σ² + σ_data²)               # line 280
def c_noise(σ): return 0.25 · log(σ)                        # line 297
```

Reading the intuition off the formulas:

- **`c_in`** rescales the noisy image so the input to the network always has
  unit variance regardless of σ.
- **`c_out`** rescales the network's raw output so its magnitude matches the
  correction the denoiser needs at this σ.
- **`c_skip`** blends the noisy input straight through the denoiser. At small σ
  it dominates (denoising = near-identity); at large σ it vanishes (denoising
  = full network prediction).
- **`c_noise`** is the time input to the backbone. It's log-σ, scaled by
  `0.25`, so the network sees a compact continuous scalar.

Assembled at `elucidated_diffusion.py:302–350`:

```python
def preconditioned_network_forward(self, noised_images, sigma, self_cond=None, clamp=False):
    padded_sigma = rearrange(sigma, 'b -> b 1 1 1')
    net_out = self.net(
        self.c_in(padded_sigma) * noised_images,
        self.c_noise(sigma),
        self_cond,
    )
    out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * net_out
    return out.clamp(-1., 1.) if clamp else out
```

Everything downstream — training loss, both samplers — goes through
`preconditioned_network_forward`. The backbone's job is only to produce
`net_out` at the network-native scale; scaling and skip are the wrapper's job.

## Training

### Noise sampling (`elucidated_diffusion.py:585–606`)

Instead of `t ~ U[0, T)`, EDM samples σ from a log-normal:

```python
def noise_distribution(self, batch_size):
    return (self.P_mean + self.P_std * torch.randn(batch_size, device=self.device)).exp()
```

With `P_mean=-1.2, P_std=1.2` the mode is around `exp(-1.2) ≈ 0.3`; support is
roughly `[0.001, 100]`. This concentrates training compute on the noise levels
where the loss weighting says the network learns the most.

### Loss (Eq. 8, Eq. 16 — `elucidated_diffusion.py:608–669`)

```python
def forward(self, images):
    images = normalize_to_neg_one_to_one(images)

    sigmas = self.noise_distribution(batch_size)             # log-normal σ
    padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')
    noise = torch.randn_like(images)
    noised_images = images + padded_sigmas * noise            # x_σ = x + σ · ε  (α = 1)

    denoised = self.preconditioned_network_forward(noised_images, sigmas)

    losses = F.mse_loss(denoised, images, reduction='none')   # MSE against clean x
    losses = reduce(losses, 'b ... -> b', 'mean')
    losses = losses * self.loss_weight(sigmas)
    return losses.mean()

def loss_weight(self, sigma):   # Eq. 16
    return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2
```

Two things:

- The forward SDE is variance-*exploding* (α = 1). There is no `√ᾱ` factor;
  noise is added at scale σ, not `√(1−ᾱ)`.
- The regression target is the clean image, not ε or v. The preconditioner
  makes this equivalent to any of them up to a linear reparameterization.

## Sampling: Heun with churn (Algorithm 2 — `elucidated_diffusion.py:391–490`)

This is the sampler `sample_ckpt.py` and periodic FID both use.

### Step 1: σ schedule (Eq. 5 — `sample_schedule`, line 357)

```python
inv_rho = 1 / self.rho
steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
sigmas = (self.sigma_max ** inv_rho +
          steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
sigmas = F.pad(sigmas, (0, 1), value=0.)      # last step is σ = 0 exactly
```

`rho=7` concentrates steps at *low* σ where fine structure is generated. The
schedule ends at exactly 0 so the last step lands on a clean image.

### Step 2: churn schedule (`elucidated_diffusion.py:439–443`)

```python
gammas = torch.where(
    (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
    min(self.S_churn / num_sample_steps, sqrt(2) - 1),
    0.,
)
```

`γ` is per-step "churn": add fresh noise to bump σ upward before denoising.
With `S_churn=80` and `num_sample_steps=32`, γ ≈ `min(80/32, √2−1) ≈ 0.414`
inside the active σ range. Setting `S_churn=0` recovers a pure deterministic
Heun ODE.

### Step 3: the loop (`elucidated_diffusion.py:459–487`)

For each `(σ, σ_next, γ)`:

```python
# 1. churn: noise the current image from σ up to σ_hat
sigma_hat = sigma + gamma * sigma
eps = self.S_noise * torch.randn(shape, device=self.device)
images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

# 2. predictor (Euler)
D  = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp=clamp)
d  = (images_hat - D) / sigma_hat
images_next = images_hat + (sigma_next - sigma_hat) * d

# 3. corrector (Heun 2nd order), skipped when σ_next == 0
if sigma_next != 0:
    D_next  = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp=clamp)
    d_next  = (images_next - D_next) / sigma_next
    images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (d + d_next)

images = images_next
```

- Two `preconditioned_network_forward` calls per step (except the last), so
  32 sampling steps means 63 backbone evaluations.
- The `d = (x - D)/σ` is exactly the score-scaled ODE right-hand side:
  `dx/dσ = (x - D_θ(x, σ)) / σ`.
- `S_noise=1.003` is a small (paper-verified) overshoot on the churn noise
  that empirically improves FID.

## DPM-Solver++ (bonus, unused by the ablation)

Also included in the file, `sample_using_dpmpp` at
`elucidated_diffusion.py:492–557`. Deterministic, 2M-style multistep in
log-SNR space. It's not called from `train.py` / `sample_ckpt.py` in this
harness; documented in `../model_inference_flow.md` if you want it.

## The EDM cell of the ablation, end to end

For `karras-edm` (the fastest per-step cell on the 4090 laptop, and the reason
it heads the queue in `configs.RUN_ORDER`):

1. `configs._make('karras-edm', 'karras', 'edm')` sets `num_sample_steps=32`,
   `lr=3e-3` (see the note in [`../backbones/karras_unet.md`](../backbones/karras_unet.md)).
2. `build_backbone(cfg)` builds `KarrasUnet(dim=72, ...)` and wraps it with
   `BackboneAdapter(continuous_noise=True, log_snr_time=False)`. The
   `random_or_learned_sinusoidal_cond=True` flag on the adapter satisfies the
   EDM assertion.
3. `build_diffusion(cfg, model)` returns
   `ElucidatedDiffusion(model, image_size=64, channels=3, num_sample_steps=32)`;
   `log_snr_time=False` so no lookup table is attached.
4. Training: `diffusion(images)` triggers the log-normal σ sampling and Eq. 16
   loss above.
5. Sampling: `diffusion.sample(batch_size=n)` runs the Heun-with-churn loop.

The `unet-edm` and `uvit-edm` cells are identical except for which backbone is
inside — no diffusion-side changes.

## Ablation caveat: time embedding contract

`Unet` under EDM must be constructed with `learned_sinusoidal_cond=True`
(`backbones.py:85`), which swaps `SinusoidalPosEmb` for
`RandomOrLearnedSinusoidalPosEmb` (`denoising_diffusion_pytorch.py:222`) —
that variant multiplies its input by `2π`, so it wants a continuous scalar
like `0.25 · log(σ)`, not an integer `t ∈ [0, 1000)`. `KarrasUnet` and `UViT`
use their own Fourier embeddings and behave the same way. Feeding an integer
timestep to any of the three under EDM would alias badly; feeding
`c_noise(σ) = 0.25 · log(σ)` to all three is what makes the shared
`backbone(x, time)` interface work.
