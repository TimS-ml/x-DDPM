# Unet walkthrough (`denoising_diffusion_pytorch.py:432`)

The reference 2D backbone shipped alongside `GaussianDiffusion` in
`denoising_diffusion_pytorch/denoising_diffusion_pytorch.py` (1714 lines).
Used as the `unet` cell in `experiments/afhq_ablation`.

- Constructor: `denoising_diffusion_pytorch.py:477`
- Forward: `denoising_diffusion_pytorch.py:607`
- Ablation wire-up: `experiments/afhq_ablation/backbones.py:80`

Compared with `KarrasUnet` and `UViT`, this is the "plain" one: GroupNorm-ish
RMSNorm normalisation, standard biased convs, FiLM-conditioned ResNet blocks,
linear attention at every stage except the bottleneck. No magnitude
preservation, no transformer bottleneck.

## The time embedding switch (`denoising_diffusion_pytorch.py:517–533`)

The single most consequential constructor flag is
`learned_sinusoidal_cond` (or the equivalent `random_fourier_features`).
It picks one of two time embeddings:

```python
self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

if self.random_or_learned_sinusoidal_cond:
    sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
    fourier_dim = learned_sinusoidal_dim + 1
else:
    sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
    fourier_dim = dim

self.time_mlp = nn.Sequential(
    sinu_pos_emb,
    nn.Linear(fourier_dim, time_dim),
    nn.GELU(),
    nn.Linear(time_dim, time_dim),
)
```

The two embeddings behave very differently:

- **`SinusoidalPosEmb`** (line 197). Standard exponential-frequency embedding
  with `theta=10000`. Frequencies span decades so the integer timestep `t ∈
  [0, T)` is well resolved. This is the mode `Unet` runs in under DDPM.
- **`RandomOrLearnedSinusoidalPosEmb`** (line 222). Fourier embedding whose
  weights multiply the input by `2π`:
  `freqs = x * self.weights * 2π; [x, sin(freqs), cos(freqs)]`. It wants a
  continuous scalar in roughly `[-15, 15]` like log-SNR. Feeding an integer
  `t ~ 1000` here aliases badly. This is the mode `Unet` runs in under EDM.

`self.random_or_learned_sinusoidal_cond` is also the *contract flag* the
diffusion wrappers check:

- `GaussianDiffusion.__init__` asserts it is **falsy**
  (`denoising_diffusion_pytorch.py:792`).
- `ElucidatedDiffusion.__init__` asserts it is **truthy**
  (`elucidated_diffusion.py:190`).

So the same `Unet` class serves both wrappers, but the constructor call must
match the wrapper. Ablation glue (`experiments/afhq_ablation/backbones.py:81–91`):

```python
Unet(
    channels=cfg.channels,
    self_condition=False,
    learned_sinusoidal_cond=(cfg.diffusion == 'edm'),   # True under EDM
    flash_attn=False,
    **cfg.backbone_kwargs,   # dim=108, dim_mults=(1,2,4), attn_dim_head=32, attn_heads=4
)
```

Under DDPM the flag is `False` → `SinusoidalPosEmb` + integer `t`. Under EDM
the flag is `True` → `RandomOrLearnedSinusoidalPosEmb` + `c_noise(σ) = 0.25 log σ`.

## Building blocks

### `Block` (line 252)

`Conv2d(3×3) → RMSNorm → optional FiLM (scale, shift) → SiLU → Dropout`.

```python
def forward(self, x, scale_shift=None):
    x = self.proj(x)
    x = self.norm(x)
    if exists(scale_shift):
        scale, shift = scale_shift
        x = x * (scale + 1) + shift        # FiLM: additive + multiplicative
    x = self.act(x)
    return self.dropout(x)
```

`RMSNorm` (line 180) is the channel-dim variant: normalise along `dim=1`, scale
by `sqrt(dim)` × a learned per-channel gain.

### `ResnetBlock` (line 277)

Two `Block`s plus a residual conv. FiLM enters only in the first block.

```python
def forward(self, x, time_emb=None):
    scale_shift = None
    if exists(self.mlp) and exists(time_emb):
        time_emb = self.mlp(time_emb)                    # SiLU + Linear(t_dim, dim_out*2)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale_shift = time_emb.chunk(2, dim = 1)         # (scale, shift)

    h = self.block1(x, scale_shift = scale_shift)
    h = self.block2(h)
    return h + self.res_conv(x)                          # plain + residual
```

The MLP `SiLU → Linear(t_dim, dim_out*2)` projects the shared time embedding
per-block and splits into `(scale, shift)`. Contrast with `KarrasUnet` where
FiLM is scale-only and Additive shift is dropped for MP.

### `LinearAttention` (line 318)

O(n) softmax-kernel attention with a small learnable `mem_kv` (four global
tokens per head). Used at every non-bottleneck resolution because full
attention at 32×32 or 64×64 is quadratic in spatial tokens and wasteful.

```python
q = q.softmax(dim=-2)          # softmax over channels (not tokens)
k = k.softmax(dim=-1)
context = einsum('b h d n, b h e n -> b h d e', k, v)   # K^T @ V first
out = einsum('b h d e, b h d n -> b h e n', context, q) # then Q @ (K^T V)
```

### `Attention` (line 379)

Standard multi-head attention with `mem_kv` and pinned through the shared
`Attend` module (`attend.py`). Used only at the bottleneck by default (see
`full_attn` handling below).

`Attend` (`attend.py`) dispatches to PyTorch 2 SDPA when `flash=True`. The
ablation sets `flash_attn=False` (`backbones.py:89`) because SDPA has no fp32
kernel on the 4090 laptop and FID sampling runs in fp32; the einsum fallback
works in every dtype.

## Encoder / mid / decoder

Encoder loop (`denoising_diffusion_pytorch.py:560–572`):

```python
for ind, ((dim_in, dim_out), layer_full_attn, ...) in enumerate(...):
    is_last = ind >= num_resolutions - 1
    attn_klass = FullAttention if layer_full_attn else LinearAttention
    self.downs.append(ModuleList([
        resnet_block(dim_in, dim_in),
        resnet_block(dim_in, dim_in),
        attn_klass(dim_in, ...),
        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
    ]))
```

`full_attn` defaults to `(False,)*(n-1) + (True,)` (line 538–539) — full
attention only at the bottleneck, linear everywhere else. The ablation leaves
this default.

`Downsample` (line 168) is a stable space-to-depth:
`Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w') → Conv2d(dim*4, dim_out, 1)`.
Not a strided conv.

Mid (line 574–578):

```python
self.mid_block1 = resnet_block(mid_dim, mid_dim)
self.mid_attn   = FullAttention(mid_dim, ...)
self.mid_block2 = resnet_block(mid_dim, mid_dim)
```

Decoder (line 581–593) mirrors the encoder but each `ResnetBlock` takes
concatenated `(x + skip)`, hence `resnet_block(dim_out + dim_in, dim_out)`:

```python
self.ups.append(ModuleList([
    resnet_block(dim_out + dim_in, dim_out),   # +dim_in for skip
    resnet_block(dim_out + dim_in, dim_out),   # +dim_in for skip
    attn_klass(dim_out, ...),
    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
]))
```

`Upsample` (line 157) is nearest-neighbour interpolation + 3×3 conv.

Finally:

```python
self.final_res_block = resnet_block(init_dim * 2, init_dim)   # *2 for hairpin skip
self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)
```

## Forward pass (`denoising_diffusion_pytorch.py:607–667`)

```python
def forward(self, x, time, x_self_cond = None):
    assert all(divisible_by(d, self.downsample_factor) for d in x.shape[-2:])

    if self.self_condition:
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x_self_cond, x), dim = 1)

    x = self.init_conv(x)          # 7×7 conv
    r = x.clone()                  # long "hairpin" skip to the tail
    t = self.time_mlp(time)

    h = []                          # skip stack

    for block1, block2, attn, downsample in self.downs:
        x = block1(x, t); h.append(x)
        x = block2(x, t)
        x = attn(x) + x;  h.append(x)     # attention with plain residual
        x = downsample(x)

    x = self.mid_block1(x, t)
    x = self.mid_attn(x) + x
    x = self.mid_block2(x, t)

    for block1, block2, attn, upsample in self.ups:
        x = torch.cat((x, h.pop()), dim = 1); x = block1(x, t)
        x = torch.cat((x, h.pop()), dim = 1); x = block2(x, t)
        x = attn(x) + x
        x = upsample(x)

    x = torch.cat((x, r), dim = 1)     # hairpin skip in
    x = self.final_res_block(x, t)
    return self.final_conv(x)
```

Two skip pushes per encoder stage → two pops per decoder stage. The `r =
init_conv.clone()` at the top and the concat at the tail is a *long* skip
that lets the head reuse the raw input features one more time.

`init_conv` uses a 7×7 kernel (line 507) — larger than the 3×3 used inside
blocks — to grab wider local context up front.

## Self-conditioning (`self_condition` flag)

`Unet` supports the Bit Diffusion self-conditioning trick: at training time
50% of the batch runs the network *twice*, the first pass's `x̂0` gets
concatenated onto `x` on the second pass. Toggle at construction
(`self_condition=True`); the `input_channels` doubles (line 503) and the
diffusion wrapper takes care of feeding the previous prediction back in.

The ablation disables self-conditioning across all runs
(`experiments/afhq_ablation/backbones.py:11–14`) because `UViT` has no
matching code path, and turning it off in `Unet`/`KarrasUnet` keeps
`backbone(x, time)` as the single shared calling convention.

## The parameter budget (ablation)

With `dim=108, dim_mults=(1,2,4), attn_dim_head=32, attn_heads=4` this
comes out to ~27M parameters at 64×64 — within 10% of `KarrasUnet` and
`UViT` under the ablation's `_BACKBONE_KWARGS`
(`experiments/afhq_ablation/configs.py:82–87`). Run
`python experiments/afhq_ablation/paramtable.py` to reproduce.

## How the ablation wires it

Under either diffusion class the adapter around `Unet` is
`BackboneAdapter(net, channels, continuous_noise, log_snr_time=False)`
(`experiments/afhq_ablation/backbones.py:114` — `log_snr_time` stays `False`
because `Unet` under DDPM already accepts an integer timestep via
`SinusoidalPosEmb`).

- **`unet-ddpm`.** `learned_sinusoidal_cond=False`; `SinusoidalPosEmb` eats
  integer `t ∈ [0, 1000)`; sampler is DDIM at 100 steps
  (see [`../diffusion/ddpm.md`](../diffusion/ddpm.md)).
- **`unet-edm`.** `learned_sinusoidal_cond=True`;
  `RandomOrLearnedSinusoidalPosEmb` eats `c_noise(σ) = 0.25 log σ`; sampler
  is Heun-with-churn at 32 steps
  (see [`../diffusion/edm.md`](../diffusion/edm.md)).

Same `Unet` class, same weights shape, one constructor bit flipped.
