# UViT walkthrough (`simple_diffusion.py`)

Single file: `denoising_diffusion_pytorch/simple_diffusion.py` (1346 lines).
The `UViT` class is at `simple_diffusion.py:648` and is what
`experiments/afhq_ablation/backbones.py:103` builds for every `uvit-*` cell.

Paper: Hoogeboom et al. 2023, *Simple Diffusion: End-to-end diffusion for high
resolution images* (<https://arxiv.org/abs/2301.11093>). The paper is
"convolutional hull + transformer bottleneck", plus a resolution-aware log-SNR
schedule. This walkthrough covers only the `UViT` backbone — the ablation does
*not* use `simple_diffusion.GaussianDiffusion`; UViT is wired to the top-level
`GaussianDiffusion` (DDPM) or `ElucidatedDiffusion` (EDM) instead.

## Constructor (`simple_diffusion.py:679–817`)

Argument list with the ablation's `_BACKBONE_KWARGS` defaults
(`experiments/afhq_ablation/configs.py:86`):

```python
UViT(
    channels=3,
    dim=96, dim_mults=(1, 2, 4), vit_depth=8,
    attn_dim_head=32, attn_heads=4,
    # left at their defaults:
    init_dim=None, out_dim=None, downsample_factor=2, vit_dropout=0.2,
    ff_mult=4, learned_sinusoidal_dim=16, patch_size=1, dual_patchnorm=False,
    init_img_transform=None, final_img_itransform=None,
)
```

Shape overview:

```
init_conv (7×7, kernel padding=3)  →  t_mlp
      ↓
[ResBlk, ResBlk, LinearAttn, Downsample(pixel-unshuffle)] × 3     # convolutional encoder
      ↓
Transformer (vit_depth=8 layers of full-attn + FiLM feed-forward)  # bottleneck
      ↓
[Upsample(pixel-shuffle), ResBlk(+skip), ResBlk(+skip), LinearAttn] × 3   # convolutional decoder
      ↓
ResBlk(+init_conv skip) → 1×1 conv → out
```

The convolutional hull is nearly a plain `Unet` — same 7×7 init conv, same
FiLM-conditioned ResNet blocks, linear attention at every stage. Everything
that's *distinctively UViT* is in the bottleneck.

## Time embedding

`LearnedSinusoidalPosEmb` (`simple_diffusion.py:266`):

```python
def forward(self, x):
    x = rearrange(x, 'b -> b 1')
    freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
    fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
    fouriered = torch.cat((x, fouriered), dim=-1)               # +1 for the raw x
    return fouriered
```

`weights` is a `nn.Parameter` (not frozen), so the Fourier frequencies are
learned. The `* 2 * math.pi` multiplication means, exactly like
`MPFourierEmbedding` in KarrasUnet, this embedding wants a *continuous
log-SNR-like* scalar in roughly `[-15, 15]`. Feeding an integer `t ∈ [0, 1000)`
aliases.

That's why `BackboneAdapter` maps integer timesteps through a log-SNR lookup
table whenever `uvit` is combined with `ddpm`
(`experiments/afhq_ablation/backbones.py:114`).

The MLP head is standard:

```python
time_dim = dim * 4
self.time_mlp = nn.Sequential(
    LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
    nn.Linear(learned_sinusoidal_dim + 1, time_dim),
    nn.GELU(),
    nn.Linear(time_dim, time_dim),
)
```

## Encoder / decoder stages

Convolutional stage list (line 780) — one entry per level:

```python
self.downs.append(nn.ModuleList([
    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
    LinearAttention(dim_in),
    Downsample(dim_in, dim_out, factor=factor),
]))
```

Blocks:

- `ResnetBlock` (`simple_diffusion.py:342`): the same `Block × 2 + residual
  conv` pattern as `Unet`, with FiLM `(scale, shift) = mlp(t).chunk(2)` on the
  first block and RMSNorm (`simple_diffusion.py:226`) in place of GroupNorm.
- `LinearAttention` (`simple_diffusion.py:394`): the O(n) softmax attention
  from Katharopoulos-style kernel-trick attention. Runs at every level
  including the highest resolution — full attention at 32×32 or 64×64 would
  quadratic-blow-up VRAM.
- `Downsample` (`simple_diffusion.py:200`): pixel-unshuffle
  `Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w')` + 1×1 conv. Same
  space-to-depth trick as the top-level `Unet`.
- `Upsample` (`simple_diffusion.py:137`): pixel-shuffle with a Kaiming-init
  weight tiled `factor²` times (line 182–186) so the shuffle starts producing
  a smooth `factor×` upsample rather than a checkerboard.

Ablation `dim=96, dim_mults=(1,2,4)` → three encoder/decoder stages,
64→32→16→8, then the transformer runs at 8×8 with channel width `dim * 4 =
384`.

## The transformer bottleneck (`simple_diffusion.py:595–645`)

```python
class Transformer(nn.Module):
    def __init__(self, dim, time_cond_dim, depth, dim_head=32, heads=4,
                 ff_mult=4, dropout=0.):
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim=dim, mult=ff_mult, cond_dim=time_cond_dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x, t) + x
        return x
```

Two things distinguish this from a vanilla ViT block:

1. `Attention` is the *full* multi-head attention
   (`simple_diffusion.py:457`), not the linear one used in the conv hull. It
   only runs at 8×8, so 64 tokens; full attention is cheap here.
2. `FeedForward` accepts `t`, and injects it via FiLM at the head of the
   feed-forward branch. The `x = ff(x, t) + x` residual therefore carries
   time conditioning at every transformer layer, not only via the input
   features from the encoder.

The bottleneck runs at 8-block depth in the ablation (`vit_depth=8`), so
there are 8 attention passes and 8 FiLM-conditioned feed-forward passes at
resolution 8×8.

## `forward` (lines 819–888)

```python
x = self.init_img_transform(x)                # identity in the ablation
x = self.init_conv(x)
r = x.clone()                                  # long skip to the tail
t = self.time_mlp(time)

h = []
for block1, block2, attn, downsample in self.downs:
    x = block1(x, t);       h.append(x)
    x = block2(x, t);       x = attn(x);   h.append(x)
    x = downsample(x)

# spatial → sequence for the transformer
x = rearrange(x, 'b c h w -> b h w c')
x, ps = pack([x], 'b * c')
x = self.vit(x, t)
x, = unpack(x, ps, 'b * c')
x = rearrange(x, 'b h w c -> b c h w')

for upsample, block1, block2, attn in self.ups:
    x = upsample(x)
    x = torch.cat((x, h.pop()), dim=1);   x = block1(x, t)
    x = torch.cat((x, h.pop()), dim=1);   x = block2(x, t);   x = attn(x)

x = torch.cat((x, r), dim=1)
x = self.final_res_block(x, t)
x = self.final_conv(x)
x = self.unpatchify(x)                         # identity in the ablation
return self.final_img_itransform(x)            # identity in the ablation
```

- `init_img_transform` / `final_img_itransform` hook points let you run the
  network in a wavelet (DWT) domain à la the paper's "downsampling for free"
  trick. The ablation leaves them at `identity`.
- `patch_size=1` in the ablation means no `nn.ConvTranspose2d` unpatchify, no
  strided patch conv; the input goes through the 7×7 init conv unchanged.
- Two skip pushes per encoder stage and two pops per decoder stage. That's the
  reason each decoder stage has two `ResnetBlock(dim_in * 2, dim_in)` — one
  per skip.
- No self-conditioning code path. `UViT.forward` takes exactly `(x, time)` —
  this is the assertion `BackboneAdapter` documents in `backbones.py:32` and
  the reason self-conditioning is disabled across the ablation.

## What the file has but the ablation doesn't use

- `simple_diffusion.GaussianDiffusion` (`simple_diffusion.py:1019`) —
  continuous-time, log-SNR-only diffusion wrapper with `noise_d`-shifted
  schedule for resolution-aware training. Not imported by the ablation. The
  UViT backbone is instead paired with the DDPM `GaussianDiffusion` from
  `denoising_diffusion_pytorch.py` (`uvit-ddpm`) or with `ElucidatedDiffusion`
  (`uvit-edm`).
- Log-SNR schedule helpers at `simple_diffusion.py:949–1015`
  (`logsnr_schedule_cosine`, `logsnr_schedule_shifted`,
  `logsnr_schedule_interpolated`). Also unused here; the ablation gets its
  log-SNR from the DDPM `alphas_cumprod` lookup table.

## How the ablation calls it

Constructor: `backbones.py:103–106`, expanded from `_BACKBONE_KWARGS['uvit']`.

Wrapper: `BackboneAdapter(net, channels=3, continuous_noise=(cfg.diffusion=='edm'),
log_snr_time=(cfg.diffusion=='ddpm'))` (`backbones.py:116`).

- **`uvit-ddpm`.** `log_snr_time=True`, so `BackboneAdapter.forward` remaps
  the integer `t` through `log_snr_table[t]` before calling
  `UViT(x, log_snr)` (`backbones.py:70`). Sampler is DDIM (100 steps).
- **`uvit-edm`.** `continuous_noise=True` (which satisfies EDM's
  `random_or_learned_sinusoidal_cond` assertion via
  `backbones.py:52`), and the wrapper hands `UViT(x, 0.25·log σ)`
  directly. Sampler is Heun-with-churn (32 steps).

Parameter budget with the ablation kwargs works out to ~27M — within 10% of
the other two backbones, as tracked by
`experiments/afhq_ablation/paramtable.py`.
