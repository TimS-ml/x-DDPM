# KarrasUnet walkthrough (`karras_unet.py`)

Single file: `denoising_diffusion_pytorch/karras_unet.py` (1556 lines). The
`KarrasUnet` class is at `karras_unet.py:1028` and is what
`experiments/afhq_ablation/backbones.py:92` builds for every `karras-*` cell.

Paper: Karras et al. 2023, *Analyzing and Improving the Training Dynamics of
Diffusion Models* (<https://arxiv.org/abs/2312.02696>) — the EDM2 paper. This
implementation follows "Config G" from that paper's Figure 21.

The one-line summary: every conv/linear force-normalizes its weight rows to L2
norm `sqrt(fan_in)` on every forward pass, biases are removed everywhere, and
`GroupNorm` is replaced by pixel-wise L2 normalization. The result is a
"magnitude-preserving" (MP) network in which activation variance is
approximately constant across depth without any learned normalization stats.

## The MP primitives

Everything in the network is built out of six pieces in
`karras_unet.py:180–625`:

### 1. `normalize_weight` (Algorithm 1, lines 412–439) and forced-norm layers

```python
def normalize_weight(weight, eps=1e-4):
    weight, ps = pack_one(weight, 'o *')                 # flatten to (o, fan_in)
    normed_weight = l2norm(weight, eps=eps)              # per-row L2 to unit norm
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])   # × sqrt(fan_in)
    return unpack_one(normed_weight, ps, 'o *')
```

`Conv2d` and `Linear` (lines 441–572) both use this in `forward`:

```python
def forward(self, x):
    if self.training:
        with torch.no_grad():
            self.weight.copy_(normalize_weight(self.weight, eps=self.eps))
    weight = normalize_weight(self.weight, eps=self.eps) / sqrt(self.fan_in)
    return F.conv2d(x, weight, padding='same')
```

Two important consequences:

- `weight.copy_(...)` in the `if self.training` branch does an *in-place*
  renormalization of the parameter itself, so the optimizer's next Adam
  moment update sees an already-unit-norm tensor. This is what makes weight
  decay unnecessary for these layers.
- The subsequent division by `sqrt(fan_in)` cancels the multiplication in
  `normalize_weight`, producing a weight with per-row L2 norm `1/sqrt(fan_in)`
  in the actual convolution. That is exactly the "fan-in" LeCun init scale,
  enforced every step.
- **The learning-rate consequence.** Only the *direction* of each filter row
  updates — the norm is clamped back to `sqrt(fan_in)` every step. Empirically
  the effective LR is roughly `sqrt(fan_in)×` smaller than a standard net at
  the same nominal LR. Median `sqrt(fan_in)` at CIFAR-/AFHQ-scale channel
  counts is ≈ 15, which is why the ablation sets `lr=3e-3` for `karras`
  (`experiments/afhq_ablation/configs.py:96–100`), and why `2e-4` (the LR that
  works for `unet` and `uvit`) leaves `karras-*` stuck around FID 85+. See the
  discussion in `experiments/afhq_ablation/README.md:34–40`.

`Conv2d.__init__` has a `concat_ones_to_input` flag used only by the input
block (`karras_unet.py:1119`); the padded 1-channel of ones restores a bias-like
degree of freedom for the very first conv without breaking MP downstream.

### 2. `MPSiLU` (line 199)

```python
def forward(self, x):
    return F.silu(x) / 0.596
```

The `0.596` is `E_{x~N(0,1)}[SiLU(x)^2]`. Dividing by it keeps unit input
variance in → unit output variance out.

### 3. `MPAdd` (Eq. 88, line 316) and `MPCat` (Eq. 103, line 257)

For every residual and skip. Naive `a + b` inflates variance; MP variants
rescale.

```python
class MPAdd(Module):
    def forward(self, x, res):
        a, b, t = x, res, self.t                         # t is the residual weight
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den

class MPCat(Module):
    def forward(self, a, b):
        Na, Nb = a.shape[dim], b.shape[dim]
        C = sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))
        a = a * (1. - t) / sqrt(Na)
        b = b * t / sqrt(Nb)
        return C * torch.cat((a, b), dim=dim)
```

Paper defaults: `t=0.3` for residual `MPAdd` inside encoder/decoder/attention,
`t=0.5` for embedding addition and skip `MPCat`. `KarrasUnet.__init__` exposes
these as `mp_add_emb_t=0.5`, `attn_res_mp_add_t=0.3`, `resnet_mp_add_t=0.3`,
`mp_cat_t=0.5` (lines 1100–1103).

### 4. `PixelNorm` (Eq. 30, line 365)

```python
def forward(self, x):
    return l2norm(x, dim=self.dim, eps=self.eps) * sqrt(x.shape[self.dim])
```

L2-normalizes along a chosen dimension and then rescales by `sqrt(dim)` so the
output has unit variance in expectation. Used at the top of every `Encoder`
block (`karras_unet.py:698`), and on Q/K/V inside `Attention`
(`karras_unet.py:1016`).

### 5. `Gain` (line 225)

A single scalar parameter initialised to `0.`. Placed at the end of the output
block (`karras_unet.py:1122`) and inside every `to_emb` head
(`karras_unet.py:704, 848`). Starting at zero means the network output and the
per-block conditioning modulations begin as identity; the gains grow during
training as they become useful. This is the MP-friendly analogue of "zero-init"
tricks used elsewhere.

### 6. `MPFourierEmbedding` (line 576)

```python
class MPFourierEmbedding(Module):
    def __init__(self, dim):
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=False)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1) * sqrt(2)
```

The frequencies are random and *frozen*. `sqrt(2)` scaling keeps unit variance
in → unit variance out. The `x * weights * 2π` multiplication means this
embedding expects a *continuous log-SNR-like* signal, roughly `[-15, 15]`. It
aliases badly if you hand it an integer `t ∈ [0, 1000)`.

This is why `BackboneAdapter` inserts a log-SNR lookup table when running
`karras` under DDPM (`experiments/afhq_ablation/backbones.py:59–72`):

```python
def attach_log_snr(self, alphas_cumprod: torch.Tensor) -> None:
    ac = alphas_cumprod.detach().clamp(1e-8, 1 - 1e-8)
    self.log_snr_table = (ac.log() - (1 - ac).log()).float()
```

Under DDPM the raw integer `t` from the sampler becomes
`log_snr_table[t]`, giving `MPFourierEmbedding` the continuous signal it wants.
Under EDM the diffusion wrapper already feeds `c_noise(σ) = 0.25 log σ`, so no
mapping is needed.

## `Encoder` and `Decoder` blocks (lines 629–919)

The MP analogue of a ResNet block. Both share the same shape modulo which
direction they change resolution in.

`Encoder.forward` (line 730):

```python
if self.downsample:
    x = F.interpolate(x, (h // 2, w // 2), mode='bilinear')   # FIR-style downsample
    x = self.downsample_conv(x)

x = self.pixel_norm(x)          # unit variance in
res = x.clone()

x = self.block1(x)              # MPSiLU → Conv2d
if exists(emb):
    scale = self.to_emb(emb) + 1              # Gain-initialised → starts at 1
    x = x * rearrange(scale, 'b c -> b c 1 1')  # FiLM-style *scale* only, no shift
x = self.block2(x)              # MPSiLU → Dropout → Conv2d

x = self.res_mp_add(x, res)     # MPAdd instead of plain +
if exists(self.attn):
    x = self.attn(x)            # MP-attention with its own MPAdd
return x
```

Two things worth calling out:

- **FiLM-scale only.** Conditioning is `x * (scale + 1)`, no additive shift.
  Adding a shift on a magnitude-preserving pipeline would break the variance
  budget, so it's dropped. The `+ 1` and `Gain()` init to `0` together mean
  the block is exactly identity-modulated at init.
- **Bilinear resample instead of pixel-shuffle.** `F.interpolate(..., mode='bilinear')`
  is the paper's FIR filter; anti-alias-friendly and MP-consistent.

`Decoder` is the mirror image with `F.interpolate(..., mode='bilinear')` for
2× upsample (`karras_unet.py:900–902`) and `res_conv` between mismatched
channel counts (line 862).

## `Attention` (lines 923–1023)

```python
qkv = self.to_qkv(x).chunk(3, dim=1)
q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

q, k, v = map(self.pixel_norm, (q, k, v))    # PixelNorm on last dim → MP attention
out = self.attend(q, k, v)                    # `Attend`: SDPA or einsum fallback
out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
out = self.to_out(out)
return self.mp_add(out, res)                  # MPAdd residual
```

- The learnable `mem_kv` (line 981) prepend `num_mem_kv=4` global tokens per
  head, giving spatial queries an always-available global context without
  changing input dimensions.
- Softmax attention over `Q·K^T` is not exactly variance-preserving, so
  `PixelNorm` on Q/K/V is the compromise: it caps the scale of the dot
  products.
- `attn_flash=False` in the ablation (`experiments/afhq_ablation/backbones.py:98`):
  the einsum fallback works in fp32 during FID sampling on this box.

## Assembly (`KarrasUnet.__init__`, lines 1086–1223)

Symmetric encoder/decoder with a two-block middle (line 1217), skip stack
built by `prepend`-ing decoders as their encoder counterparts are `append`-ed
(lines 1178–1209). Every `Encoder` output goes on `skips` (line 1317); every
`Decoder` that flags `needs_skip=True` pops one and passes it through
`skip_mp_cat` (line 1329).

Ablation constructor call (from `backbones.py:92–100`):

```python
KarrasUnet(
    image_size=64, channels=3,
    self_condition=False,
    attn_flash=False,
    dim=72, dim_max=192, num_downsamples=4, num_blocks_per_stage=2,
    attn_res=(16, 8), attn_dim_head=32, dropout=0.0,
)
```

`dim_max=192` caps channel doubling at the third downsample; `num_downsamples=4`
takes 64→32→16→8→4. Attention runs at the 16 and 8 stages. `dropout=0.0`
overrides KarrasUnet's default `0.1` so the three ablation backbones share the
same regulariser (`configs.py:82–86`).

## Forward flow (`karras_unet.py:1234–1335`)

```python
time_emb = self.to_time_emb(time)              # MPFourierEmbedding → Linear
# (no class embedding in the ablation — num_classes=None)
emb = self.emb_activation(time_emb)            # MPSiLU

skips = [self.input_block(x)]                  # x_0 is a skip too
x = skips[-1]

for encoder in self.downs:
    x = encoder(x, emb=emb)
    skips.append(x)

for decoder in self.mids:
    x = decoder(x, emb=emb)

for decoder in self.ups:
    if decoder.needs_skip:
        x = self.skip_mp_cat(x, skips.pop())
    x = decoder(x, emb=emb)

return self.output_block(x)                    # Conv2d → Gain
```

`emb` gets FiLM-scaled into every encoder/decoder block through its `to_emb`
head (`Linear → Gain`). The `Gain()` starts at 0, so `scale = to_emb(emb) + 1`
starts at `1` for every block: conditioning is identity at init and grows
during training.

## Extras in the file (unused by the ablation)

- `MPFeedForward` (line 1339) and `MPImageTransformer` (line 1401): an MP
  transformer stack you can slot in as a decoder-side block. Not used here.
- `InvSqrtDecayLRSched` (line 1480): the paper's shipped `lr(t) = σ_ref /
  sqrt(max(t/t_ref, 1))` scheduler, peak ~1e-2. The ablation uses a constant
  `lr=3e-3` instead — see the discussion in
  `experiments/afhq_ablation/README.md:36–40` for why the matched-constant
  approximation was chosen over the schedule.

## How the ablation calls it

Under DDPM (`karras-ddpm`), the DDPM sampler hands integer `t ∈ [0, 1000)` to
`BackboneAdapter.forward`, which remaps it via `log_snr_table[t]` before
calling `KarrasUnet(x, log_snr)` (`backbones.py:70–72`).

Under EDM (`karras-edm`), the wrapper feeds `c_noise(σ) = 0.25 log σ`
directly, and the adapter's `log_snr_time` flag stays `False` so no remap
happens (`backbones.py:114`).

Either way the *only* thing the backbone sees is a continuous log-SNR-ish
scalar for `time`, which is what `MPFourierEmbedding` was designed to eat.
